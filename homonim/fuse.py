"""
    Homonim: Correction of aerial and satellite imagery to surface reflectance
    Copyright (C) 2021 Dugal Harris
    Email: dugalh@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging
import threading
from concurrent import futures
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Dict, Union, Iterator, Optional

import numpy as np
import rasterio
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.io import DatasetWriter
from tqdm.auto import tqdm

from homonim import utils
from homonim.enums import Model, ProcCrs
from homonim.errors import IoError
from homonim.kernel_model import KernelModel, RefSpaceModel, SrcSpaceModel
from homonim.raster_array import RasterArray
from homonim.raster_pair import RasterPairReader, BlockPair
from homonim.matched_pair import MatchedPairReader

logger = logging.getLogger(__name__)


class RasterFuse(MatchedPairReader):

    def __init__(self, *args, **kwargs):
        """
        Class for correcting an image to surface reflectance, by fusion with a reference.

        For best results, reference and source image(s) should be concurrent, co-located and spectrally similar.
        Reference image extents must encompass those of the source image.

        The reference image should contain bands that are approximate (wavelength) matches to the source image bands.
        Where source and reference images are RGB, or have ``center_wavelength`` metadata, bands are matched
        automatically based on wavelength.  Where there are the same number of source and reference bands, and no
        ``center_wavelength`` metadata, bands are assumed to be in matching order.  Subsets and ordering of source
        and reference bands can be specified with the ``src_bands`` and ``ref_bands`` parameters.

        .. note::

            Images downloaded with `geedim <https://github.com/dugalh/geedim>`_ have ``center_wavelength`` metadata
            compatible with ``homonim``.

        Parameters
        ----------
        src_filename: str, Path
            Path to a source image file.
        ref_filename: str, Path
            Path to a reference image file.
        proc_crs: homonim.enums.ProcCrs, optional
            :class:`~homonim.enums.ProcCrs` instance specifying which of the source/reference image CRS and pixel
            grid to use for processing.  For most use cases, it can be left as the default of
            :attr:`~homonim.enums.ProcCrs.auto` i.e. the lowest resolution of the source and reference image CRS's.
        src_bands: list of int, optional.
            Indexes of source spectral bands to be processed (1 based).  If not specified, all bands with the
            ``center_wavelength`` property, or all non-alpha bands, are used.
        ref_bands: list of int, optional.
            Indexes of reference spectral bands to match and fuse with source bands (1 based).  Should contain at
            least as many elements as ``src_bands``, or the number of valid bands in the source image file,
            if ``src_bands`` is not specified.  If ``ref_bands`` is not specified, all reference bands with the
            ``center_wavelength`` property, or all non-alpha bands are used.
        force: bool, optional
            Bypass auto wavelength matching, and any band-matching errors.  Use with caution.
        """
        MatchedPairReader.__init__(self, *args, **kwargs)
        self._corr_lock = threading.Lock()
        self._param_lock = threading.Lock()

    create_model_config = KernelModel.create_config

    @staticmethod
    def create_block_config(threads: int = 0, max_block_mem: float = 100) -> Dict:
        """
        Utility method to create a block processing configuration dictionary that can be passed to
        :meth:`RasterFuse.process`.  Without arguments, the default configuration is returned.

        Parameters
        ----------
        threads: int, optional
            Number of image blocks to process concurrently.  A maximum of the number of processors on your
            system is allowed.  Increasing this number will increase the memory required for processing.
            0 = use all processors.
        max_block_mem: float, optional
            Maximum size of an image block in megabytes. Note that the total memory consumed by a thread is
            proportional to, but larger than this number.

        Returns
        -------
        dict
            Block processing configuration dictionary.
        """
        return dict(threads=utils.validate_threads(threads), max_block_mem=max_block_mem)

    @staticmethod
    def create_out_profile(
        driver: str = 'GTiff',
        dtype: str = RasterArray.default_dtype,
        nodata: float = RasterArray.default_nodata,
        creation_options: Optional[Dict] = None
    ) -> Dict:  # yapf: disable
        """
        Utility method to create a profile for the output image(s) that can be passed to :meth:`RasterFuse.process`.
        Without arguments, the default profile is returned.

        Parameters
        ----------
        driver: str, optional
            Output format driver.  See the `GDAL docs <https://gdal.org/drivers/raster/index.html>`_ for
            available options.
        dtype: str, optional
            Output image data type.  One of: uint8|uint16|int16|uint32|int32|float32|float64.
        nodata: float, optional
            Output image nodata value.
        creation_options: dict, optional
            Driver specific creation options e.g. ``dict(compression='deflate')``.
            See the `GDAL docs <https://gdal.org/drivers/raster/index.html>`_ for available keys and values.

        Returns
        -------
        dict
            Output image profile.
        """
        # TODO: add compress_overview='auto' when gdal >=3.6
        creation_options = creation_options or dict(
            tiled=True, blockxsize=512, blockysize=512, compress='deflate', interleave='band', photometric='minisblack',
            bigtiff='if_safer',
        )
        return dict(driver=driver, dtype=dtype, nodata=nodata, creation_options=creation_options)

    @staticmethod
    def _build_overviews(im: DatasetWriter, max_num_levels: int = 8, min_level_pixels: int = 256):
        """
        Build internal overviews, downsampled by successive powers of 2, for an open rasterio dataset.
        Overviews are limited so that the highest level has at least ``min_level_pixels`` pixels along the shortest
        dimension, and so there are no more than ``max_num_levels`` levels.
        """
        if im.closed:
            raise IoError(f'The raster dataset is closed: {im.name}')

        max_ovw_levels = int(np.min(np.log2(im.shape)))
        min_level_shape_pow2 = int(np.log2(min_level_pixels))
        num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
        ovw_levels = [2 ** m for m in range(1, num_ovw_levels + 1)]
        im.build_overviews(ovw_levels, Resampling.average)

    def _merge_corr_profile(self, out_profile: Optional[Dict] = None) -> Dict:
        """
        Return a rasterio profile for the corrected image, by merging the source image profile with ``out_profile``.
        """
        out_profile = self.create_out_profile(**(out_profile or {}))
        corr_profile = utils.combine_profiles(self.src_im.profile, out_profile)
        corr_profile['count'] = len(self.src_bands)
        return corr_profile

    def _merge_param_profile(self, out_profile: Optional[Dict] = None) -> Dict:
        """
        Create a rasterio profile for the parameter image, using a merge of the ``proc_crs`` image profile,
        and ``out_profile`` as a starting point.
        """
        if self.proc_crs == ProcCrs.ref:
            init_profile = self.ref_im.profile
        else:
            init_profile = self.src_im.profile
        out_profile = self.create_out_profile(**(out_profile or {}))
        param_profile = utils.combine_profiles(init_profile, out_profile)
        # force dtype and nodata to defaults
        param_profile.update(
            dtype=RasterArray.default_dtype, count=len(self.src_bands) * 3, nodata=RasterArray.default_nodata
        )
        return param_profile

    def _set_metadata(self, im: DatasetWriter, **kwargs):
        """
        Utility method to convert ``**kwargs``, and RasterPairReader attributes to configuration metadata in an
        open rasterio dataset.
        """
        if im.closed:
            raise IoError(f'The raster dataset is closed: {im.name}')

        kwargs_meta_dict = {f'FUSE_{k.upper()}': v.name if hasattr(v, 'name') else v for k, v in kwargs.items()}
        meta_dict = dict(
            FUSE_SRC_FILE=self._src_filename.name, FUSE_REF_FILE=self._ref_filename.name,
            FUSE_PROC_CRS=self.proc_crs.name, **kwargs_meta_dict,
        )
        im.update_tags(**meta_dict)

    def _set_corr_metadata(self, im: DatasetWriter, **kwargs):
        """
        Utility method to convert ``**kwargs`` configuration and reference band info to metadata in a corrected
        image.
        """
        if im.closed:
            raise IoError(f'The raster dataset is closed: {im.name}')

        self._set_metadata(im, **kwargs)
        for bi in range(0, min(im.count, len(self.ref_bands))):
            ref_bi = self.ref_bands[bi]
            ref_meta_dict = self.ref_im.tags(ref_bi)
            geedim_meta_keys = ['center_wavelength', 'name', 'description', 'offset', 'scale']
            # copy geedim metadata from reference if the keys do not already exist in corrected image
            corr_meta_dict = {
                k: v for k, v in ref_meta_dict.items()
                if (k in geedim_meta_keys) and (k not in im.tags(bi + 1))
            }  # yapf. disable
            im.update_tags(bi + 1, **corr_meta_dict)
            # copy description from reference if the corrected file does not have one already
            if im.descriptions[bi] is None:
                im.set_band_description(bi + 1, self.ref_im.descriptions[ref_bi - 1])

    def _set_param_metadata(self, im: DatasetWriter, **kwargs):
        """
        Utility method to convert ``**kwargs`` to configuration metadata ino a parameter image, and set the band
        metadata to describe the corresponding parameter.
        """
        if im.closed:
            raise IoError(f'The raster dataset is closed: {im.name}')

        self._set_metadata(im, **kwargs)
        num_src_bands = len(self.src_bands)
        for bi in range(0, num_src_bands):
            ref_bi = self.ref_bands[bi]
            ref_descr = self.ref_im.descriptions[ref_bi - 1] or f'B{ref_bi}'
            ref_meta_dict = self.ref_im.tags(ref_bi)
            param_names = ['GAIN', 'OFFSET', 'R2']
            for param_i, param_name in zip(range(bi, im.count, num_src_bands), param_names):
                im.set_band_description(param_i + 1, f'{ref_descr}_{param_name}')
                param_meta_dict = {
                    k: f'{v.upper()} {param_name}' for k, v in ref_meta_dict.items() if k in ['ABBREV', 'ID', 'NAME']
                }
                im.update_tags(param_i + 1, **param_meta_dict)

    @contextmanager
    def _out_files(
        self, corr_filename: Union[Path, str], param_filename: Union[Path, str] = None, out_profile: Dict = None,
        overwrite: bool = False, build_ovw: bool = False, **kwargs
    ) -> Iterator[Tuple[rasterio.DatasetReader, Union[rasterio.DatasetReader, None]]]:
        """
        Internal context manager to handle the corrected, and optional parameter, output file(s).

        On entry, the image files are configured and created using ``out_profile``.
        On exit, image metadata is set with ``**kwargs``, overviews are built if ``build_ovw`` is True, and the file(s)
        are closed.

        Existing files are not overwritten unless ``overwrite`` is True.
        """
        # entry
        if not overwrite and corr_filename.exists():
            raise FileExistsError(
                f"Corrected image file exists and won't be overwritten without the `overwrite` option: {corr_filename}"
            )
        if not overwrite and param_filename and param_filename.exists():
            raise FileExistsError(
                f"Parameter image file exists and won't be overwritten without the `overwrite` option: {param_filename}"
            )
        # the images below will be opened in the RasterPairReader context, with its rasterio environment i.e. we
        # don't need to enter another environment context here
        out_im = rio.open(corr_filename, 'w', **self._merge_corr_profile(out_profile))
        param_im = rio.open(param_filename, 'w', **self._merge_param_profile(out_profile)) if param_filename else None
        try:
            yield out_im, param_im
        finally:
            # exit
            self._set_corr_metadata(out_im, **kwargs)
            if build_ovw:
                self._build_overviews(out_im)
            out_im.close()
            if param_im:
                self._set_param_metadata(param_im, **kwargs)
                if build_ovw:
                    self._build_overviews(param_im)
                param_im.close()

    def _process_block(
        self, block_pair: BlockPair, model: KernelModel, corr_im: DatasetWriter,
        param_im: Optional[DatasetWriter] = None,
    ):
        """
        Thread-safe method to correct an image block to surface reflectance using the supplied correction ``model``.
        Corrected, and optionally parameter, blocks are written to the supplied image dataset(s).
        """
        # read source and reference blocks
        src_ra, ref_ra = self.read(block_pair)
        # fit and apply the sliding kernel models
        param_ra = model.fit(src_ra, ref_ra)
        corr_ra = model.apply(src_ra, param_ra)
        # change the corrected nodata value so that is masked correctly for corr_im
        corr_ra.nodata = corr_im.nodata
        with self._corr_lock:  # write the corrected block
            corr_ra.to_rio_dataset(corr_im, indexes=block_pair.band_i + 1, window=block_pair.src_out_block)

        if param_im:
            with self._param_lock:  # write the parameter block
                indexes = np.arange(param_ra.count) * len(self.src_bands) + block_pair.band_i + 1
                param_out_block = (
                    block_pair.ref_out_block if self.proc_crs == ProcCrs.ref else block_pair.src_out_block
                )
                param_ra.to_rio_dataset(param_im, indexes=indexes, window=param_out_block)

    def process(
        self,
        corr_filename: Union[Path, str],
        model: Model = KernelModel.default_model,
        kernel_shape: Tuple[int, int] = KernelModel.default_kernel_shape,
        param_filename: Optional[Union[Path, str]] = None,
        build_ovw: bool = True,
        overwrite: bool = False,
        model_config: Optional[Dict] = None,
        out_profile: Optional[Dict] = None,
        block_config: Optional[Dict] = None,
    ):  # yapf: disable
        r"""
        Correct source image to surface reflectance by fusion with a reference.

        To improve speed and reduce memory usage, images are divided into blocks for concurrent processing.

        Parameters
        ----------
        corr_filename: str, Path
            Path to the corrected file to create.
        model: homonim.enums.Model, optional
            Correction model to use.  See :class:`~homonim.enums.Model` for options.
        kernel_shape: tuple of int, optional
            (height, width) of the kernel in pixels of the :attr:`proc_crs` image (the lowest resolution
            image, if :attr:`proc_crs` = :attr:`~homonim.enums.ProcCrs.auto`).  Larger kernels are less
            susceptible to over-fitting on noisy data, but provide lower resolution correction.
        param_filename: str, Path, optional
            Path to an optional parameter file to write with correction parameters and *R*\ :sup:`2` values.  By
            default, no parameter file is written.
        build_ovw: bool, optional
            Build overviews for the output image(s).
        overwrite: bool, optional
            Overwrite the output image(s) if they exist.
        model_config: dict, optional
            Configuration dictionary for the correction model.  See :meth:`create_model_config` for keys and default
            values.
        out_profile: dict, optional
            Configuration dictionary for the output image(s).   See :meth:`~RasterFuse.create_out_profile` for keys and
            default values.
        block_config: dict, optional
            Configuration dictionary for block processing.  See :meth:`~RasterFuse.create_block_config` for keys and
            default values.
        """
        self._assert_open()

        # prepare configuration
        model_type = Model(model)
        # kernel_shape = tuple(utils.validate_kernel_shape(kernel_shape, model=model))
        overlap = utils.overlap_for_kernel(kernel_shape)
        model_config = RasterFuse.create_model_config(**(model_config or {}))
        block_config = RasterFuse.create_block_config(**(block_config or {}))

        # create the KernelModel according to proc_crs
        model_cls = SrcSpaceModel if self.proc_crs == ProcCrs.src else RefSpaceModel
        model = model_cls(model, kernel_shape, find_r2=param_filename is not None, **model_config)

        # arguments to self.block_pairs()
        block_pair_args = dict(overlap=overlap, max_block_mem=block_config['max_block_mem'])
        # tqdm progress bar format
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'

        # create and open the output files
        with self._out_files(
            corr_filename, param_filename=param_filename, out_profile=out_profile, overwrite=overwrite,
            build_ovw=build_ovw, model=model_type, kernel_shape=kernel_shape, **model_config, **block_config
        ) as (out_im, param_im):  # yapf: disable
            if block_config['threads'] == 1:
                # correct blocks consecutively in the main thread (useful for profiling)
                block_pairs = [block_pair for block_pair in self.block_pairs(**block_pair_args)]
                for block_pair in tqdm(block_pairs, bar_format=bar_format):
                    self._process_block(block_pair, model, corr_im=out_im, param_im=param_im)
            else:
                # correct blocks concurrently
                with futures.ThreadPoolExecutor(max_workers=block_config['threads']) as executor:
                    # submit block correction jobs to the thread pool
                    proc_futures = [
                        executor.submit(self._process_block, block_pair, model, out_im, param_im)
                        for block_pair in self.block_pairs(**block_pair_args)
                    ]

                    # wait for threads in order of completion, and raise any thread generated exceptions
                    for future in tqdm(
                        futures.as_completed(proc_futures), bar_format=bar_format, total=len(proc_futures),
                        dynamic_ncols=True,
                    ):  # yapf: disable
                        future.result()
