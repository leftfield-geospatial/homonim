# Copyright Leftfield Geospatial
#
# This file is part of Homonim.
#
# Homonim is free software: you can redistribute it and/or modify it under the terms
# of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# Homonim is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with
# Homonim. If not, see <https://www.gnu.org/licenses/>.

import logging
import os
import threading
import warnings
from collections.abc import Iterator
from concurrent import futures
from contextlib import contextmanager
from os import PathLike, fspath
from pathlib import Path
from typing import Any

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
from homonim.matched_pair import MatchedPairReader
from homonim.raster_array import RasterArray
from homonim.raster_pair import BlockPair

logger = logging.getLogger(__name__)


class RasterFuse(MatchedPairReader):
    def __init__(
        self,
        src_filename: str | PathLike,
        ref_filename: str | PathLike,
        proc_crs: str | ProcCrs = ProcCrs.auto,
        src_bands: tuple[int, ...] | None = None,
        ref_bands: tuple[int, ...] | None = None,
        force: bool = False,
    ):
        """
        Class to correct a source image to surface reflectance by fusion with a
        reference.

        For best results, source and reference images should be concurrent.
        Reference extents must encompass those of the source.

        The reference should contain bands that are approximate wavelength matches to
        the source bands.  When source and reference bands are RGB, or have
        ``center_wavelength`` tags, bands are matched automatically based on
        wavelength.  Otherwise, source and reference bands are assumed to be in
        matching order.  Subsets and ordering of bands to use can be specified with
        the ``src_bands`` and ``ref_bands`` parameters.

        :param src_filename:
            Path or URI of a source image.
        :param ref_filename:
            Path or URI of a reference image.
        :param proc_crs:
            Which of the source or reference CRS and pixel grids should be used for
            estimating correction parameters.  By default, the CRS and pixel grid with
            the lowest resolution is used (recommended).
        :param src_bands:
            Indexes of source bands to be corrected (1 based).  Defaults to all bands
            with a ``center_wavelength`` tag if any exist, otherwise to all non-alpha
            bands.
        :param ref_bands:
            Indexes of reference bands to match with source bands (1 based).
            Defaults to all bands with a ``center_wavelength`` tag if any exist,
            otherwise to all non-alpha bands.
        :param force:
            Whether to bypass wavelength band matching.
        """
        # TODO: 'processing' or 'estimating correction parameters'.  here and
        #  elsewhere for proc_crs
        super().__init__(
            src_filename,
            ref_filename,
            proc_crs=proc_crs,
            src_bands=src_bands,
            ref_bands=ref_bands,
            force=force,
        )
        self._corr_lock = threading.Lock()
        self._param_lock = threading.Lock()

    @staticmethod
    def create_model_config(
        r2_inpaint_thresh: float = 0.25,
        mask_partial: bool = False,
        downsampling: Resampling = Resampling.average,
        upsampling: Resampling = Resampling.cubic_spline,
    ) -> dict[str, Any]:
        """
        Return a model configuration that can be passed as the ``model_config``
        argument to :meth:`~RasterFuse.process`.

        .. deprecated:: 0.5.0

            This method is deprecated and will be removed in a future release. Please
            pass the arguments to :meth:`~RasterFuse.process` directly.

        :param r2_inpaint_thresh:
            R\N{SUPERSCRIPT TWO} (coefficient of determination) threshold below which to
            interpolate ("in-paint") model offsets from surrounding values.  Applies
            to the :attr:`~enums.Model.gain_offset` model only.  If ``None``, no
            interpolation is performed.
        :param mask_partial:
            Whether to mask corrected pixels not produced by full kernel or source /
            reference image coverage.  Can help reduce seam-lines between overlapping
            images.
        :param downsampling:
             Resampling method to use when downsampling.
        :param upsampling:
            Resampling method to use when upsampling.

        :return:
            Model configuration.
        """
        warnings.warn(
            'This method is deprecated and will be removed in a future release. '
            "Please pass the arguments to 'RasterFuse.process()' directly.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return dict(
            r2_inpaint_thresh=r2_inpaint_thresh,
            mask_partial=mask_partial,
            downsampling=downsampling,
            upsampling=upsampling,
        )

    @staticmethod
    def create_block_config(
        threads: int = 0, max_block_mem: float = 100
    ) -> dict[str, Any]:
        """
        Return a block processing configuration dictionary that can be passed as the
        ``block_config`` argument to :meth:`~RasterFuse.process`.

        .. deprecated:: 0.5.0

            This method is deprecated and will be removed in a future release. Please
            pass the arguments to :meth:`~RasterFuse.process` directly.

        :param threads:
            Number of image blocks to process concurrently.  ``0`` will use the
            number of CPUs.
        :param max_block_mem:
            Maximum size of an image block in megabytes.

        :return:
            Configuration dictionary.
        """
        warnings.warn(
            'This method is deprecated and will be removed in a future release. '
            "Please pass the arguments to 'RasterFuse.process()' directly.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return dict(
            threads=utils.validate_threads(threads), max_block_mem=max_block_mem
        )

    @staticmethod
    def create_out_profile(
        driver: str = 'GTiff',
        dtype: str = RasterArray.default_dtype,
        nodata: float = RasterArray.default_nodata,
        creation_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Return a profile for the output image(s) that can be passed as the
        ``out_profile`` argument to :meth:`~RasterFuse.process`.

        .. deprecated:: 0.5.0

            This method is deprecated and will be removed in a future release. Please
            pass the arguments to :meth:`~RasterFuse.process` directly.

        :param driver:
            Format driver.  See the `GDAL docs
            <https://gdal.org/en/stable/drivers/raster/index.html>`__ for available
            options.
        :param dtype:
            Data type (``uint8``, ``uint16``, ``int16``, ``uint32``, ``int32``,
            ``float32`` or ``float64``).
        :param nodata:
            Nodata value.  If ``None``, an internal mask is written (recommended when
            ``creation_options`` are configured for lossy, e.g. JPEG, compression).
        :param creation_options:
             Driver specific creation options as a dictionary of ``name: value``
             pairs.  See the `GDAL docs
             <https://gdal.org/en/stable/drivers/raster/index.html>`__ corresponding
             to ``driver`` for available options.  If ``None``, default options are
             set when ``driver`` is ``GTiff``, otherwise no defaults are set.

        :return:
            Profile dictionary.
        """
        warnings.warn(
            'This method is deprecated and will be removed in a future release. '
            "Please pass the arguments to 'RasterFuse.process()' directly.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        # TODO: consider limiting driver to GTiff and COG, like in oty, and setting
        #  defaults for both.  i don't think things like building overviews, nodata
        #  or copying color_interp would be supported for all drivers.
        if driver.lower() == 'GTiff':
            default_creation_options = dict(
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress='deflate',
                interleave='band',
                photometric='minisblack',
                bigtiff='if_safer',
            )
        else:
            default_creation_options = {}

        creation_options = creation_options or default_creation_options
        return dict(
            driver=driver, dtype=dtype, nodata=nodata, creation_options=creation_options
        )

    @staticmethod
    def _build_overviews(
        im: DatasetWriter, max_num_levels: int = 8, min_level_pixels: int = 256
    ):
        """Build internal overviews for an open rasterio dataset.  Each overview
        level is decimated by a factor of 2.  The number of overview levels is
        determined by whichever of the ``max_num_levels`` or ``min_level_pixels``
        limits is reached first.
        """
        if im.closed:
            raise IoError(f'The raster dataset is closed: {im.name}')

        max_ovw_levels = int(np.min(np.log2(im.shape)))
        min_level_shape_pow2 = int(np.log2(min_level_pixels))
        num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
        ovw_levels = [2**m for m in range(1, num_ovw_levels + 1)]
        im.build_overviews(ovw_levels, Resampling.average)

    def _merge_corr_profile(
        self, out_profile: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return a rasterio profile for the corrected image, by merging the source
        image profile with ``out_profile``.
        """
        # TODO: see if we can leave this out, and just use the source crs, transform,
        #  width & height, and count as below.  you could end up with some v weird
        #  and unexpected results, like combining e.g. source jpeg creation options
        #  with out_profile deflate creation options.  maybe copying source
        #  color_interp is legit, like in oty.
        out_profile = self.create_out_profile(**(out_profile or {}))
        corr_profile = utils.combine_profiles(self.src_im.profile, out_profile)
        corr_profile['count'] = len(self.src_bands)
        return corr_profile

    def _merge_param_profile(
        self, out_profile: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return a rasterio profile for the parameter image, using a merge of the
        ``proc_crs`` image profile, and ``out_profile`` as a starting point.
        """
        if self.proc_crs == ProcCrs.ref:
            init_profile = self.ref_im.profile
        else:
            init_profile = self.src_im.profile
        out_profile = self.create_out_profile(**(out_profile or {}))
        # TODO: if out_profile contains e.g. compression options only compatible with
        #  the out_profile dtype, this will not work.  maybe fix parameter profiles
        #  to a certain driver & compression.
        param_profile = utils.combine_profiles(init_profile, out_profile)
        # force dtype and nodata to defaults
        param_profile.update(
            dtype=RasterArray.default_dtype,
            count=len(self.src_bands) * 3,
            nodata=RasterArray.default_nodata,
        )
        return param_profile

    def _set_metadata(self, im: DatasetWriter, **kwargs):
        """Convert ``kwargs``, and RasterPairReader attributes to configuration
        metadata in an open rasterio dataset.
        """
        if im.closed:
            raise IoError(f'The raster dataset is closed: {im.name}')

        kwargs_meta_dict = {
            f'FUSE_{k.upper()}': v.name if hasattr(v, 'name') else v
            for k, v in kwargs.items()
        }
        src_name = Path(self._src_filename).name
        ref_name = Path(self._ref_filename).name
        meta_dict = dict(
            FUSE_SRC_FILE=src_name,
            FUSE_REF_FILE=ref_name,
            FUSE_PROC_CRS=self.proc_crs.name,
            **kwargs_meta_dict,
        )
        im.update_tags(**meta_dict)

    def _set_corr_metadata(self, im: DatasetWriter, **kwargs):
        """Convert ``kwargs`` and reference band info to metadata in a corrected
        image.
        """
        if im.closed:
            raise IoError(f'The raster dataset is closed: {im.name}')

        self._set_metadata(im, **kwargs)
        for bi in range(0, min(im.count, len(self.ref_bands))):
            ref_bi = self.ref_bands[bi]
            ref_meta_dict = self.ref_im.tags(ref_bi)
            geedim_meta_keys = [
                'center_wavelength',
                'name',
                'description',
                'offset',
                'scale',
            ]
            # copy geedim metadata from reference if the keys do not already exist in
            # corrected image
            corr_meta_dict = {
                k: v
                for k, v in ref_meta_dict.items()
                if (k in geedim_meta_keys) and (k not in im.tags(bi + 1))
            }
            im.update_tags(bi + 1, **corr_meta_dict)
            # copy description from reference if the corrected file does not have one
            # already
            if im.descriptions[bi] is None:
                im.set_band_description(bi + 1, self.ref_im.descriptions[ref_bi - 1])

    def _set_param_metadata(self, im: DatasetWriter, **kwargs):
        """Convert ``kwargs`` to configuration metadata in a parameter image,
        and set band metadata to describe the corresponding parameter.
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
            for param_i, param_name in zip(
                range(bi, im.count, num_src_bands), param_names, strict=True
            ):
                im.set_band_description(param_i + 1, f'{ref_descr}_{param_name}')
                param_meta_dict = {
                    k: f'{v.upper()} {param_name}'
                    for k, v in ref_meta_dict.items()
                    if k in ['ABBREV', 'ID', 'NAME']
                }
                im.update_tags(param_i + 1, **param_meta_dict)

    @contextmanager
    def _out_files(
        self,
        corr_filename: str | PathLike,
        param_filename: str | PathLike | None = None,
        out_profile: dict[str, Any] | None = None,
        overwrite: bool = False,
        build_ovw: bool = False,
        **kwargs,
    ) -> Iterator[tuple[rasterio.DatasetReader, rasterio.DatasetReader | None]]:
        """Context manager to handle the corrected, and optional parameter output
        file(s).

        On entry, the image files are configured and created using ``out_profile``.
        On exit, image metadata is set with ``kwargs``, overviews are built if
        ``build_ovw`` is True, and the file(s) are closed.

        Existing files are not overwritten unless ``overwrite`` is True.
        """
        # entry
        # TODO: this does not work with URIs
        if not overwrite and Path(corr_filename).exists():
            raise FileExistsError(
                f"Corrected image file exists and won't be overwritten without the "
                f"'overwrite' option: '{fspath(corr_filename)}'"
            )
        if not overwrite and param_filename and Path(param_filename).exists():
            raise FileExistsError(
                f"Parameter image file exists and won't be overwritten without the "
                f"'overwrite' option: '{fspath(param_filename)}'"
            )
        # the images below will be opened in the RasterPairReader context, with its
        # rasterio environment i.e. we don't need to enter another environment
        # context here
        out_im = rio.open(corr_filename, 'w', **self._merge_corr_profile(out_profile))
        param_im = (
            rio.open(param_filename, 'w', **self._merge_param_profile(out_profile))
            if param_filename
            else None
        )
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
        self,
        block_pair: BlockPair,
        model: KernelModel,
        corr_im: DatasetWriter,
        param_im: DatasetWriter | None = None,
    ):
        """Thread-safe method to correct an image block to surface reflectance using
        ``model``.  Corrected, and optionally parameter, blocks are written to the
        supplied image dataset(s).
        """
        # read source and reference blocks
        src_ra, ref_ra = self.read(block_pair)
        # fit and apply the sliding kernel models
        param_ra = model.fit(src_ra, ref_ra)
        corr_ra = model.apply(src_ra, param_ra)

        # write the corrected block
        with self._corr_lock:
            corr_ra.to_rio_dataset(
                corr_im, indexes=block_pair.band_i + 1, window=block_pair.src_out_block
            )

        if param_im:
            with self._param_lock:  # write the parameter block
                indexes = (
                    np.arange(param_ra.count) * len(self.src_bands)
                    + block_pair.band_i
                    + 1
                )
                param_out_block = (
                    block_pair.ref_out_block
                    if self.proc_crs == ProcCrs.ref
                    else block_pair.src_out_block
                )
                param_ra.to_rio_dataset(
                    param_im, indexes=indexes, window=param_out_block
                )

    def process(
        self,
        corr_filename: str | PathLike,
        model: Model = KernelModel.default_model,
        kernel_shape: tuple[int, int] = KernelModel.default_kernel_shape,
        param_filename: str | PathLike | None = None,
        build_ovw: bool = True,
        overwrite: bool = False,
        model_config: dict[str, Any] | None = None,
        out_profile: dict[str, Any] | None = None,
        block_config: dict[str, Any] | None = None,
        *,
        r2_inpaint_thresh: float = 0.25,
        mask_partial: bool = False,
        downsampling: Resampling = Resampling.average,
        upsampling: Resampling = Resampling.cubic_spline,
        driver: str = 'GTiff',
        dtype: str = RasterArray.default_dtype,
        nodata: float = RasterArray.default_nodata,
        creation_options: dict[str, Any] | None = None,
        threads: int = 0,
        max_block_mem: float = 100,
    ):
        """
        Correct the source image to surface reflectance.

        TODO: note the default format of the corrected file, including ordering of
        bands.

        :param corr_filename:
            Path or URI of the corrected image.
        :param model:
            Correction model type to use.
        :param kernel_shape:
            Kernel (height, width) in pixels of the :attr:`proc_crs` image.
        :param param_filename:
            Optional path or URI of an image to write with correction parameters and
            their R\N{SUPERSCRIPT TWO} values.  If ``None`` (the default), no parameter
            image is written.
        :param build_ovw:
            Whether to build overviews for the output image(s).
        :param overwrite:
            Whether to overwrite output image(s) if they exist.
        :param model_config:
            Correction model configuration as returned by :meth:`create_model_config`.

            .. deprecated:: 0.5.0

                This parameter will be removed in a future release.  Please pass the
                :meth:`create_model_config` arguments to this method directly.

        :param out_profile:
            Profile for the output image(s) as returned by :meth:`create_out_profile`.

            .. deprecated:: 0.5.0

                This parameter will be removed in a future release.  Please pass the
                :meth:`create_out_profile` arguments to this method directly.

        :param block_config:
            Block processing configuration as returned by :meth:`create_block_config`.

            .. deprecated:: 0.5.0

                This parameter will be removed in a future release.  Please pass the
                :meth:`create_block_config` arguments to this method directly.

        :param r2_inpaint_thresh:
            R\N{SUPERSCRIPT TWO} (coefficient of determination) threshold below which to
            interpolate ("in-paint") model offsets from surrounding values.  Applies
            to the :attr:`~enums.Model.gain_offset` model only.  If ``None``, no
            interpolation is performed.
        :param mask_partial:
            Whether to mask corrected pixels not produced by full kernel or source /
            reference image coverage.  Can help reduce seam-lines between overlapping
            images.
        :param downsampling:
             Resampling method to use when downsampling.
        :param upsampling:
            Resampling method to use when upsampling.
        :param driver:
            Corrected image format driver.  See the `GDAL docs
            <https://gdal.org/en/stable/drivers/raster/index.html>`__ for available
            options.
        :param dtype:
            Corrected image data type (``uint8``, ``uint16``, ``int16``, ``uint32``,
            ``int32``, ``float32`` or ``float64``).
        :param nodata:
            Corrected image nodata value.  If ``None``, an internal mask is written
            (recommended when ``creation_options`` are configured for lossy,
            e.g. JPEG, compression).
        :param creation_options:
             Driver specific creation options for the corrected image as a dictionary
             of ``name: value`` pairs.  See the `GDAL docs
             <https://gdal.org/en/stable/drivers/raster/index.html>`__ corresponding
             to ``driver`` for available options.  If ``None``, default options are
             set when ``driver`` is ``GTiff``, otherwise no defaults are set.
        :param threads:
            Number of image blocks to process concurrently.  ``0`` will use the
            number of CPUs.
        :param max_block_mem:
            Maximum size of an image block in megabytes.
        """
        # TODO: is it possible to have an auto block_config that adjusts threads and
        #  block mem to available memory
        self._assert_open()

        # prepare configuration
        model_type = Model(model)
        # kernel_shape = tuple(utils.validate_kernel_shape(kernel_shape, model=model))
        overlap = utils.overlap_for_kernel(kernel_shape)
        warn_msg = (
            "The '{}' parameter is deprecated and will be removed in a future "
            'release. Please pass its items as keyword arguments to '
            "'RasterFuse.process()' directly."
        )
        if model_config:
            warnings.warn(
                warn_msg.format('model_config'),
                category=DeprecationWarning,
                stacklevel=2,
            )
            model_config = self.create_model_config(**(model_config or {}))
        else:
            model_config = dict(
                r2_inpaint_thresh=r2_inpaint_thresh,
                mask_partial=mask_partial,
                downsampling=downsampling,
                upsampling=upsampling,
            )
        if out_profile:
            warnings.warn(
                warn_msg.format('out_profile'),
                category=DeprecationWarning,
                stacklevel=2,
            )
        else:
            out_profile = dict(
                driver=driver,
                dtype=dtype,
                nodata=nodata,
                creation_options=creation_options,
            )
        if block_config:
            warnings.warn(
                warn_msg.format('block_config'),
                category=DeprecationWarning,
                stacklevel=2,
            )
            block_config = self.create_block_config(**(block_config or {}))
        else:
            threads = threads or os.cpu_count()
            block_config = dict(threads=threads, max_block_mem=max_block_mem)

        # create the KernelModel according to proc_crs
        model_cls = SrcSpaceModel if self.proc_crs == ProcCrs.src else RefSpaceModel
        model = model_cls(
            model, kernel_shape, find_r2=param_filename is not None, **model_config
        )

        # arguments to self.block_pairs()
        block_pair_args = dict(
            overlap=overlap, max_block_mem=block_config['max_block_mem']
        )
        # tqdm progress bar format
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'

        # create and open the output files
        with self._out_files(
            corr_filename,
            param_filename=param_filename,
            out_profile=out_profile,
            overwrite=overwrite,
            build_ovw=build_ovw,
            model=model_type,
            kernel_shape=kernel_shape,
            **model_config,
            **block_config,
        ) as (out_im, param_im):
            if block_config['threads'] == 1:
                # correct blocks consecutively in the main thread (useful for profiling)
                block_pairs = [
                    block_pair for block_pair in self.block_pairs(**block_pair_args)
                ]
                for block_pair in tqdm(block_pairs, bar_format=bar_format):
                    self._process_block(
                        block_pair, model, corr_im=out_im, param_im=param_im
                    )
            else:
                # correct blocks concurrently
                with futures.ThreadPoolExecutor(
                    max_workers=block_config['threads']
                ) as executor:
                    # submit block correction jobs to the thread pool
                    proc_futures = [
                        executor.submit(
                            self._process_block, block_pair, model, out_im, param_im
                        )
                        for block_pair in self.block_pairs(**block_pair_args)
                    ]

                    # wait for threads in order of completion, and raise any thread
                    # generated exceptions
                    for future in tqdm(
                        futures.as_completed(proc_futures),
                        bar_format=bar_format,
                        total=len(proc_futures),
                        dynamic_ncols=True,
                    ):
                        future.result()
