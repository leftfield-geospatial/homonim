"""
    Homonim: Correction of aerial and satellite imagery to surface relfectance
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
import multiprocessing
import threading
from concurrent import futures
from contextlib import ExitStack, contextmanager
from typing import Tuple, Dict, Union
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.io import DatasetWriter
from rasterio.enums import Resampling
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from homonim import utils
from homonim.enums import Method, ProcCrs
from homonim.errors import IoError
from homonim.kernel_model import KernelModel, RefSpaceModel, SrcSpaceModel
from homonim.raster_array import RasterArray
from homonim.raster_pair import RasterPairReader, BlockPair

logger = logging.getLogger(__name__)


class RasterFuse(RasterPairReader):
    """
    Class for homogenising ('fusing') a source image with a reference image.
    """

    # default configuration dicts
    default_fuse_config = dict(param_image=False, threads=multiprocessing.cpu_count(), max_block_mem=100)
    # TODO: rename fuse_config and any other homo variables to keep with 'correction' terminology
    # TODO: rethink these config dicts... maybe something like cloup settings
    def __init__(
        self, src_filename: Union[Path, str], ref_filename: Union[Path, str], proc_crs: ProcCrs = ProcCrs.auto,
    ):
        """
        Create a RasterFuse class.

        Parameters
        ----------
        src_filename: str, Path
            Path to the source image file.
        ref_filename: str, Path
            Path to the reference image file.  The extents of this image should cover src_filename with at least a 2
            pixel boundary, and it should have at least as many bands as src_filename.  The ordering of the bands
            in src_filename and ref_filename should match.
        proc_crs: homonim.enums.ProcCrs, optional
            The initial proc_crs setting, specifying which of the source/reference image spaces should be used for
            parameter estimation .  If proc_crs=ProcCrs.auto (recommended), the lowest resolution image space will be
            used. [default: ProcCrs.auto]
        """
        RasterPairReader.__init__(self, src_filename, ref_filename, proc_crs=proc_crs)
        self._write_lock = threading.Lock()
        self._param_lock = threading.Lock()

    @staticmethod
    def create_out_profile(
        driver: str = 'GTiff',
        dtype: str = RasterArray.default_dtype,
        nodata: float = RasterArray.default_nodata,
        creation_options: Dict = dict(
            tiled=True, blockxsize=512, blockysize=512, compress='deflate', interleave='band', photometric=None
        )
    )->Dict: # yapf: disable
        """
        Utility method to create a rasterio image profile for the output image(s) that can be passed to
        :meth:`RasterFuse.__init__`.  Without arguments, the default profile values are returned.

        Parameters
        ----------
        driver: str, optional
            Output format driver.
        dtype: str, optional
            Output image data type.
        nodata: float, optional
            Output image nodata value.
        creation_options: dict, optional
            Driver specific creation options.  See the GDAL documentation for more information.

        Returns
        -------
        dict
            rasterio image profile.
        """
        return dict(driver=driver, dtype=dtype, nodata=nodata, creation_options=creation_options)


    def _merge_out_profile(self, out_profile: Dict=None) -> dict:
        """ Create an output image rasterio profile from the source image profile and output configuration. """
        out_profile = self.create_out_profile(**out_profile)
        out_profile = utils.combine_profiles(self.src_im.profile, out_profile)
        out_profile['count'] = len(self.src_bands)
        return out_profile

    def _merge_param_profile(self, out_profile: Dict=None) -> dict:
        """ Create a debug image rasterio profile from the proc_crs image and configuration. """
        if self.proc_crs == ProcCrs.ref:
            init_profile = self.ref_im.profile
        else:
            init_profile = self.src_im.profile
        param_profile = utils.combine_profiles(init_profile, out_profile)
        # force dtype and nodata to defaults
        param_profile.update(
            dtype=RasterArray.default_dtype, count=len(self.src_bands) * 3,
            nodata=RasterArray.default_nodata
        )
        return param_profile

    def _set_metadata(self, im: DatasetWriter, **kwargs):
        """ Helper function to copy the RasterFuse configuration info to an image (GeoTiff) file. """
        # TODO: rather than keep copies of e.g. method and kernel_shape, have them as properties of the actual class
        #  that uses them (KernelModel) in this case, and retrieve them via the e.g. self._kernel_model attr
        fuse_config = {f'FUSE_{k.upper()}' : v.name if hasattr(v, 'name') else v for k, v in kwargs.items()}
        meta_dict = dict(
            FUSE_SRC_FILE=self._src_filename.name, FUSE_REF_FILE=self._ref_filename.name,
            FUSE_PROC_CRS=self.proc_crs.name, **fuse_config,
        )
        im.update_tags(**meta_dict)

    def _set_homo_metadata(self, im: DatasetWriter, **kwargs):
        """
        Copy useful metadata to a corrected image (GeoTIFF) file.

        Parameters
        ----------
        im: rasterio.io.DatasetWriter
            An open rasterio dataset to write the metadata to.
        """
        if im.closed:
            raise errors.IoError(f'The raster dataset is closed: {im.name}')

        self._set_metadata(im, **kwargs)
        for bi in range(0, min(im.count, len(self.ref_bands))):
            ref_bi = self.ref_bands[bi]
            ref_meta_dict = self.ref_im.tags(ref_bi)
            homo_meta_dict = {k: v for k, v in ref_meta_dict.items() if k in ['ABBREV', 'ID', 'NAME']}
            im.set_band_description(bi + 1, self.ref_im.descriptions[ref_bi - 1])
            im.update_tags(bi + 1, **homo_meta_dict)

    def _set_param_metadata(self, im: DatasetWriter, **kwargs):
        """
        Copy useful metadata to a parameter image (GeoTIFF) dataset.

        Parameters
        ----------
        im: rasterio.io.DatasetWriter
            An open rasterio dataset to write the metadata to.
        """
        # Use reference file band descriptions to make debug image band descriptions
        if im.closed:
            raise errors.IoError(f'The raster dataset is closed: {im.name}')

        self._set_metadata(im, **kwargs)
        num_src_bands = len(self.src_bands)
        for bi in range(0, num_src_bands):
            ref_bi = self.ref_bands[bi]
            ref_descr = self.ref_im.descriptions[ref_bi - 1] or f'B{ref_bi}'
            ref_meta_dict = self.ref_im.tags(ref_bi)
            for param_i, param_name in zip(range(bi, im.count, num_src_bands), ['GAIN', 'OFFSET', 'R2']):
                im.set_band_description(param_i + 1, f'{ref_descr}_{param_name}')
                param_meta_dict = {
                    k: f'{v.upper()} {param_name}' for k, v in ref_meta_dict.items() if k in ['ABBREV', 'ID', 'NAME']
                }
                im.update_tags(param_i + 1, **param_meta_dict)

    @contextmanager
    def _out_files(
        self, out_filename:Union[Path, str], param_filename:Union[Path, str]=None, out_profile: Dict=None,
        overwrite: bool = False, build_ovw:bool = False, **kwargs
    ):
        if not overwrite and out_filename.exists():
            raise FileExistsError(
                f"Corrected image file exists and won't be overwritten without the `overwrite` option: "
                f"{out_filename}"
            )
        if not overwrite and param_filename and param_filename.exists():
            raise FileExistsError(
                f"Parameter image file exists and won't be overwritten without the `overwrite` option: "
                f"{param_filename}"
            )

        out_im = rio.open(out_filename, 'w', **self._merge_out_profile(out_profile))
        param_im = rio.open(param_filename, 'w', **self._merge_param_profile(out_profile)) if param_filename else None
        yield (out_im, param_im)

        self._set_homo_metadata(out_im, **kwargs)
        if build_ovw:
            self._build_overviews(out_im)
        out_im.close()
        if param_im:
            self._set_param_metadata(param_im, **kwargs)
            if build_ovw:
                self._build_overviews(param_im)
            param_im.close()


    @staticmethod
    def _build_overviews(im, max_num_levels=8, min_level_pixels=256):
        """
        Build internal overviews, downsampled by successive powers of 2, for a rasterio dataset.

        Parameters
        ----------
        im: rasterio.io.DatasetWriter
            An open rasterio dataset to write the metadata to.
        max_num_levels: int, optional
            Maximum number of overview levels to build.
        min_level_pixels: int, pixel
            Minimum width/height (in pixels) of any overview level.
        """

        # limit overviews so that the highest level has at least 2**8=256 pixels along the shortest dimension,
        # and so there are no more than 8 levels.
        max_ovw_levels = int(np.min(np.log2(im.shape)))
        min_level_shape_pow2 = int(np.log2(min_level_pixels))
        num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
        ovw_levels = [2 ** m for m in range(1, num_ovw_levels + 1)]
        im.build_overviews(ovw_levels, Resampling.average)


    def _process_block(
        self, block_pair: BlockPair, model: KernelModel, out_im: DatasetWriter, param_im: DatasetWriter = None,
    ):
        """ Thread-safe function to homogenise a source image block. """

        # read source and reference blocks
        src_ra, ref_ra = self.read(block_pair)
        # fit and apply the sliding kernel models
        param_ra = model.fit(ref_ra, src_ra)
        out_ra = model.apply(src_ra, param_ra)
        # change the output nodata value so that is is masked correctly for out_im
        out_ra.nodata = out_im.nodata

        with self._write_lock:  # write the output block
            out_ra.to_rio_dataset(out_im, indexes=block_pair.band_i + 1, window=block_pair.src_out_block)

        if param_im:
            with self._param_lock:  # write the parameter block
                indexes = np.arange(param_ra.count) * len(self.src_bands) + block_pair.band_i + 1
                param_out_block = (
                    # TODO: incorporate into BlockPair?
                    block_pair.ref_out_block if self.proc_crs == ProcCrs.ref else block_pair.src_out_block
                )
                param_ra.to_rio_dataset(param_im, indexes=indexes, window=param_out_block)

    def process(
        self, out_filename: Union[Path, str], method: Method, kernel_shape: Tuple[int, int],
        param_filename: Union[Path, str] = None, overwrite: bool = False, model_config: Dict = None,
        out_profile: Dict = None, fuse_config: Dict = None, build_ovw: bool = True,
    ):
        """
        Correct the source image in blocks.

        Parameters
        ----------
        out_filename: str, Path
            Path to the corrected file to create, or a directory in which in which to create an automatically named
            file.
        method: homonim.enums.Method
            The surface reflectance correction method.
        kernel_shape: tuple
            The (height, width) of the kernel in pixels of the proc_crs image (the lowest resolution image, if
            proc_crs=ProcCrs.auto).
        param_filename: str, Path, optional
            Path to an optional parameter file to write with correction parameters and R2 values.  By default,
            no parameter file is written.
        overwrite: bool, optional
            Overwrite the output file(s) if they exist. [default: True]
        fuse_config: dict, optional
            Surface reflectance correction configuration with items:
                param_image: bool
                    Turn on/off the production of a debug image containing correction parameters and R2 values.
                threads: int
                    The number of blocks process concurrently (requires more memory). 0 = use all cpus.
                max_block_mem: float
                    An upper limit on the image block size in MB.  Useful for limiting the memory used by each block-
                    processing thread. Note that this is not a limit on the total memory consumed by a thread, as a
                    number of working block-sized arrays are created, but is proportional to the total memory consumed
                    by a thread.
        model_config: dict, optional
            Radiometric modelling configuration dict with items:
                downsampling: rasterio.enums.Resampling, str
                    Resampling method for downsampling.
                upsampling: rasterio.enums.Resampling, str
                    Resampling method for upsampling.
                r2_inpaint_thresh: float
                    R2 threshold below which to inpaint model parameters from "surrounding areas.
                    For 'gain-offset' method only.
                mask_partial: bool
                    Mask corrected pixels produced from partial kernel or image coverage.
        out_profile: dict, optional
            Output image configuration dict with the items:
                driver: str
                    Output format driver.
                dtype: str
                    Output image data type.
                nodata: float
                    Output image nodata value.
                creation_options: dict
                    Driver specific creation options.  See the rasterio documentation for more information.
        build_ovw: bool
            Whether to build overviews.
        """
        # call out_filename something else, in unit tests too, fuse_filename, corr_filename...?
        # TODO: might we pass in a model, rather than its config
        # TODO: pass in param_image, threads, max_block_mem here too i.e. fuse_config?
        # TODO: can we make at least some of these kwargs to simplify passing from cli and to self._process_block
        self._assert_open()
        method = Method(method)
        kernel_shape = tuple(utils.validate_kernel_shape(kernel_shape, method=method))
        overlap = utils.overlap_for_kernel(kernel_shape)
        out_profile = self.create_out_profile(**(out_profile or {}))
        model_config = KernelModel.create_config(**(model_config or {}))
        fuse_config = fuse_config if fuse_config else self.default_fuse_config
        fuse_config['threads'] = utils.validate_threads(fuse_config['threads'])

        # create the KernelModel according to proc_crs
        model_cls = SrcSpaceModel if self.proc_crs == ProcCrs.src else RefSpaceModel
        self._model = model_cls(
            method=method, kernel_shape=kernel_shape, find_r2=fuse_config['param_image'], **model_config,
        )

        block_pair_args = dict(overlap=overlap, max_block_mem=fuse_config['max_block_mem'])
        # get block pairs and process
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'  # tqdm progress bar format
        with self._out_files(
            out_filename, param_filename=param_filename, out_profile=out_profile, overwrite=overwrite,
            build_ovw=build_ovw, method=method, kernel_shape=kernel_shape, **model_config, **fuse_config
        ) as (out_im, param_im):
            if fuse_config['threads'] == 1:
                # process blocks consecutively in the main thread (useful for profiling)
                block_pairs = [block_pair for block_pair in self.block_pairs(**block_pair_args)]
                for block_pair in tqdm(block_pairs, bar_format=bar_format):
                    self._process_block(block_pair, self._model, out_im=out_im, param_im=param_im)
            else:
                # process blocks concurrently
                with futures.ThreadPoolExecutor(max_workers=fuse_config['threads']) as executor:
                    # submit blocks for processing
                    proc_futures = [
                        executor.submit(self._process_block, block_pair, self._model, out_im, param_im)
                        for block_pair in self.block_pairs(**block_pair_args)
                    ]

                    # wait for threads in order of completion, and raise any thread generated exceptions
                    for future in tqdm(
                        futures.as_completed(proc_futures), bar_format=bar_format, total=len(proc_futures)
                    ): # yapf: disable
                        future.result()
