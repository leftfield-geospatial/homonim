"""
    Homonim: Radiometric homogenisation of aerial and satellite imagery
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
import cProfile
import concurrent.futures
import logging
import multiprocessing
import pathlib
import pstats
import threading
import tracemalloc

import numpy as np
import rasterio as rio
import rasterio.io
from rasterio.warp import Resampling
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from contextlib import ExitStack

from homonim import errors
from homonim import utils
from homonim.enums import Method, ProcCrs
from homonim.kernel_model import KernelModel, RefSpaceModel, SrcSpaceModel
from homonim.raster_array import RasterArray
from homonim.raster_pair import RasterPairReader, BlockPair

logger = logging.getLogger(__name__)


class RasterFuse():
    """Class for radiometrically homogenising ('fusing') a source image with a reference image."""

    # default configuration dicts
    default_homo_config = dict(debug_image=False, multithread=True, max_block_mem=100)
    default_out_profile = dict(driver='GTiff', dtype=RasterArray.default_dtype, nodata=RasterArray.default_nodata,
                               creation_options=dict(tiled=True, blockxsize=512, blockysize=512, compress='deflate',
                                                     interleave='band', photometric=None))
    default_model_config = KernelModel.default_config

    def __init__(self, src_filename, ref_filename, method, kernel_shape, proc_crs=ProcCrs.auto,
                 homo_config=default_homo_config, model_config=default_model_config, out_profile=default_out_profile):
        """
        Create a RasterFuse class.

        Parameters
        ----------
        src_filename: str, pathlib.Path
            Path to the source image file.
        ref_filename: str, pathlib.Path
            Path to the reference image file.  The extents of this image should cover src_filename with at least a 2
            pixel boundary, and it should have at least as many bands as src_filename.  The ordering of the bands
            in src_filename and ref_filename should match.
        method: homonim.enums.Method
                The radiometric homogenisation method.
        kernel_shape: tuple
                The (height, width) of the kernel in pixels of the proc_crs image (the lowest resolution image, if
                proc_crs=ProcCrs.auto).
        proc_crs: homonim.enums.ProcCrs
            The initial proc_crs setting, specifying which of the source/reference image spaces should be used for
            processing.  If proc_crs=ProcCrs.auto (recommended), the lowest resolution image space will be used.
            [default: ProcCrs.auto]
        homo_config: dict
            General homogenisation configuration dict with items:
                debug_image: bool
                    Turn on/off the production of a debug image containing homogenisation parameters and R2 values.
                multithread: bool
                    Turn on/off the concurrent processing of image blocks.
                max_block_mem: float
                    An upper limit on the image block size in MB.  Useful for limiting the memory used by each block-
                    processing thread. Note that this is not a limit on the total memory consumed by a thread, as a
                    number of working block-sized arrays are created, but is proportional to the total memory consumed
                    by a thread.
        model_config: dict
            Radiometric modelling configuration dict with items:
                downsampling: rasterio.enums.Resampling, str
                    Resampling method for downsampling.
                upsampling: rasterio.enums.Resampling, str
                    Resampling method for upsampling.
                r2_inpaint_thresh: float
                    R2 threshold below which to inpaint model parameters from "surrounding areas.
                    For 'gain-offset' method only.
                mask_partial: bool
                    Mask homogenised pixels produced from partial kernel or image coverage.
        out_profile: dict
            Output image configuration dict with the items:
                driver: str
                    Output format driver.
                dtype: str
                    Output image data type.
                nodata: float
                    Output image nodata value.
                creation_options: dict
                    Driver specific creation options.  See the rasterio documentation for more information.
        """
        if not isinstance(proc_crs, ProcCrs):
            raise ValueError("'proc_crs' must be an instance of homonim.enums.ProcCrs")
        if not isinstance(method, Method):
            raise ValueError("'method' must be an instance of homonim.enums.Method")
        self._method = method

        self._kernel_shape = utils.check_kernel_shape(kernel_shape)

        self._src_filename = pathlib.Path(src_filename)
        self._ref_filename = pathlib.Path(ref_filename)
        self._config = homo_config
        self._out_profile = out_profile
        self._profile = False

        # get the block overlap for kernel_shape
        overlap = utils.overlap_for_kernel(self._kernel_shape)

        # check the reference and source image validity via RasterPairReader, and get the proc_crs attribute
        raster_pair_args = dict(proc_crs=proc_crs, overlap=overlap, max_block_mem=self._config['max_block_mem'])
        self._raster_pair = RasterPairReader(src_filename, ref_filename, **raster_pair_args)
        with self._raster_pair as raster_pair:
            self._proc_crs = raster_pair.proc_crs
            block_shape = raster_pair.block_shape
            # raise an exception if the block shape is smaller than a kernel
            if np.any(block_shape < self._kernel_shape):
                raise errors.BlockSizeError(
                    f"The block shape ({block_shape}) that satisfies the maximum block size "
                    f"({self._homo_config['max_block_mem']}MB) is too small to accommodate a kernel shape of "
                    f"({self._kernel_shape}). Increase 'max_block_mem' in 'homo_config', or decrease "
                    f"'kernel_shape'."
                )

        # create the KernelModel according to proc_crs
        if self._proc_crs == ProcCrs.ref:
            self._model = RefSpaceModel(method=method, kernel_shape=kernel_shape,
                                        debug_image=self._config['debug_image'], **model_config)
        elif self._proc_crs == ProcCrs.src:
            self._model = SrcSpaceModel(method=method, kernel_shape=kernel_shape,
                                        debug_image=self._config['debug_image'], **model_config)
        else:
            raise ValueError(f'Invalid proc_crs: {proc_crs}, should be resolved to ProcCrs.ref or ProcCrs.src')

    @property
    def method(self):
        """Homogenisation method."""
        return self._method

    @property
    def kernel_shape(self):
        """Kernel shape."""
        return self._kernel_shape

    def _combine_with_config(self, in_profile):
        """Update an input rasterio profile with the configuration profile."""

        if in_profile['driver'].lower() != self._out_profile['driver'].lower():
            # copy only non driver specific keys from input profile when the driver is different to the configured val
            copy_keys = ['driver', 'width', 'height', 'count', 'dtype', 'crs', 'transform']
            out_profile = {copy_key: in_profile[copy_key] for copy_key in copy_keys}
        else:
            out_profile = in_profile.copy()     # copy the whole input profile

        def nested_update(self_dict, other_dict):
            """Update self_dict with a flattened version of other_dict"""
            for other_key, other_value in other_dict.items():
                if isinstance(other_value, dict):
                    # flatten the driver specific nested dict into the root dict
                    nested_update(self_dict, other_value)
                elif other_value is not None:
                    self_dict[other_key] = other_value
            return self_dict

        # update out_profile with a flattened self._out_profile
        return nested_update(out_profile, self._out_profile)

    def _create_out_profile(self):
        """Create an output image rasterio profile from the source image profile and output configuration"""
        out_profile = self._combine_with_config(self._raster_pair.src_im.profile)
        out_profile['count'] = len(self._raster_pair.src_bands)
        return out_profile

    def _create_debug_profile(self):
        """Create a debug image rasterio profile from the proc_crs image and configuration"""
        if self._raster_pair.proc_crs == ProcCrs.ref:
            init_profile = self._raster_pair.ref_im.profile
        else:
            init_profile = self._raster_pair.src_im.profile
        debug_profile = self._combine_with_config(init_profile)
        debug_profile.update(dtype=RasterArray.default_dtype, count=len(self._raster_pair.src_bands) * 3,
                             nodata=RasterArray.default_nodata)
        return debug_profile

    def _debug_filename(self, filename):
        """Create a debug image filename, given the homogenised image filename"""
        filename = pathlib.Path(filename)
        return filename.parent.joinpath(f'{filename.stem}_DEBUG{filename.suffix}')

    def _set_metatdata(self, filename):
        """Helper function to copy the RasterFuse configuration info to an image (GeoTiff) file."""
        filename = pathlib.Path(filename)
        meta_dict = dict(HOMO_SRC_FILE=self._src_filename.name, HOMO_REF_FILE=self._ref_filename.name,
                         HOMO_PROC_CRS=self._proc_crs.name, HOMO_METHOD=self._method.name,
                         HOMO_KERNEL_SHAPE=self._kernel_shape, HOMO_CONF=str(self._config),
                         HOMO_MODEL_CONF=str(self._model.config))

        if not filename.exists():
            raise FileNotFoundError(f'{filename} does not exist')

        with rio.open(filename, 'r+') as im:
            im.update_tags(**meta_dict)

    def _set_homo_metadata(self, filename):
        """
        Copy useful metadata to a homogenised image (GeoTIFF) file.

        Parameters
        ----------
        filename: str, pathlib.Path
                  Path to the GeoTIFF image file to copy metadata to.
        """
        filename = pathlib.Path(filename)
        self._set_metatdata(filename)

        # Copy any band metadata (like geedim band names etc.) from the reference file
        with rio.open(self._ref_filename, 'r') as ref_im, rio.open(filename, 'r+') as homo_im:
            for bi in range(0, min(homo_im.count, ref_im.count)):
                ref_meta_dict = ref_im.tags(bi + 1)
                homo_meta_dict = {k: v for k, v in ref_meta_dict.items() if k in ['ABBREV', 'ID', 'NAME']}
                homo_im.set_band_description(bi + 1, ref_im.descriptions[bi])
                homo_im.update_tags(bi + 1, **homo_meta_dict)

    def _set_debug_metadata(self, filename):
        """
        Copy useful metadata to the debug image (GeoTIFF) file.

        Parameters
        ----------
        filename: str, pathlib.Path
                  Path to the image file to copy metadata to.
        """
        filename = pathlib.Path(filename)
        self._set_metatdata(filename)

        with rio.open(self._ref_filename, 'r') as ref_im, rio.open(filename, 'r+') as dbg_im:
            # Use reference file band descriptions to make debug image band descriptions
            num_src_bands = int(dbg_im.count / 3)
            for ri in range(0, num_src_bands):
                ref_descr = ref_im.descriptions[ri] or f'B{ri + 1}'
                ref_meta_dict = ref_im.tags(ri + 1)
                for param_i, param_name in zip(range(ri, dbg_im.count, num_src_bands), ['GAIN', 'OFFSET', 'R2']):
                    dbg_im.set_band_description(param_i + 1, f'{ref_descr}_{param_name}')
                    dbg_meta_dict = {k: f'{v.upper()} {param_name}' for k, v in ref_meta_dict.items() if
                                     k in ['ABBREV', 'ID', 'NAME']}
                    dbg_im.update_tags(param_i + 1, **dbg_meta_dict)

    def _write_debug_block(self, block_pair:BlockPair, param_ra:RasterArray, dbg_im:rasterio.io.DatasetWriter):
        """Write a block of parameter data to a debug image"""
        # nonlocal accum_stats
        # accum_stats += [param_ra.array[2, param_mask].sum(), param_mask.sum()]
        #
        indexes = np.arange(param_ra.count) * len(self._raster_pair.src_bands) + block_pair.band_i + 1
        if self._proc_crs == ProcCrs.ref:
            dbg_out_block = block_pair.ref_out_block
        else:
            dbg_out_block = block_pair.src_out_block
        param_ra.to_rio_dataset(dbg_im, indexes=indexes, window=dbg_out_block)

    def build_overviews(self, filename):
        """
        Builds internal overviews for an existing image file.

        Parameters
        ----------
        filename: str, pathlib.Path
                  Path to the image file to build overviews for.
        """
        filename = pathlib.Path(filename)

        if not filename.exists():
            raise Exception(f'{filename} does not exist')
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'r+') as im:
            # limit overviews so that the highest level has at least 2**8=256 pixels along the shortest dimension,
            # and so there are no more than 8 levels.
            max_ovw_levels = int(np.min(np.log2(im.shape)))
            num_ovw_levels = np.min([8, max_ovw_levels - 8])
            ovw_levels = [2 ** m for m in range(1, num_ovw_levels + 1)]
            im.build_overviews(ovw_levels, Resampling.average)


    def homogenise(self, out_filename):
        """
        Homogenise the source image in blocks.

        Parameters
        ----------
        out_filename: str, pathlib.Path
                      Path of the homogenised image file to create.
        """
        write_lock = threading.Lock()
        dbg_lock = threading.Lock()
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'
        dbg_filename = self._debug_filename(out_filename)

        with ExitStack() as stack:
            # redirect console logging to tqdm.write()
            stack.enter_context(logging_redirect_tqdm())
            # open the source and reference images for reading
            stack.enter_context(self._raster_pair)
            # open the output image for writing
            out_im = stack.enter_context(rio.open(out_filename, 'w', **self._create_out_profile()))

            if self._config['debug_image']: # open the debug image for writing
                dbg_im = stack.enter_context(rio.open(dbg_filename, 'w', **self._create_debug_profile()))

            if self._profile:
                # setup profiling
                tracemalloc.start()
                proc_profile = cProfile.Profile()
                proc_profile.enable()

            def process_block(block_pair: BlockPair):
                """Thread-safe function to homogenise a source image block"""
                # read source and reference blocks
                src_ra, ref_ra = self._raster_pair.read(block_pair)
                # fit and apply the sliding kernel models
                param_ra = self._model.fit(ref_ra, src_ra)
                out_ra = self._model.apply(src_ra, param_ra)
                # change the output nodata value if necessary
                out_ra.nodata = out_im.nodata

                with write_lock:
                    out_ra.to_rio_dataset(out_im, indexes=block_pair.band_i + 1, window=block_pair.src_out_block)

                if self._config['debug_image']:
                    with dbg_lock:
                        self._write_debug_block(block_pair, param_ra, dbg_im)

            if self._config['multithread']:
                # process blocks in concurrent threads
                future_list = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    for block_pair in self._raster_pair.block_pairs():
                        future = executor.submit(process_block, block_pair)
                        future_list.append(future)

                    # wait for threads and raise any thread generated exceptions
                    for future in tqdm(future_list, bar_format=bar_format):
                        future.result()
            else:
                # process blocks consecutively
                for block_pair in tqdm(self._raster_pair.block_pairs(), bar_format=bar_format):
                    process_block(block_pair)

        self._set_homo_metadata(out_filename)
        if self._config['debug_image']:
            self._set_debug_metadata(dbg_filename)

        if self._profile:
            # print profiling info
            # (tottime is the total time spent in the function alone. cumtime is the total time spent in the function
            # plus all functions that this function called)
            proc_profile.disable()
            proc_stats = pstats.Stats(proc_profile).sort_stats('cumtime')
            logger.debug(f'Processing time:')
            proc_stats.print_stats(20)

            current, peak = tracemalloc.get_traced_memory()
            logger.debug(f"Memory usage: current: {current / 10 ** 6:.1f} MB, peak: {peak / 10 ** 6:.1f} MB")
