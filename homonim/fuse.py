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
from collections import namedtuple

import numpy as np
import rasterio as rio
from rasterio.warp import Resampling
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from homonim.kernel_model import KernelModel, RefSpaceModel, SrcSpaceModel
from homonim.raster_array import RasterArray
from homonim.raster_pair import RasterPairReader, BlockPair
from homonim.enums import Method, ProcCrs
from homonim import utils
from homonim import errors

logger = logging.getLogger(__name__)

"""Overlapping block object"""
OvlBlock = namedtuple('OvlBlock', ['band_i', 'src_in_block', 'src_out_block', 'outer'])


def _update_from_nested(to_dict, from_nested_dict):
    for key, value in from_nested_dict.items():
        if isinstance(value, dict):
            _update_from_nested(to_dict, value)
        elif value is not None:
            to_dict[key] = value
    return to_dict


class RasterFuse():
    default_homo_config = dict(debug_image=False, mask_partial=False, multithread=True, max_block_mem=100)
    default_out_profile = dict(driver='GTiff', dtype=RasterArray.default_dtype, nodata=RasterArray.default_nodata,
                               creation_options=dict(tiled=True, blockxsize=512, blockysize=512, compress='deflate',
                                                     interleave='band', photometric=None))
    default_model_config = KernelModel.default_config

    def __init__(self, src_filename, ref_filename, method, kernel_shape, proc_crs=ProcCrs.auto,
                 homo_config=default_homo_config, model_config=default_model_config, out_profile=default_out_profile):
        """
        Class for homogenising images

        Parameters
        ----------
        src_filename : str, pathlib.Path
            Source image filename.
        ref_filename: str, pathlib.Path
            Reference image filename.
        homo_config: HomoConfig, optional

        model_config: ModelConfig, optional

        out_profile: OutConfig, optional

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

        # Block overlap should be at least half the kernel 'shape' to ensure full kernel coverage at block edges, and a
        # minimum of (1, 1) to avoid including extrapolated (rather than interpolated) pixels when up-sampling.
        overlap = utils.overlap_for_kernel(self._kernel_shape)

        # check the ref and src images via RasterPairReader, and get the proc_crs attribute
        raster_pair_args = dict(proc_crs=proc_crs, overlap=overlap,
                                max_block_mem=self._config['max_block_mem'])
        self._raster_pair =  RasterPairReader(src_filename, ref_filename, **raster_pair_args)
        with self._raster_pair as raster_pair:
            self._proc_crs = raster_pair.proc_crs
            block_shape = raster_pair.block_shape
            if np.any(block_shape < self._kernel_shape):
                raise errors.BlockSizeError(
                    f"The block shape ({block_shape}) that satisfies the maximum block size "
                    f"({self._homo_config['max_block_mem']}MB) setting is too small to accommodate a kernel shape of "
                    f"({self._kernel_shape}). Increase 'max_block_mem' in 'homo_config', or decrease "
                    f"'kernel_shape'."
                )

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
        return self._method

    @property
    def kernel_shape(self):
        return self._kernel_shape

    def _profile_from(self, in_profile):
        """ create a raster profile by combining an input profile with the configuration profile """

        if in_profile['driver'].lower() != self._out_profile['driver'].lower():
            # only copy non driver specific keys from input profile when the driver is different to the configured val
            copy_keys = ['driver', 'width', 'height', 'count', 'dtype', 'crs', 'transform']
            out_profile = {copy_key: in_profile[copy_key] for copy_key in copy_keys}
        else:
            out_profile = in_profile.copy()

        def nested_update(self_dict, other_dict):
            """Update self_dict with a flattened version of other_dict"""
            for other_key, other_value in other_dict.items():
                if isinstance(other_value, dict):
                    # flatten the driver specific nested dict into the root dict
                    nested_update(self_dict, other_value)
                elif other_value is not None:
                    self_dict[other_key] = other_value
            return self_dict

        return nested_update(out_profile, self._out_profile)

    def _create_out_profile(self, raster_pair: RasterPairReader):
        """Create a rasterio profile for the output image based on a starting profile and configuration"""
        out_profile = self._profile_from(raster_pair.src_im.profile)
        out_profile['count'] = len(raster_pair.src_bands)
        return out_profile

    def _create_debug_profile(self, raster_pair: RasterPairReader):
        """Create a rasterio profile for the debug parameter image based on a reference or source profile"""
        init_profile = raster_pair.ref_im.profile if raster_pair.proc_crs == ProcCrs.ref else raster_pair.src_im.profile
        debug_profile = self._profile_from(init_profile)
        debug_profile.update(dtype=RasterArray.default_dtype, count=len(raster_pair.src_bands) * 3,
                             nodata=RasterArray.default_nodata)
        return debug_profile

    def _create_debug_filename(self, filename):
        """Return a debug image filename, given the homogenised image filename"""
        filename = pathlib.Path(filename)
        return filename.parent.joinpath(f'{filename.stem}_DEBUG{filename.suffix}')

    def build_overviews(self, filename):
        """
        Builds internal overviews for a existing image file.

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
            # and so there are no more than 8 levels
            max_ovw_levels = int(np.min(np.log2(im.shape)))
            num_ovw_levels = np.min([8, max_ovw_levels - 8])
            ovw_levels = [2 ** m for m in range(1, num_ovw_levels + 1)]
            im.build_overviews(ovw_levels, Resampling.average)

    def _set_homo_metadata(self, filename):
        """
        Copy various metadata to a homogenised image (GeoTIFF) file.

        Parameters
        ----------
        filename: str, pathlib.Path
                  Path to the GeoTIFF image file to copy metadata to.
        """
        filename = pathlib.Path(filename)
        meta_dict = dict(HOMO_SRC_FILE=self._src_filename.name, HOMO_REF_FILE=self._ref_filename.name,
                         HOMO_PROC_CRS=self._proc_crs.name, HOMO_METHOD=self._method.name,
                         HOMO_KERNEL_SHAPE=self._kernel_shape, HOMO_CONF=str(self._config),
                         HOMO_MODEL_CONF=str(self._model.config))

        if not filename.exists():
            raise FileNotFoundError(f'{filename} does not exist')

        with rio.open(self._ref_filename, 'r') as ref_im, rio.open(filename, 'r+') as homo_im:
            # Set user-supplied metadata
            homo_im.update_tags(**meta_dict)
            # Copy any geedim generated metadata from the reference file
            for bi in range(0, min(homo_im.count, ref_im.count)):
                ref_meta_dict = ref_im.tags(bi + 1)
                homo_meta_dict = {k: v for k, v in ref_meta_dict.items() if k in ['ABBREV', 'ID', 'NAME']}
                homo_im.set_band_description(bi + 1, ref_im.descriptions[bi])
                homo_im.update_tags(bi + 1, **homo_meta_dict)

    def _set_debug_metadata(self, filename):
        """
        Copy various metadata to a homogenised image (GeoTIFF) file.

        Parameters
        ----------
        filename: str, pathlib.Path
                  Path to the GeoTIFF image file to copy metadata to.
        """
        filename = pathlib.Path(filename)
        meta_dict = dict(HOMO_SRC_FILE=self._src_filename.name, HOMO_REF_FILE=self._ref_filename.name,
                         HOMO_SPACE=self._proc_crs.name, HOMO_METHOD=self._method,
                         HOMO_KERNEL_SHAPE=tuple(self._kernel_shape),
                         HOMO_CONF=str(self._config), HOMO_MODEL_CONF=str(self._model.config))

        if not filename.exists():
            raise FileNotFoundError(f'{filename} does not exist')

        with rio.open(self._ref_filename, 'r') as ref_im, rio.open(filename, 'r+') as dbg_im:
            # Set user-supplied metadata
            dbg_im.update_tags(**meta_dict)
            # Use reference file band descriptions to make debug image band descriptions
            num_src_bands = int(dbg_im.count / 3)
            for ri in range(0, num_src_bands):
                ref_descr = ref_im.descriptions[ri] or f'B{ri + 1}'
                ref_meta_dict = ref_im.tags(ri + 1)
                for pi, pname in zip(range(ri, dbg_im.count, num_src_bands), ['GAIN', 'OFFSET', 'R2']):
                    dbg_im.set_band_description(pi + 1, f'{ref_descr}_{pname}')
                    dbg_meta_dict = {k: f'{v.upper()} {pname}' for k, v in ref_meta_dict.items() if
                                     k in ['ABBREV', 'ID', 'NAME']}
                    dbg_im.update_tags(pi + 1, **dbg_meta_dict)

    def homogenise(self, out_filename):
        """
        Homogenise an image file by block.

        Parameters
        ----------
        out_filename: str, pathlib.Path
                      Path of the homogenised image file to create.
        """
        write_lock = threading.Lock()
        dbg_lock = threading.Lock()
        accum_stats = np.array([0., 0.])
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'

        with logging_redirect_tqdm(), self._raster_pair as raster_pair:
            # TODO: use im_pair src_win to create expanded profile?
            out_profile = self._create_out_profile(raster_pair)
            out_im = rio.open(out_filename, 'w', **out_profile)
            if self._profile:
                # setup profiling
                tracemalloc.start()
                proc_profile = cProfile.Profile()
                proc_profile.enable()

            if self._config['debug_image']:
                # create debug image file
                dbg_profile = self._create_debug_profile(raster_pair)
                dbg_file_name = self._create_debug_filename(out_filename)
                dbg_im = rio.open(dbg_file_name, 'w', **dbg_profile)

            try:
                def process_block(block_pair: BlockPair):
                    """Thread-safe function to homogenise a block of src_im"""
                    src_ra, ref_ra = raster_pair.read(block_pair)
                    param_ra = self._model.fit(ref_ra, src_ra)
                    mask_partial = block_pair.outer and self._config['mask_partial']
                    out_ra = self._model.apply(src_ra, param_ra, mask_partial=mask_partial)
                    out_ra.nodata = out_im.nodata

                    with write_lock:
                        out_ra.to_rio_dataset(out_im, indexes=block_pair.band_i + 1, window=block_pair.src_out_block)

                    if self._config['debug_image']:
                        with dbg_lock:
                            # nonlocal accum_stats
                            # accum_stats += [param_ra.array[2, param_mask].sum(), param_mask.sum()]
                            #
                            indexes = np.arange(param_ra.count) * len(raster_pair.src_bands) + block_pair.band_i + 1
                            if self._proc_crs == ProcCrs.ref:
                                dbg_out_block = block_pair.ref_out_block
                            else:
                                dbg_out_block = block_pair.src_out_block
                            param_ra.to_rio_dataset(dbg_im, indexes=indexes, window=dbg_out_block)

                if self._config['multithread']:
                    # process blocks in concurrent threads
                    future_list = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                        for block_pair in raster_pair.block_pairs():
                            future = executor.submit(process_block, block_pair)
                            future_list.append(future)

                        # wait for threads and raise any thread generated exceptions
                        for future in tqdm(future_list, bar_format=bar_format):
                            future.result()
                else:
                    # process bands consecutively
                    for block_pair in tqdm(raster_pair.block_pairs(), bar_format=bar_format):
                        process_block(block_pair)
            finally:
                out_im.close()
                self._set_homo_metadata(out_filename)
                if self._config['debug_image']:
                    dbg_im.close()
                    self._set_debug_metadata(dbg_file_name)
                    # logger.debug(
                    #     f'Average kernel model R\N{SUPERSCRIPT TWO}: {accum_stats[0] / accum_stats[1]:.2f}')

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
