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
import logging
import warnings
import cProfile
import concurrent.futures
import multiprocessing
import pathlib
import pstats
import threading
import tracemalloc
from collections import namedtuple
from itertools import product

import numpy as np
import rasterio as rio
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from homonim.kernel_model import KernelModel, RefSpaceModel, SrcSpaceModel
from homonim.raster_array import RasterArray, round_window_to_grid, expand_window_to_grid
from homonim.errors import (UnsupportedImageError, ImageContentError, BlockSizeError)

logger = logging.getLogger(__name__)


def _inspect_image(im_filename):
    im_filename = pathlib.Path(im_filename)
    if not im_filename.exists():
        raise FileNotFoundError(f'{im_filename.name} does not exist')

    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(im_filename, 'r') as im:
        try:
            tmp_array = im.read(1, window=im.block_window(1, 0, 0))
        except Exception as ex:
            if im.profile['compress'] == 'jpeg':  # assume it is a 12bit JPEG
                raise UnsupportedImageError(
                    f'Could not read image {im_filename.name}.  JPEG compression with NBITS==12 is unsupported, '
                    f'you probably need to recompress this image.'
                )
            else:
                raise ex
        is_masked = any([MaskFlags.all_valid not in im.mask_flag_enums[bi] for bi in range(im.count)])
        if im.nodata is None and not is_masked:
            logger.warning(f'{im_filename.name} has no mask or nodata value, '
                           f'any invalid pixels should be masked before homogenising.')
        im_bands = [bi + 1 for bi in range(im.count) if im.colorinterp[bi] != ColorInterp.alpha]
    return im_bands

def _inspect_image_pair(cmp_filename, ref_filename):
    # check ref_filename has enough bands
    cmp_bands = _inspect_image(cmp_filename)
    ref_bands = _inspect_image(ref_filename)
    if len(cmp_bands) > len(ref_bands):
        raise ImageContentError(f'{ref_filename.name} has fewer non-alpha bands than {cmp_filename.name}.')

    # warn if band counts don't match
    if len(cmp_bands) != len(ref_bands):
        logger.warning(f'Image non-alpha band counts don`t match. Using the first {len(cmp_bands)} non-alpha bands '
                         f'of {ref_filename.name}.')

    with rio.open(cmp_filename, 'r') as cmp_im:
        with WarpedVRT(rio.open(ref_filename, 'r'), crs=cmp_im.crs, resampling=Resampling.bilinear) as ref_im:
            # check coverage
            if not box(*ref_im.bounds).covers(box(*cmp_im.bounds)):
                raise ImageContentError('Reference extent does not cover image.')
    return cmp_bands, ref_bands

"""Overlapping block object"""
OvlBlock = namedtuple('OvlBlock', ['band_i', 'src_in_block', 'src_out_block', 'outer'])


class ImFuse():
    default_homo_config = dict(debug_image=False, mask_partial=False, multithread=True, max_block_mem=100)
    default_out_profile = dict(driver='GTiff', dtype=RasterArray.default_dtype, blockxsize=512, blockysize=512,
                               compress='deflate', interleave='band', photometric=None,
                               nodata=RasterArray.default_nodata)
    default_model_config = KernelModel.default_config

    def __init__(self, src_filename, ref_filename, method, kernel_shape, model_crs='auto',
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
        self._src_filename = pathlib.Path(src_filename)
        self._ref_filename = pathlib.Path(ref_filename)
        if not method in ['gain', 'gain-im-offset', 'gain-offset']:
            raise ValueError('method should be one of "gain", "gain-im-offset" or "gain-offset"')
        self._method = method
        if not np.all(np.mod(kernel_shape, 2) == 1):
            raise ValueError('kernel_shape must be odd in both dimensions')
        self._kernel_shape = np.array(kernel_shape).astype(int)
        self._config = homo_config
        self._out_profile = out_profile

        self._ref_bands = None
        self._src_bands = None
        self._ref_warped_vrt_dict = None
        self._profile = False
        self._model_crs = model_crs
        self._image_init()

        if self._model_crs == 'ref':
            self._model = RefSpaceModel(method=method, kernel_shape=kernel_shape,
                                        debug_image=self._config['debug_image'], **model_config)
        elif self._model_crs == 'src':
            self._model = SrcSpaceModel(method=method, kernel_shape=kernel_shape,
                                        debug_image=self._config['debug_image'], **model_config)
        else:
            raise ValueError(f'Unknown model_crs option: {model_crs}')

    @property
    def method(self):
        return self._method

    @property
    def kernel_shape(self):
        return self._kernel_shape

    @property
    def space(self):
        return self._model_crs

    def _image_init(self):
        """Check bounds, band count, and compression type of source and reference images"""
        self._src_bands, self._ref_bands = _inspect_image_pair(self._src_filename, self._ref_filename)

        with rio.open(self._src_filename, 'r') as src_im, rio.open(self._ref_filename, 'r') as _ref_im:
            if src_im.crs.to_proj4() != _ref_im.crs.to_proj4():  # re-project the reference image to source CRS
                logger.info(f'Re-projecting reference image to source CRS.')
            with WarpedVRT(_ref_im, crs=src_im.crs, resampling=Resampling.bilinear) as ref_im:
                src_pixel_smaller = np.prod(src_im.res) < np.prod(ref_im.res)
                if self._model_crs == 'auto':
                    self._model_crs = 'ref' if src_pixel_smaller else 'src'
                elif ((self._model_crs == 'src' and src_pixel_smaller) or
                      (self._model_crs == 'ref' and not src_pixel_smaller)):
                    logger.warning(f'model_crs == "{"ref" if src_pixel_smaller else "src"}" is recommended when '
                                  f'the source image pixel size is '
                                  f'{"smaller" if src_pixel_smaller else "larger"} than the reference.')

                ref_win = expand_window_to_grid(
                    ref_im.window(*src_im.bounds),
                    expand_pixels=np.ceil(np.divide(src_im.res, ref_im.res)).astype('int')
                )
                ref_transform = ref_im.window_transform(ref_win)
                self._ref_warped_vrt_dict = dict(crs=src_im.crs, transform=ref_transform, width=ref_win.width,
                                                 height=ref_win.height, resampling=Resampling.bilinear)

    def _auto_block_shape(self, src_shape):
        max_block_mem = self._config['max_block_mem'] * (2 ** 20)  # MB to Bytes
        dtype_size = np.dtype(RasterArray.default_dtype).itemsize

        block_shape = np.array(src_shape)
        while (np.product(block_shape) * dtype_size > max_block_mem):
            div_dim = np.argmax(block_shape)
            block_shape[div_dim] /= 2
        return np.round(block_shape).astype('int')

    def _create_ovl_blocks(self):
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._src_filename, 'r') as src_im:
            with WarpedVRT(rio.open(self._ref_filename, 'r'), **self._ref_warped_vrt_dict) as ref_im:
                src_shape = np.array(src_im.shape)
                # TODO deal with res_ratio < 1 (everywhere), and also src_kernel_shape even with model_crs='src'
                src_kernel_shape = np.ceil(self._kernel_shape * np.divide(ref_im.res, src_im.res)).astype(int)
                res_ratio = np.ceil(np.divide(ref_im.res, src_im.res)).astype(int)
                overlap = np.ceil(res_ratio + src_kernel_shape / 2).astype(int)
                ovl_blocks = []
                block_shape = self._auto_block_shape(src_im.shape)
                if np.any(block_shape <= src_kernel_shape):
                    raise BlockSizeError('Block size is less than kernel size, increase "max_block_mem" or decrease '
                                         '"kernel_shape"')

                for band_i in range(len(self._src_bands)):
                    for ul_row, ul_col in product(range(-overlap[0], (src_shape[0] - 2 * overlap[0]), block_shape[0]),
                                                  range(-overlap[1], (src_shape[1] - 2 * overlap[1]), block_shape[1])):
                        ul = np.array((ul_row, ul_col))
                        br = ul + block_shape + (2 * overlap)
                        # include a ref pixel beyond src boundary to allow ref-space reprojections there
                        # TODO rethink this, and the implications for src res > ref res
                        src_ul = np.fmax(ul, -res_ratio)
                        src_br = np.fmin(br, src_shape + res_ratio)
                        src_block_shape = np.subtract(src_br, src_ul)
                        outer = np.any(src_ul <= 0) or np.any(src_br >= src_shape)
                        out_ul = ul + overlap
                        out_br = br - overlap

                        src_in_block = Window.from_slices((src_ul[0], src_br[0]), (src_ul[1], src_br[1]),
                                                          width=src_block_shape[1], height=src_block_shape[0],
                                                          boundless=outer)
                        src_out_block = Window.from_slices((out_ul[0], out_br[0]), (out_ul[1], out_br[1]))

                        ovl_blocks.append(OvlBlock(band_i, src_in_block, src_out_block, outer))
        return ovl_blocks

    def _create_out_profile(self, init_profile):
        """Create a rasterio profile for the output image based on a starting profile and configuration"""
        out_profile = init_profile.copy()
        out_profile['count'] = len(self._src_bands)
        for key, value in self._out_profile.items():
            if value is not None:
                out_profile.update(**{key: value})
        out_profile.update(tiled=True)
        return out_profile

    def _create_debug_profile(self, src_profile, ref_profile):
        """Create a rasterio profile for the debug parameter image based on a reference or source profile"""
        if self._model_crs == 'ref':
            debug_profile = ref_profile.copy()
        else:
            debug_profile = src_profile.copy()

        for key, value in self._out_profile.items():
            if value is not None:
                debug_profile.update(**{key: value})
        debug_profile.update(dtype=RasterArray.default_dtype, count=len(self._src_bands) * 3,
                             nodata=RasterArray.default_nodata, tiled=True)
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
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'r+') as homo_im:
            homo_im.build_overviews([2, 4, 8, 16, 32], Resampling.average)

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
                         HOMO_SPACE=self._model_crs, HOMO_METHOD=self._method, HOMO_KERNEL_SHAPE=self._kernel_shape,
                         HOMO_CONF=str(self._config), HOMO_MODEL_CONF=str(self._model.config))

        if not filename.exists():
            raise FileNotFoundError(f'{filename} does not exist')

        with rio.open(self._ref_filename, 'r') as ref_im, rio.open(filename, 'r+') as homo_im:
            # Set user-supplied metadata
            homo_im.update_tags(**meta_dict)
            # Copy any geedim generated metadata from the reference file
            for bi in range(0, homo_im.count):
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
                         HOMO_SPACE=self._model_crs, HOMO_METHOD=self._method,
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
        ovl_blocks = self._create_ovl_blocks()
        bar = tqdm(total=len(ovl_blocks),
                   bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]')
        bar.update(0)

        src_read_lock = threading.Lock()
        ref_read_lock = threading.Lock()
        write_lock = threading.Lock()
        dbg_lock = threading.Lock()

        with logging_redirect_tqdm(), rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._src_filename, 'r') as src_im:
            with WarpedVRT(rio.open(self._ref_filename, 'r'), **self._ref_warped_vrt_dict) as ref_im:

                if self._profile:
                    # setup profiling
                    tracemalloc.start()
                    proc_profile = cProfile.Profile()
                    proc_profile.enable()

                if self._config['debug_image']:
                    # create debug image file
                    dbg_profile = self._create_debug_profile(src_im.profile, ref_im.profile)
                    dbg_file_name = self._create_debug_filename(out_filename)
                    dbg_im = rio.open(dbg_file_name, 'w', **dbg_profile)

                # create the output image file
                out_profile = self._create_out_profile(src_im.profile)
                out_im = rio.open(out_filename, 'w', **out_profile)  # avoid too many nested indents with 'with' statements
                r2_sum = 0.
                mask_sum = 0.

                try:
                    def process_block(ovl_block: OvlBlock):
                        """Thread-safe function to homogenise a block of src_im"""
                        with src_read_lock:
                            src_ra = RasterArray.from_rio_dataset(
                                src_im,
                                indexes=self._src_bands[ovl_block.band_i],
                                window=ovl_block.src_in_block,
                                boundless=ovl_block.outer
                            )

                        with ref_read_lock:
                            src_in_bounds = src_im.window_bounds(ovl_block.src_in_block)
                            # TODO: if src res >> ref res, ref_in_block can extend beyond ref limits
                            ref_in_block = round_window_to_grid(ref_im.window(*src_in_bounds))
                            ref_ra = RasterArray.from_rio_dataset(
                                ref_im,
                                indexes=self._ref_bands[ovl_block.band_i],
                                window=ref_in_block
                            )

                        param_ra = self._model.fit(ref_ra, src_ra)
                        out_ra = self._model.apply(src_ra, param_ra)
                        out_ra.mask = src_ra.mask
                        if ovl_block.outer and self._config['mask_partial']:
                            out_ra = self._model.mask_partial(out_ra, ref_ra.res)
                        out_ra.nodata = out_im.nodata

                        with write_lock:
                            out_array = out_ra.slice_array(*out_im.window_bounds(ovl_block.src_out_block))
                            out_im.write(out_array, window=ovl_block.src_out_block, indexes=ovl_block.band_i + 1)
                            bar.update(1)

                        if self._config['debug_image']:
                            param_mask = param_ra.mask
                            with dbg_lock:
                                nonlocal mask_sum, r2_sum
                                mask_sum += param_mask.sum()
                                r2_sum += param_ra.array[2, param_mask].sum()

                                src_out_bounds = src_im.window_bounds(ovl_block.src_out_block)
                                dbg_array = param_ra.slice_array(*src_out_bounds)
                                dbg_out_block = round_window_to_grid(dbg_im.window(*src_out_bounds))
                                indexes = np.arange(param_ra.count) * len(self._src_bands) + ovl_block.band_i + 1
                                dbg_im.write(dbg_array, window=dbg_out_block, indexes=indexes)

                    if self._config['multithread']:
                        # process blocks in concurrent threads
                        future_list = []
                        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                            for ovl_block in ovl_blocks:
                                future = executor.submit(process_block, ovl_block)
                                future_list.append(future)

                            # wait for threads and raise any thread generated exceptions
                            for future in future_list:
                                future.result()
                    else:
                        # process bands consecutively
                        for ovl_block in ovl_blocks:
                            process_block(ovl_block)
                finally:
                    out_im.close()
                    bar.close()
                    self._set_homo_metadata(out_filename)
                    if self._config['debug_image']:
                        dbg_im.close()
                        self._set_debug_metadata(dbg_file_name)
                        logger.debug(f'Average kernel model R\N{SUPERSCRIPT TWO}: {r2_sum/mask_sum:.2f}')

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
