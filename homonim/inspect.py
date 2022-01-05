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
import pathlib
import threading
from collections import namedtuple
from itertools import product

import numpy as np
import rasterio as rio
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds
from rasterio.windows import Window
from shapely.geometry import box

from homonim.errors import UnsupportedImageError, ImageContentError, BlockSizeError, IoError
from homonim.raster_array import RasterArray, expand_window_to_grid, round_window_to_grid

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
                           f'any invalid pixels should be masked before processing.')
        im_bands = [bi + 1 for bi in range(im.count) if im.colorinterp[bi] != ColorInterp.alpha]
    return im_bands


def _inspect_image_pair(src_filename, ref_filename, proc_crs='auto'):
    # check ref_filename has enough bands
    cmp_bands = _inspect_image(src_filename)
    ref_bands = _inspect_image(ref_filename)
    if len(cmp_bands) > len(ref_bands):
        raise ImageContentError(f'{ref_filename.name} has fewer non-alpha bands than {src_filename.name}.')

    # warn if band counts don't match
    if len(cmp_bands) != len(ref_bands):
        logger.warning(f'Image non-alpha band counts don`t match. Using the first {len(cmp_bands)} non-alpha bands '
                       f'of {ref_filename.name}.')

    with rio.open(src_filename, 'r') as cmp_im:
        with WarpedVRT(rio.open(ref_filename, 'r'), crs=cmp_im.crs, resampling=Resampling.bilinear) as ref_im:
            # check coverage
            if not box(*ref_im.bounds).covers(box(*cmp_im.bounds)):
                raise ImageContentError('Reference extent does not cover image.')

            src_pixel_smaller = np.prod(cmp_im.res) < np.prod(ref_im.res)
            cmp_str = "smaller" if src_pixel_smaller else "larger"
            if proc_crs == 'auto':
                proc_crs = 'ref' if src_pixel_smaller else 'src'
                logger.debug(f'Source pixel size is {cmp_str} than the reference, '
                             f'using model_crs="{proc_crs}"')
            elif ((proc_crs == 'src' and src_pixel_smaller) or
                  (proc_crs == 'ref' and not src_pixel_smaller)):
                rec_crs_str = "ref" if src_pixel_smaller else "src"
                logger.warning(f'model_crs="{rec_crs_str}" is recommended when '
                               f'the source image pixel size is {cmp_str} than the reference.')

    return cmp_bands, ref_bands, proc_crs

"""Overlapping block object"""
BlockPair = namedtuple('BlockPair', ['band_i', 'src_in_block', 'ref_in_block', 'src_out_block', 'ref_out_block', 'outer'])

class ImPairBlockReader():
    def __init__(self, src_filename, ref_filename, proc_crs='auto', overlap=(0,0), max_block_mem=100):
        self._src_filename = pathlib.Path(src_filename)
        self._ref_filename = pathlib.Path(ref_filename)
        self._overlap = overlap if not np.isscalar(overlap) else (overlap, overlap)
        self._max_block_mem = max_block_mem
        self._src_bands, self._ref_bands, self._proc_crs = _inspect_image_pair(self._src_filename, self._ref_filename,
                                                                               proc_crs)
        self._src_im = None
        self._ref_im = None
        self._env = None
        self._ref_lock = threading.Lock()
        self._src_lock = threading.Lock()

    def __enter__(self):
        self._env = rio.Env(GDAL_NUM_THREADS='ALL_CPUs').__enter__()
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if self._env:
            self._env.__exit__(exc_type, exc_val, exc_tb)
            self._env = None

    def open(self):
        # self._blocks = self._create_ovl_blocks(overlap=self._overlap)
        with self._src_lock:
            self._src_im = rio.open(self._src_filename, 'r')
        with self._ref_lock:
            self._ref_im = rio.open(self._src_filename, 'r')

    def close(self):
        if self._src_im:
            self._src_im.close()
            self._src_im = None
        if self._ref_im:
            self._ref_im.close()
            self._ref_im = None


    def read(self, block: BlockPair):
        if not self._src_im or not self._ref_im:
            raise IoError('Datasets are closed')
        with self._src_lock:
            src_ra = RasterArray.from_rio_dataset(self._src_im, indexes=self._src_bands[block.band_i],
                                                  window=block.src_in_block, boundless=block.outer)
        with self._ref_lock:
            ref_ra = RasterArray.from_rio_dataset(self._ref_im, indexes=self._ref_bands[block.band_i],
                                                  window=block.ref_in_block, boundless=block.outer)
        return src_ra, ref_ra


    def _auto_block_shape(self, im_shape):
        max_block_mem = self._max_block_mem * (2 ** 20)  # MB to Bytes
        dtype_size = np.dtype(RasterArray.default_dtype).itemsize

        block_shape = np.array(im_shape)
        while (np.product(block_shape) * dtype_size > max_block_mem):
            div_dim = np.argmax(block_shape)
            block_shape[div_dim] /= 2
        return np.round(block_shape).astype('int')

    def block_pairs(self):
        # form the src and ref image windows
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._src_filename, 'r') as src_im:
            with rio.open(self._ref_filename, 'r') as ref_im:
                # src bounds in ref crs
                _src_bounds = transform_bounds(src_im.crs, ref_im.crs, *src_im.bounds)
                # find expanded ref window that covers src image
                ref_win = expand_window_to_grid(ref_im.window(*_src_bounds))
                # ref_transform = ref_im.window_transform(ref_win)

                # expanded ref bounds in src crs
                _ref_bounds = transform_bounds(ref_im.crs, src_im.crs, *ref_im.window_bounds(ref_win))
                # find expanded src window that covers expanded ref image
                src_win = expand_window_to_grid(src_im.window(*_ref_bounds))
                # src_transform = src_im.window_transform(src_win)

        # ref_dict = dict(fn=self._ref_filename, window=ref_win, bands=self._ref_bands, im=ref_im,
        #                 profile=dict(transform=ref_transform, width=ref_win.width, height=ref_win.height))
        # src_dict = dict(fn=self._src_filename, window=src_win, bands=self._src_bands, im=src_im,
        #                 profile=dict(transform=src_transform, width=src_win.width, height=src_win.height))
        if self._proc_crs == 'ref':
            proc_win, proc_im, other_im = (ref_win, ref_im, src_im)
        else:
            proc_win, proc_im, other_im = (src_win, src_im, ref_im)

        # def _create_block_pairs(proc_win: rasterio.windows.Window, proc_im: rasterio.DatasetReader,
        #                          other_im: rasterio.DatasetReader, overlap=self._overlap):
        overlap = np.array(self._overlap)
        proc_shape = np.array((proc_win.height, proc_win.width))
        proc_im_ul = np.array((proc_win.row_off, proc_win.col_off))
        proc_im_br = np.array((proc_win.height + proc_win.row_off, proc_win.width + proc_win.col_off))
        block_shape = self._auto_block_shape(proc_shape)
        if np.any(block_shape <= 2 * overlap):
            raise BlockSizeError("Block size is too small, increase 'max_block_mem' or decrease 'overlap'")
        # TODO: avoid repeating calcs over bands, it is slow
        for band_i in range(len(self._src_bands)):
            for ul_row, ul_col in product(
                    range(proc_win.row_off - overlap[0], proc_win.row_off + proc_win.height - 2 * overlap[0],
                          block_shape[0]),
                    range(proc_win.col_off - overlap[1], proc_win.col_off + proc_win.width - 2 * overlap[1],
                          block_shape[1])
            ):
                ul = np.array((ul_row, ul_col))
                br = ul + block_shape + (2 * overlap)
                proc_ul = np.fmax(ul, proc_im_ul)
                proc_br = np.fmin(br, proc_im_br)
                outer = np.any(proc_ul <= proc_im_ul) or np.any(proc_br >= proc_im_br)
                out_ul = ul + overlap
                out_br = br - overlap

                proc_in_block = Window(*proc_ul[::-1], *np.subtract(proc_br, proc_ul)[::-1])
                proc_out_block = Window(*out_ul[::-1], *np.subtract(out_br, out_ul)[::-1])

                other_in_bounds = transform_bounds(proc_im.crs, other_im.crs,
                                                   *proc_im.window_bounds(proc_in_block))
                other_in_block = expand_window_to_grid(other_im.window(*other_in_bounds))

                other_out_bounds = transform_bounds(proc_im.crs, other_im.crs,
                                                    *proc_im.window_bounds(proc_out_block))
                other_out_block = round_window_to_grid(other_im.window(*other_out_bounds))
                if self._proc_crs == 'ref':
                    block_pair = BlockPair(band_i, other_in_block, proc_in_block, other_out_block, proc_out_block,
                                           outer)
                else:
                    block_pair = BlockPair(band_i, proc_in_block, other_in_block, proc_out_block, other_out_block,
                                           outer)
                yield block_pair

