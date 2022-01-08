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
import rasterio
import rasterio as rio
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds
from rasterio.windows import Window
from typing import Tuple

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
        im_bands = tuple([bi + 1 for bi in range(im.count) if im.colorinterp[bi] != ColorInterp.alpha])
    return im_bands


def _inspect_image_pair(src_filename, ref_filename, proc_crs='auto'):
    # check ref_filename has enough bands
    src_bands = _inspect_image(src_filename)
    ref_bands = _inspect_image(ref_filename)
    if len(src_bands) > len(ref_bands):
        raise ImageContentError(f'{ref_filename.name} has fewer non-alpha bands than {src_filename.name}.')

    # warn if band counts don't match
    if len(src_bands) != len(ref_bands):
        logger.warning(f'Image non-alpha band counts don`t match. Using the first {len(src_bands)} non-alpha bands '
                       f'of {ref_filename.name}.')

    with rio.open(src_filename, 'r') as src_im:
        with WarpedVRT(rio.open(ref_filename, 'r'), crs=src_im.crs) as ref_im:
            # check coverage
            _ref_win = expand_window_to_grid(ref_im.window(*src_im.bounds), expand_pixels=(1, 1))
            _ref_ul = np.array((_ref_win.row_off, _ref_win.col_off))
            _ref_shape = np.array((_ref_win.height, _ref_win.width))
            # if not box(*ref_im.bounds).covers(box(*src_im.bounds)):
            if np.any(_ref_ul < 0) or np.any(_ref_shape > ref_im.shape):
                raise ImageContentError(f'Reference extent does not cover image: {src_filename.name}')

            src_pixel_smaller = np.prod(src_im.res) < np.prod(ref_im.res)
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

    return src_bands, ref_bands, proc_crs

"""Overlapping block object"""
BlockPair = namedtuple('BlockPair', ['band_i', 'src_in_block', 'ref_in_block', 'src_out_block', 'ref_out_block', 'outer'])

class ImPairReader():
    def __init__(self, src_filename, ref_filename, proc_crs='auto', overlap=(0,0), max_block_mem=np.inf):
        self._src_filename = pathlib.Path(src_filename)
        self._ref_filename = pathlib.Path(ref_filename)
        self._proc_crs = proc_crs
        self._overlap = overlap if not np.isscalar(overlap) else (overlap, overlap)
        self._max_block_mem = max_block_mem
        self._src_im:rasterio.DatasetReader = None
        self._ref_im:rasterio.DatasetReader = None
        self._env = None
        self._src_lock = threading.Lock()
        self._ref_lock = threading.Lock()
        self._src_win = None
        self._ref_win = None
        self._clipped_ref_profile = None
        self._src_profile = None

    @property
    def src_im(self)-> rasterio.DatasetReader:
        return self._src_im

    @property
    def ref_im(self)-> rasterio.DatasetReader:
        # TODO: get rid of this as it will conflict with clipped ref profile
        return self._ref_im

    @property
    def src_bands(self)-> Tuple[int,]:
        return self._src_bands

    @property
    def ref_bands(self)-> Tuple[int,]:
        return self._ref_bands

    @property
    def src_profile(self)-> rasterio.DatasetReader:
        return self._src_im.profile if self._src_im else {}

    @property
    def ref_profile(self)-> rasterio.DatasetReader:
        return self._clipped_ref_profile if self._ref_im else {}

    def __enter__(self):
        self._env = rio.Env(GDAL_NUM_THREADS='ALL_CPUs').__enter__()
        self.open()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if self._env:
            self._env.__exit__(exc_type, exc_val, exc_tb)
            self._env = None

    def _init_pair(self):
        self._src_bands, self._ref_bands, self._proc_crs = _inspect_image_pair(self._src_filename, self._ref_filename,
                                                                               self._proc_crs)

        # image-wide windows that allow re-projections between src and ref without loss of data
        self._ref_win = expand_window_to_grid(self._ref_im.window(*self._src_im.bounds))
        self._src_win = expand_window_to_grid(self._src_im.window(*self._ref_im.window_bounds(self._ref_win)))

        self._clipped_ref_profile = self._ref_im.profile.copy()
        self._clipped_ref_profile['transform'] = self._ref_im.window_transform(self._ref_win)
        self._clipped_ref_profile['height'] = self._ref_win.height
        self._clipped_ref_profile['width'] = self._ref_win.width


    def open(self):
        # self._blocks = self._create_ovl_blocks(overlap=self._overlap)
        self._src_im = rio.open(self._src_filename, 'r')
        self._ref_im = rio.open(self._ref_filename, 'r')

        if self._src_im.crs.to_proj4() != self._ref_im.crs.to_proj4():
            # open the image pair in the same CRS, re-projecting the lower resolution image into the CRS of the other
            logger.warning(f'Source and reference image pair are not in the same CRS: {self._src_filename.name} and '
                           f'{self._ref_filename.name}')
            if self._proc_crs == 'ref':
                self._ref_im = WarpedVRT(self._ref_im, crs=self._src_im.crs, resampling=Resampling.bilinear)
            else:
                self._src_im = WarpedVRT(self._src_im, crs=self._ref_im.crs, resampling=Resampling.bilinear)
        self._init_pair()

    def close(self):
        if self._src_im:
            self._src_im.close()
            self._src_im = None
        if self._ref_im:
            self._ref_im.close()
            self._ref_im = None


    def read(self, block: BlockPair):
        if not self._src_im or self._src_im.closed or not self._ref_im or self._ref_im.closed :
            raise IoError(f'Source and reference image pair are closed: {self._src_filename.name} and '
                           f'{self._ref_filename.name}')
        with self._src_lock:
            src_ra = RasterArray.from_rio_dataset(self._src_im, indexes=self._src_bands[block.band_i],
                                                  window=block.src_in_block, boundless=block.outer)
        with self._ref_lock:
            ref_ra = RasterArray.from_rio_dataset(self._ref_im, indexes=self._ref_bands[block.band_i],
                                                  window=block.ref_in_block, boundless=block.outer)
        return src_ra, ref_ra


    def _auto_block_shape(self, proc_im: rasterio.DatasetReader, other_im: rasterio.DatasetReader):
        # convert _max_block_mem from MB to Bytes in lowest res image space
        mem_scale = np.product(np.divide(proc_im.res, other_im.res))
        max_block_mem = self._max_block_mem * (2 ** 20) / mem_scale if self._max_block_mem > 0 else np.inf
        dtype_size = np.dtype(RasterArray.default_dtype).itemsize

        # if proc_im.is_tiled:    # adjust block_shape to be on tile boundaries, only realy useful for proc-crs==src
        #     tile_shape = np.array((proc_im.block_shapes[0]))
        #     tile_div = np.floor(block_shape / tile_shape)
        #     if np.all(tile_div > 0):
        #         block_shape = tile_div * tile_shape
        block_shape = np.array(proc_im.shape).astype('float')
        while ((np.product(block_shape) * dtype_size > max_block_mem)):
            div_dim = np.argmax(block_shape)
            block_shape[div_dim] /= 2


        return np.ceil(block_shape).astype('int')


    def block_pairs(self):
        """ Iterator over co-located pairs of source and reference image blocks.  For use in read(). """

        if not self._src_im or self._src_im.closed or not self._ref_im or self._ref_im.closed :
            raise IoError(f'Source and reference image pair are closed: {self._src_filename.name} and '
                          f'{self._ref_filename.name}')
        #
        # # image-wide windows that allow re-projections between src and ref without loss of data
        # ref_win = expand_window_to_grid(self._ref_im.window(*self._src_im.bounds))
        # src_win = expand_window_to_grid(self._src_im.window(*self._ref_im.window_bounds(ref_win)))

        # assign 'src' and 'ref' to 'proc' and 'other' according to proc_crs
        # (blocks are first formed in the 'proc' (usually the lowest resolution) image space, from which the equivalent
        # blocks in 'other' image space are then inferred)
        if self._proc_crs == 'ref':
            proc_win, proc_im, other_im = (self._ref_win, self._ref_im, self._src_im)
        else:
            proc_win, proc_im, other_im = (self._src_win, self._src_im, self._ref_im)

        # initialise block formation variables
        overlap = np.array(self._overlap)
        proc_win_shape = np.array((proc_win.height, proc_win.width))
        proc_win_ul = np.array((proc_win.row_off, proc_win.col_off))
        proc_win_br = np.array((proc_win.height + proc_win.row_off, proc_win.width + proc_win.col_off))
        proc_im_ul = np.array([0, 0])
        proc_im_br = np.array(proc_im.shape)

        # calculate a block shape (in proc_crs) that satisfies the max_block_mem requirement
        block_shape = self._auto_block_shape(proc_im, other_im)

        # TODO: form an error message in terms of kernel shape, perhaps by catching this exception
        if np.any(block_shape <= np.fmax(2 * overlap, (3, 3))):
            raise BlockSizeError(f"Block size {block_shape} is smaller than overlap, "
                                 f"increase 'max_block_mem', or decrease 'max_blocks', or 'overlap'")

        # form the overlapping blocks in 'proc' space, and find their equivalents in 'other' space
        block_pairs = []
        for band_i in range(len(self._src_bands)): # outer loop over bands - faster for band interleaved images
            for ul_row, ul_col in product(
                    range(proc_win.row_off - overlap[0], proc_win.row_off + proc_win.height - 2 * overlap[0],
                          block_shape[0]),
                    range(proc_win.col_off - overlap[1], proc_win.col_off + proc_win.width - 2 * overlap[1],
                          block_shape[1])
            ):
                # find UL and BR corners for overlapping block in proc space
                ul = np.array((ul_row, ul_col))
                br = ul + block_shape + (2 * overlap)
                in_ul = np.fmax(ul, proc_win_ul)
                in_br = np.fmin(br, proc_win_br)
                # find UL and BR corners for non-overlapping block in proc space
                out_ul = np.fmax(ul + overlap, (0, 0))
                out_br = np.fmin(br - overlap, proc_im.shape)
                # block touches image boundary?
                outer = np.any(in_ul <= proc_win_ul) or np.any(in_br >= proc_win_br)

                # form rasterio windows corresponding to above block corners
                proc_in_block = Window(*in_ul[::-1], *np.subtract(in_br, in_ul)[::-1])
                proc_out_block = Window(*out_ul[::-1], *np.subtract(out_br, out_ul)[::-1])

                # form equivalent rasterio windows in 'other' space
                #TODO: expand or round for other_in_block?  Proc_crs=ref: For ref->src round is good as that means there will be no
                # nodata boundaries due to src transform beyond ref transform, but for src->ref, expand is good, as that
                # means there will be no nodata for ref pixels straddling src transform.
                other_in_block = round_window_to_grid(other_im.window(*proc_im.window_bounds(proc_in_block)))
                other_out_block = round_window_to_grid(other_im.window(*proc_im.window_bounds(proc_out_block)))

                other_out_ul = np.fmax(np.array(other_out_block.toranges())[:, 0], (0, 0))
                other_out_br = np.fmin(np.array(other_out_block.toranges())[:, 1], other_im.shape)
                other_out_block = Window(*other_out_ul[::-1], *np.subtract(other_out_br, other_out_ul)[::-1])

                # form the BlockPair named tuple, assigning 'proc' and 'other' back to 'src' and 'ref' for use in read()
                if self._proc_crs == 'ref':
                    block_pair = BlockPair(band_i, other_in_block, proc_in_block, other_out_block, proc_out_block,
                                           outer)
                else:
                    block_pair = BlockPair(band_i, proc_in_block, other_in_block, proc_out_block, other_out_block,
                                           outer)
                block_pairs.append(block_pair)
        return block_pairs
