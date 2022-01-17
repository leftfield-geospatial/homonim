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
from typing import Tuple

import numpy as np
import rasterio
import rasterio as rio
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window

from homonim import errors
from homonim import utils
from homonim.enums import ProcCrs
from homonim.raster_array import RasterArray

logger = logging.getLogger(__name__)

"""Named tuple to contain a set of matching block windows for a source-reference image pair"""
BlockPair = namedtuple('BlockPair',
                       ['band_i',           # band index (0 based)
                        'src_in_block',     # overlapping source window
                        'ref_in_block',     # overlapping reference window
                        'src_out_block',    # non-overlapping source window
                        'ref_out_block',    # non-overlapping reference window
                        'outer'])           # True if src_*_block touches the boundary of the source image


class RasterPairReader():
    """
    A thread-safe class for reading matching blocks from a source and reference image pair.

    Images are divided into (optionally overlapping) blocks to satisfy a maximum block memory limit.  Blocks extents
    are constructed so that re-projections between source and reference do not lose valid data.
    """

    def __init__(self, src_filename, ref_filename, proc_crs=ProcCrs.auto, overlap=(0, 0), max_block_mem=np.inf):
        """
        Construct a RasterPairReader.

        Parameters
        ----------
        src_filename: str, pathlib.Path
            Path to the source image file.
        ref_filename: str, pathlib.Path
            Path to the reference image file.  The extents of this image should cover src_filename with at least a 2
            pixel boundary, and it should have at least as many bands as src_filename.  The ordering of the bands
            in src_filename and ref_filename should match.
        proc_crs: homonim.enums.ProcCrs
            The initial proc_crs setting, if proc_crs=ProcCrs.auto (recommended), the RasterPairReader.proc_crs
            attribute will be set to represent the lowest resolution of the source and reference image.
            [default: ProcCrs.auto]
        overlap: tuple
            Block overlap (rows, columns) in pixels of the proc_crs image. [default: (0,0)]
        max_block_mem: float
            The maximum allowable block size in MB. The image is divided into 2**n blocks with n the smallest number
            where max_block_mem is satisfied.  If max_block_mem=float('inf'), the block shape will be set to the
            encompass full extent of the source image. [default: float('inf')]
        """
        self._src_filename = pathlib.Path(src_filename)
        self._ref_filename = pathlib.Path(ref_filename)
        if not isinstance(proc_crs, ProcCrs):
            raise ValueError("'proc_crs' must be an instance of homonim.enums.ProcCrs")
        self._proc_crs = proc_crs
        self._overlap = np.array(overlap).astype('int')
        self._max_block_mem = max_block_mem

        self._src_im: rasterio.DatasetReader = None
        self._ref_im: rasterio.DatasetReader = None
        self._env = None
        self._src_lock = threading.Lock()
        self._ref_lock = threading.Lock()
        self._src_win = None
        self._ref_win = None
        self._crop_ref_profile = None
        self._src_profile = None
        self._block_shape = None
        self._src_bands, self._ref_bands, self._proc_crs = self._inspect_image_pair(self._src_filename,
                                                                                    self._ref_filename, self._proc_crs)

    @staticmethod
    def _inspect_image(filename):
        """Inspect an image file for valid use as a source or reference, and return it's non-alpha band indices."""
        filename = pathlib.Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f'{filename} does not exist')

        # check for 12 bit JPEG format
        with rio.open(filename, 'r') as im:
            try:
                tmp_array = im.read(1, window=im.block_window(1, 0, 0))
            except Exception as ex:
                if im.profile['compress'] == 'jpeg':  # assume it is a 12bit JPEG
                    raise errors.UnsupportedImageError(
                        f'Could not read image {filename.name}.  JPEG compression with NBITS==12 is not supported, '
                        f'you probably need to recompress this image.'
                    )
                else:
                    raise ex

            # warn if there is no nodata or associated mask
            is_masked = any([MaskFlags.all_valid not in im.mask_flag_enums[bi] for bi in range(im.count)])
            if im.nodata is None and not is_masked:
                logger.warning(f'{filename.name} has no mask or nodata value, '
                               f'any invalid pixels should be masked before processing.')

            # form a tuple of non-alpha band indices
            bands = tuple([bi + 1 for bi in range(im.count) if im.colorinterp[bi] != ColorInterp.alpha])
        return bands

    @staticmethod
    def _inspect_image_pair(src_filename, ref_filename, proc_crs=ProcCrs.auto):
        """ Inspect a source - reference image pair for valid use, and return the non-alpha source & reference."""
        src_bands = RasterPairReader._inspect_image(src_filename)
        ref_bands = RasterPairReader._inspect_image(ref_filename)
        # check reference has enough bands
        if len(src_bands) > len(ref_bands):
            raise errors.ImageContentError(f'{ref_filename.name} has fewer non-alpha bands than {src_filename.name}.')

        # warn if source and reference band counts don't match
        if len(src_bands) != len(ref_bands):
            logger.warning(f'Image non-alpha band counts don`t match. Using the first {len(src_bands)} non-alpha bands '
                           f'of {ref_filename.name}.')

        with rio.open(src_filename, 'r') as src_im:
            with WarpedVRT(rio.open(ref_filename, 'r'), crs=src_im.crs) as ref_im:
                # check reference image extents cover the source
                _ref_win = utils.expand_window_to_grid(ref_im.window(*src_im.bounds), expand_pixels=(1, 1))
                _ref_ul = np.array((_ref_win.row_off, _ref_win.col_off))
                _ref_shape = np.array((_ref_win.height, _ref_win.width))
                if np.any(_ref_ul < 0) or np.any(_ref_shape > ref_im.shape):
                    raise errors.ImageContentError(f'Reference extent does not cover source image: {src_filename.name}')

                # compare source and reference resolutions
                src_pixel_smaller = np.prod(src_im.res) < np.prod(ref_im.res)
                cmp_str = "smaller" if src_pixel_smaller else "larger"
                if proc_crs == ProcCrs.auto:
                    # set proc_crs to the lowest resolution of the source and reference images
                    proc_crs = ProcCrs.ref if src_pixel_smaller else ProcCrs.src
                    logger.debug(f"Source pixel size {src_im.res} is {cmp_str} than the reference {ref_im.res}, "
                                 f"using proc_crs='{proc_crs}'")
                elif ((proc_crs == ProcCrs.src and src_pixel_smaller) or
                      (proc_crs == ProcCrs.ref and not src_pixel_smaller)):
                    # warn if the proc_crs value does not correspond to the lowest resolution of the source and
                    # reference images
                    rec_crs_str = ProcCrs.ref if src_pixel_smaller else ProcCrs.src
                    logger.warning(f"proc_crs='{rec_crs_str}' is recommended when "
                                   f"the source image pixel size is {cmp_str} than the reference.")

        return src_bands, ref_bands, proc_crs

    def open(self):
        """Open the source and reference images for reading."""
        self._src_im = rio.open(self._src_filename, 'r')
        self._ref_im = rio.open(self._ref_filename, 'r')

        # It is essential that the source and reference are in the same CRS so that rectangular regions of valid
        # data in one will re-project to rectangular regions of valid data in the other.
        if self._src_im.crs.to_proj4() != self._ref_im.crs.to_proj4():
            # open the image pair in the same CRS, re-projecting the proc_crs (usually lower resolution) image into the
            # CRS of the other
            logger.warning(f'Source and reference image pair are not in the same CRS: {self._src_filename.name} and '
                           f'{self._ref_filename.name}')
            if self._proc_crs == ProcCrs.ref:
                self._ref_im = WarpedVRT(self._ref_im, crs=self._src_im.crs, resampling=Resampling.bilinear)
            else:
                self._src_im = WarpedVRT(self._src_im, crs=self._ref_im.crs, resampling=Resampling.bilinear)

        # create image windows that allow re-projections between source and reference without loss of data
        self._ref_win = utils.expand_window_to_grid(self._ref_im.window(*self._src_im.bounds))
        self._src_win = utils.expand_window_to_grid(self._src_im.window(*self._ref_im.window_bounds(self._ref_win)))

    def close(self):
        """Close the source and reference image datasets."""
        if not self._src_im.closed:
            self._src_im.close()
        if not self._ref_im.closed:
            self._ref_im.close()

    def __enter__(self):
        self._env = rio.Env(GDAL_NUM_THREADS='ALL_CPUs').__enter__()
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self._env.__exit__(exc_type, exc_val, exc_tb)

    def _assert_open(self):
        """Raise an IoError if the source and reference images are not open."""
        if not self._src_im or not self._ref_im or self._src_im.closed or self._ref_im.closed:
            raise errors.IoError(f'The raster pair has not been opened: {self._src_filename.name} and '
                                 f'{self._ref_filename.name}')

    def _auto_block_shape(self, proc_win=None):
        """Find a block shape that satisfies max_block_mem."""

        # adjust max_block_mem to represent the size of a block in the highest resolution image, but scaled to the
        # equivalent in proc_crs.
        src_pix_area = np.product(self._src_im.res)
        ref_pix_area = np.product(self._ref_im.res)
        if self._proc_crs == ProcCrs.ref:
            mem_scale = src_pix_area / ref_pix_area if ref_pix_area > src_pix_area else 1.
        elif self._proc_crs == ProcCrs.src:
            mem_scale = 1. if ref_pix_area > src_pix_area else ref_pix_area / src_pix_area
        max_block_mem = self._max_block_mem * mem_scale if self._max_block_mem > 0 else np.inf

        max_block_mem *= 2 ** 20  # convert MB to bytes
        dtype_size = np.dtype(RasterArray.default_dtype).itemsize  # the size of the RasterArray data type

        if proc_win is None:
            # set the starting block_shape to correspond to the entire band
            block_shape = (self._ref_im.shape if self._proc_crs == ProcCrs.ref else self._src_im.shape).astype('float')
        else:
            # set the starting block_shape to correspond to the entire window
            block_shape = np.array((proc_win.height, proc_win.width)).astype('float')

        # keep halving the block_shape along the longest dimension until it satisfies max_block_mem
        while ((np.product(block_shape) * dtype_size) > max_block_mem):
            div_dim = np.argmax(block_shape)
            block_shape[div_dim] /= 2

        return np.ceil(block_shape).astype('int')

    @property
    def src_im(self) -> rasterio.DatasetReader:
        """The source rasterio dataset."""
        return self._src_im

    @property
    def ref_im(self) -> rasterio.DatasetReader:
        """The reference rasterio dataset."""
        return self._ref_im

    @property
    def src_bands(self) -> Tuple[int,]:
        """The source non-alpha band indices (1-based)."""
        return self._src_bands

    @property
    def ref_bands(self) -> Tuple[int,]:
        """The reference non-alpha band indices (1-based)."""
        return self._ref_bands

    @property
    def proc_crs(self) -> ProcCrs:
        """The 'processing CRS' i.e. which of the source/reference image spaces is selected for processing."""
        return self._proc_crs

    @property
    def overlap(self) -> Tuple[int,]:
        """The block overlap (rows, columns) in pixels of the 'proc_crs' image."""
        return tuple(self._overlap)

    @property
    def block_shape(self) -> Tuple[int,]:
        """The image block shape (rows, columns) in pixels of the 'proc_crs' image."""
        self._assert_open()
        # find the block shape if it has not been found already
        if self._block_shape is None:
            proc_win = self._ref_win if self._proc_crs == ProcCrs.ref else self._src_win
            self._block_shape = self._auto_block_shape(proc_win=proc_win)

        # check that the block shape is at least (3, 3) and bigger than the overlap
        if np.any(self._block_shape <= np.fmax(self._overlap, (3, 3))):
            raise errors.BlockSizeError(
                f"The block shape ({self.block_shape}) that satisfies the maximum block size ({self._max_block_mem}MB) "
                f"is too small to accommodate the overlap ({self._overlap}) between consecutive blocks. "
                f"Increase the 'max_block_mem', or decrease the 'overlap' argument(s) to RasterPairReader.__init__()"
            )
        return self._block_shape

    def read(self, block: BlockPair):
        """
        Read a matching pair of source and reference image blocks.

        Parameters
        ----------
        block: BlockPair
            The BlockPair named tuple (as returned by block_pairs()) specifying the source and reference blocks to be
            read.

        Returns
        -------
        src_ra: RasterArray
            The source image block wrapped in a RasterArray.
        ref_ra: RasterArray
            The reference image block wrapped in a RasterArray.
        """
        self._assert_open()

        with self._src_lock:
            src_ra = RasterArray.from_rio_dataset(self._src_im, indexes=self._src_bands[block.band_i],
                                                  window=block.src_in_block)
        with self._ref_lock:
            ref_ra = RasterArray.from_rio_dataset(self._ref_im, indexes=self._ref_bands[block.band_i],
                                                  window=block.ref_in_block)
        return src_ra, ref_ra

    def block_pairs(self):
        """
        Iterator over the paired source-reference image blocks.

        Yields
        -------
        block_pair: BlockPair
            A BlockPair named tuple specifying the overlapping ('*_in_block'), and non-overlapping ('*_out_block')
            source and reference image blocks.
        """
        self._assert_open()

        # initialise block formation variables
        # blocks are first formed in proc_crs, then transformed to the 'other' image crs, so here we assign the src/ref
        # windows etc to proc_* equivalents
        if self._proc_crs == ProcCrs.ref:
            proc_win, proc_im, other_im = (self._ref_win, self._ref_im, self._src_im)
        else:
            proc_win, proc_im, other_im = (self._src_win, self._src_im, self._ref_im)
        overlap = self._overlap
        block_shape = self.block_shape
        proc_win_ul = np.array((proc_win.row_off, proc_win.col_off))
        proc_win_br = np.array((proc_win.height + proc_win.row_off, proc_win.width + proc_win.col_off))

        # Form the overlapping blocks in proc_crs, and find their equivalents in 'other' space.
        # Outer loop over bands to have so that all blocks in band are yielded consecutively - this is fastest for
        # reading band interleaved images.
        for band_i in range(len(self._src_bands)):
            # Inner loop over the upper left corner row, col for each overlapping block
            for ul_row, ul_col in product(
                    range(proc_win.row_off - overlap[0], proc_win.row_off + proc_win.height - 2 * overlap[0],
                          block_shape[0]),
                    range(proc_win.col_off - overlap[1], proc_win.col_off + proc_win.width - 2 * overlap[1],
                          block_shape[1])
            ):
                # find UL and BR corners for overlapping block in proc space
                ul = np.array((ul_row, ul_col))
                br = ul + block_shape + (2 * overlap)
                # limit block extents to image window extents
                in_ul = np.fmax(ul, proc_win_ul)
                in_br = np.fmin(br, proc_win_br)
                # find UL and BR corners for non-overlapping block in proc space
                out_ul = np.fmax(ul + overlap, proc_win_ul)
                out_br = np.fmin(br - overlap, proc_win_br)
                # block touches image boundary?
                outer = np.any(in_ul <= proc_win_ul) or np.any(in_br >= proc_win_br)

                # create rasterio windows corresponding to above block corners
                proc_in_block = Window(*in_ul[::-1], *np.subtract(in_br, in_ul)[::-1])
                proc_out_block = Window(*out_ul[::-1], *np.subtract(out_br, out_ul)[::-1])

                # create equivalent rasterio windows in 'other' space
                other_in_block = utils.round_window_to_grid(other_im.window(*proc_im.window_bounds(proc_in_block)))
                other_out_block = utils.round_window_to_grid(other_im.window(*proc_im.window_bounds(proc_out_block)))

                # create the BlockPair named tuple, assigning 'proc' and 'other' back to 'src' and 'ref' for passing to
                # read()
                if self._proc_crs == ProcCrs.ref:
                    block_pair = BlockPair(band_i, other_in_block, proc_in_block, other_out_block, proc_out_block,
                                           outer)
                else:
                    block_pair = BlockPair(band_i, proc_in_block, other_in_block, proc_out_block, other_out_block,
                                           outer)
                yield block_pair
