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
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from functools import wraps
from itertools import product
from os import PathLike
from pathlib import Path
from typing import NamedTuple, ParamSpec, TypeVar

import numpy as np
import rasterio as rio
from rasterio.enums import MaskFlags
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from homonim import utils
from homonim.enums import ProcCrs
from homonim.errors import HomonimError, HomonimWarning
from homonim.raster_array import RasterArray

logger = logging.getLogger(__name__)

P = ParamSpec('P')
R = TypeVar('R')


def assert_open(meth: Callable[P, R]) -> Callable[P, R]:
    """Decorator for RasterPairReader and subclass methods to ensure source and
    reference datasets are open.
    """

    @wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if args and getattr(args[0], 'closed', False):
            raise RuntimeError('Source and reference datasets are closed.')
        return meth(*args, **kwargs)

    return wrapper


class BlockPair(NamedTuple):
    """A set of matching block windows for a source-reference image pair."""

    band_i: int
    """Band index (0 based)."""
    src_in_block: Window
    """Overlapping / input source window."""
    ref_in_block: Window
    """Overlapping / input reference window."""
    src_out_block: Window
    """Non-overlapping / output source window."""
    ref_out_block: Window
    """Non-overlapping / output reference window."""
    outer: bool
    """True if any part of the source blocks touch the source image boundary,
    otherwise False.
    """


class RasterPairReader:
    def __init__(
        self,
        src_filename: str | PathLike,
        ref_filename: str | PathLike,
        proc_crs: str | ProcCrs = ProcCrs.auto,
    ):
        """
        Base class for processing matching blocks from a source and reference image
        pair.

        Reference extents must encompass those of the source.

        Source and reference bands should be in wavelength matched order.

        :param src_filename:
            Path or URI of a source image.
        :param ref_filename:
            Path or URI of a reference image.
        :param proc_crs:
            # TODO: this doc is not accurate (here and elsewhere) - source and
                reference are projected to the source CRS for proc_crs=ref and vice
                versa.
            Which of the source or reference CRS and pixel grids to use for
            processing.  By default, the CRS and pixel grid with the lowest
            resolution is used (recommended).
        """
        self._src_file = os.fspath(src_filename)
        self._ref_file = os.fspath(ref_filename)
        src_name = Path(self._src_file).name
        ref_name = Path(self._ref_file).name
        stack = ExitStack()

        try:
            # open and validate the image pair
            env = rio.Env(
                GDAL_NUM_THREADS='ALL_CPUS',
                GTIFF_FORCE_RGBA=False,
                CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=True,
            )
            stack.enter_context(env)
            src_im = stack.enter_context(rio.open(self._src_file, 'r'))
            ref_im = stack.enter_context(rio.open(self._ref_file, 'r'))
            self._validate_image(src_im)
            self._validate_image(ref_im)

            # match bands and resolve proc_crs=ProcCrs.auto
            self._src_bands, self._ref_bands = self._match_bands(src_im, ref_im)
            self._proc_crs = self._resolve_proc_crs(src_im, ref_im, proc_crs)

            if src_im.crs != ref_im.crs:
                # reproject so that both images are in the same CRS
                if self._proc_crs is ProcCrs.ref:
                    warn_names = ('reference', ref_name, 'source', src_name)
                    ref_im = stack.enter_context(WarpedVRT(ref_im, crs=src_im.crs))
                else:
                    warn_names = ('source', src_name, 'reference', ref_name)
                    src_im = stack.enter_context(WarpedVRT(src_im, crs=ref_im.crs))
                warnings.warn(
                    "The {} '{}' will be reprojected into the CRS of the {} '{}'.  "
                    'Processing times can be improved if source and reference are in '
                    'the same CRS.'.format(*warn_names),
                    category=HomonimWarning,
                    stacklevel=2,
                )

            # create windows of the source / reference extents that allow reprojections
            # without loss of data
            self._ref_win = utils.expand_window_to_grid(ref_im.window(*src_im.bounds))
            ref_ranges = np.array(self._ref_win.toranges()).T
            if any(ref_ranges[0] < 0) or any(ref_ranges[1] > ref_im.shape):
                raise HomonimError('Reference extent does not cover source image')
            self._src_win = utils.expand_window_to_grid(
                src_im.window(*ref_im.window_bounds(self._ref_win))
            )
        except Exception:
            stack.close()
            raise

        self._stack = stack
        self._src_im = src_im
        self._ref_im = ref_im
        self._src_lock = threading.Lock()
        self._ref_lock = threading.Lock()

    @property
    def src_im(self) -> DatasetReader:
        """Source dataset."""
        return self._src_im

    @property
    def ref_im(self) -> DatasetReader:
        """Reference dataset."""
        return self._ref_im

    @property
    def src_bands(self) -> tuple[int, ...]:
        """Source non-alpha band indices (1-based)."""
        return self._src_bands

    @property
    def ref_bands(self) -> tuple[int, ...]:
        """Reference non-alpha band indices (1-based)."""
        return self._ref_bands

    @property
    def proc_crs(self) -> ProcCrs:
        """Which of the source and reference image CRS and pixel grids will be used
        for processing.
        """
        return self._proc_crs

    @property
    def closed(self) -> bool:
        """Whether the source and reference datasets are closed."""
        # source & reference should only be both open or both closed, but test with
        # an or in case
        return self._src_im.closed or self._ref_im.closed

    @staticmethod
    def _validate_image(im: DatasetReader):
        """Validate a dataset for use as a source or reference image."""
        name = Path(im.name).name
        is_masked = any(
            MaskFlags.all_valid not in im.mask_flag_enums[bi] for bi in range(im.count)
        )
        if im.nodata is None and not is_masked:
            warnings.warn(
                f"'{name}' has no mask or nodata value, any invalid pixels should "
                f'be masked before processing.',
                category=HomonimWarning,
                stacklevel=2,
            )
        if not (
            im.transform.a > 0
            and im.transform.e < 0
            and im.transform.b == 0
            and im.transform.d == 0
        ):
            raise HomonimError(f"'{name}' is not in a standard North-up orientation.")

    def _match_bands(
        self, src_im: DatasetReader, ref_im: DatasetReader
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Validate and match source and reference bands."""
        # retrieve non-alpha bands
        src_name = Path(src_im.name).name
        ref_name = Path(ref_im.name).name
        src_bands = utils.get_nonalpha_bands(src_im)
        logger.debug(f'{src_name} non-alpha bands: {src_bands}')
        ref_bands = utils.get_nonalpha_bands(ref_im)
        logger.debug(f'{ref_name} non-alpha bands: {ref_bands}')

        # check reference has enough bands
        if len(src_bands) > len(ref_bands):
            raise HomonimError(
                f"Reference '{ref_name}' has fewer non-alpha bands than source '"
                f"{src_name}'."
            )
        # warn if source and reference band counts don't match
        if len(src_bands) != len(ref_bands):
            warnings.warn(
                f'Source and reference non-alpha band counts don`t match. Using the '
                f'first {len(src_bands)} non-alpha bands of the reference.',
                category=HomonimWarning,
                stacklevel=2,
            )
        return src_bands, ref_bands

    @staticmethod
    def _resolve_proc_crs(
        src_im: DatasetReader, ref_im: DatasetReader, proc_crs: ProcCrs
    ) -> ProcCrs:
        """Return a ProcCrs instance defining which of the source or reference CRS
        and pixel grids should be used for processing.
        """
        proc_crs = ProcCrs(proc_crs)
        if proc_crs is not ProcCrs.auto:
            return proc_crs

        with ExitStack() as stack:
            if src_im.crs != ref_im.crs:
                # project reference into source CRS, so their resolutions are in the
                # same units for comparison
                ref_im = stack.enter_context(WarpedVRT(ref_im, crs=src_im.crs))
            src_smaller = np.prod(src_im.res) <= np.prod(ref_im.res)

        proc_crs, cmp_str = (
            (ProcCrs.ref, 'smaller') if src_smaller else (ProcCrs.src, 'larger')
        )

        logger.debug(
            f'Source resolution is {cmp_str} than the reference resolution. Using '
            f"proc_crs='{proc_crs}'."
        )
        return proc_crs

    def _find_block_shape(self, max_block_mem: float = np.inf) -> tuple[int, int]:
        """Find the shape of a proc_crs block that satisfies max_block_mem in the
        highest resolution image.
        """
        # scale max_block_mem to limit a proc_crs block so that the corresponding
        # highest resolution image block will be limited by the given max_block_mem
        src_area = np.prod(self._src_im.res)
        ref_area = np.prod(self._ref_im.res)
        if self.proc_crs is ProcCrs.ref:
            mem_scale = src_area / ref_area if ref_area > src_area else 1.0
            proc_win = self._ref_win
        elif self.proc_crs is ProcCrs.src:
            mem_scale = 1.0 if ref_area > src_area else ref_area / src_area
            proc_win = self._src_win
        else:
            raise ValueError("'proc_crs' has not been resolved.")
        max_block_mem = max_block_mem * mem_scale if max_block_mem > 0 else np.inf

        max_block_mem *= 2**20  # convert MB to bytes
        # TODO: make dtype a param?
        # the size of the RasterArray data type
        dtype_size = np.dtype(RasterArray._default_dtype).itemsize

        # TODO: find block_shape ~exacly as sqrt(max_block_mem/dtype_size), possibly
        #  limiting rows to 512.
        # set the starting block_shape to correspond to the entire window
        block_shape = np.array((proc_win.height, proc_win.width)).astype('float')

        # keep halving the block_shape along the longest dimension until it satisfies
        # max_block_mem
        while block_shape.prod() * dtype_size > max_block_mem:
            block_shape[block_shape.argmax()] /= 2

        if np.any(block_shape < (1, 1)):
            raise HomonimError(
                "Block shape is smaller than a pixel.  Increase 'max_block_mem'."
            )

        block_shape = np.ceil(block_shape).astype('int')
        block_shape_ = tuple(block_shape.tolist())
        logger.debug(
            f'Using block shape: {block_shape_}, of image shape: '
            f'{(proc_win.height, proc_win.width)} ({self.proc_crs} pixels)'
        )

        # warn if the block shape in the highest res image is less than a typical tile
        if np.any(block_shape / mem_scale < (256, 256)) and np.any(
            block_shape < (proc_win.height, proc_win.width)
        ):
            warnings.warn(
                f"Block shape is small: {block_shape_}.  Increasing 'max_block_mem' "
                f'will improve processing times.',
                category=HomonimWarning,
                stacklevel=2,
            )
        return block_shape_

    def close(self):
        """Close the source and reference datasets."""
        self._stack.close()

    @assert_open
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.__exit__(exc_type, exc_val, exc_tb)

    @assert_open
    def read(self, block_pair: BlockPair) -> tuple[RasterArray, RasterArray]:
        """
        Read source and reference image blocks (thread safe).

        :param block_pair:
            :class:`BlockPair` instance defining the source and reference blocks to
            read.

        :return:
            (Source, reference) blocks.
        """
        # TODO: could block_pairs() be simplified by just passing a proc_crs window &
        #  overlap here, and working out the block_pair windows?
        with self._src_lock:
            src_ra = RasterArray.from_rio_dataset(
                self._src_im,
                indexes=self._src_bands[block_pair.band_i],
                window=block_pair.src_in_block,
            )
        with self._ref_lock:
            ref_ra = RasterArray.from_rio_dataset(
                self._ref_im,
                indexes=self._ref_bands[block_pair.band_i],
                window=block_pair.ref_in_block,
            )
        return src_ra, ref_ra

    @assert_open
    def block_pairs(
        self, overlap: tuple[int, int] = (0, 0), max_block_mem: float = np.inf
    ) -> Iterable[BlockPair]:
        """
        Generate matched source - reference image blocks.

        :param overlap:
            (row, column) block overlap in pixels of the :attr:`proc_crs` image.
        :param max_block_mem:
            Maximum block size in the highest resolution of the source and reference
            images (MB).  If ``float('inf')``, a block will correspond to a full
            image band.

        :return:
            :class:`BlockPair` instance defining matched source - reference image
            blocks.
        """
        # find the proc_crs block shape
        block_shape = np.array(self._find_block_shape(max_block_mem=max_block_mem))
        overlap = np.array(overlap).astype('int')
        if np.any(block_shape <= overlap):
            raise HomonimError(
                f'Block shape {block_shape} is smaller than the overlap {overlap}.  '
                f"Increase 'max_block_mem'."
            )

        # initialise block formation variables
        # blocks are first formed in proc_crs, then transformed to the 'other'
        # image crs, so here we assign the src/ref windows etc. to proc_* equivalents
        if self.proc_crs is ProcCrs.ref:
            proc_win, proc_im, other_im = (self._ref_win, self._ref_im, self._src_im)
        else:
            proc_win, proc_im, other_im = (self._src_win, self._src_im, self._ref_im)

        proc_win_ul = np.array((proc_win.row_off, proc_win.col_off))
        proc_win_br = np.array(
            (proc_win.height + proc_win.row_off, proc_win.width + proc_win.col_off)
        )

        # outer loop over bands so that all blocks in a band are yielded
        # consecutively - this is fastest for reading band interleaved images.
        for band_i in range(len(self._src_bands)):
            # Inner loop over the upper left corner row, col for each overlapping block
            # TODO: the start stop of these ranges is strange - see the fmin/fmax below
            ul_row_range = range(
                proc_win.row_off - overlap[0],
                proc_win.row_off + proc_win.height - overlap[0],
                block_shape[0],
            )
            ul_col_range = range(
                proc_win.col_off - overlap[1],
                proc_win.col_off + proc_win.width - overlap[1],
                block_shape[1],
            )
            # TODO: it would be clearer if ul_row, ul_col corresponded to the
            #  non-overlapping blocks, then subtract / add overlap to make the
            #  overlapping blocks
            for ul_row, ul_col in product(ul_row_range, ul_col_range):
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

                # Create rasterio windows corresponding to above block corners.
                # Note:
                # - Consecutive proc_in_block's will overlap by exactly ``overlap``.
                # - Consecutive proc_out_block's will be exactly adjacent.
                proc_in_block = Window(*in_ul[::-1], *np.subtract(in_br, in_ul)[::-1])
                proc_out_block = Window(
                    *out_ul[::-1], *np.subtract(out_br, out_ul)[::-1]
                )

                # Create equivalent rasterio windows in 'other' space.
                # Note:
                # - other_in_block boundaries are expanded to ensure that
                # re-projecting between source/reference CRSs does not mask valid
                # *_out_block pixels.  This means that consecutive other_in_blocks
                # may overlap by more than ``overlap``.
                # - consecutive other_out_block's may overlap by a pixel.
                other_in_block = utils.expand_window_to_grid(
                    other_im.window(*proc_im.window_bounds(proc_in_block))
                )
                other_out_block = utils.round_window_to_grid(
                    other_im.window(*proc_im.window_bounds(proc_out_block))
                )

                # create the BlockPair named tuple, assigning 'proc' and 'other' back
                # to 'src' and 'ref' for passing to read()
                if self.proc_crs is ProcCrs.ref:
                    block_pair = BlockPair(
                        band_i,
                        other_in_block,
                        proc_in_block,
                        other_out_block,
                        proc_out_block,
                        outer,
                    )
                else:
                    block_pair = BlockPair(
                        band_i,
                        proc_in_block,
                        other_in_block,
                        proc_out_block,
                        other_out_block,
                        outer,
                    )
                yield block_pair
