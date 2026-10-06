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
from dataclasses import dataclass
from functools import wraps
from itertools import product
from os import PathLike
from pathlib import Path
from typing import ParamSpec, TypeVar

import numpy as np
import rasterio as rio
from rasterio.enums import MaskFlags
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, intersection

from homonim import utils
from homonim.enums import ProcCrs
from homonim.errors import HomonimError, HomonimWarning
from homonim.raster_array import RasterArray

logger = logging.getLogger(__name__)

P = ParamSpec('P')
R = TypeVar('R')


def assert_open(meth: Callable[P, R]) -> Callable[P, R]:
    """Decorator for RasterPairReader and subclass methods to ensure source and
    reference images are open.
    """

    @wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if args and getattr(args[0], 'closed', False):
            raise RuntimeError('Source and reference images are closed.')
        return meth(*args, **kwargs)

    return wrapper


@dataclass
class BlockPair:
    """Matching block windows for a source-reference image pair."""

    # Band index (0 based)
    band_i: int
    # Overlapping source window
    src_in_block: Window
    # Overlapping reference window
    ref_in_block: Window
    # Adjacent source window
    src_out_block: Window
    # Adjacent reference window
    ref_out_block: Window

    @classmethod
    def from_proc_other(
        cls,
        proc_crs: ProcCrs,
        band_i: int,
        proc_in_block: Window,
        proc_out_block: Window,
        other_in_block: Window,
        other_out_block: Window,
    ):
        """Return a BlockPair for the given proc_crs, and 'proc' & 'other' space
        windows.
        """
        if proc_crs is ProcCrs.ref:
            return cls(
                band_i, other_in_block, proc_in_block, other_out_block, proc_out_block
            )
        elif proc_crs is ProcCrs.src:
            return cls(
                band_i, proc_in_block, other_in_block, proc_out_block, other_out_block
            )
        else:
            raise ValueError("'proc_crs' has not been resolved.")


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

        Can be used as a context manager to automatically close source and reference
        images on exiting.

        :param src_filename:
            Path or URI of a source image.
        :param ref_filename:
            Path or URI of a reference image.
        :param proc_crs:
            Which of the source or reference pixel grids to use for processing.  By
            default, the grid with the lowest resolution is used (recommended).
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
                raise HomonimError(
                    f"Reference '{ref_name}' extent does does not cover the source "
                    f"'{src_name}'."
                )
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
        """Whether the source and reference images are closed."""
        # source & reference should only be both open or both closed, but test with
        # 'or' in case
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
            f'proc_crs={proc_crs}.'
        )
        return proc_crs

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

    def _find_block_shape(self, max_block_mem: float = 0) -> tuple[int, int]:
        """Find the shape of a proc_crs block that satisfies max_block_mem in the
        highest resolution image.
        """
        if max_block_mem < 0:
            raise HomonimError(
                "'max_block_mem' should be greater than or equal to zero."
            )

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
        max_block_mem = np.inf if max_block_mem == 0 else max_block_mem * mem_scale

        max_block_mem *= 2**20  # convert MB to bytes
        dtype_size = np.dtype(RasterArray._default_dtype).itemsize

        # find a block shape that divides the image into equal sized blocks
        # and satisfies max_block_mem
        max_block_shape = np.array((proc_win.height, proc_win.width))
        block_shape = max_block_shape.astype('float', copy=True)
        # keep halving the block_shape along the longest dimension until it satisfies
        # max_block_mem
        while block_shape.prod() * dtype_size > max_block_mem:
            block_shape[block_shape.argmax()] /= 2
        block_shape = np.ceil(block_shape).astype('int')

        def totuple(a):
            return tuple(a.tolist())

        logger.debug(
            f'Using block shape {totuple(block_shape)} in image shape '
            f'{totuple(max_block_shape)}.'
        )

        # warn if the block shape in the highest res image is less than a typical tile
        if np.any(
            (block_shape / mem_scale < (256, 256)) & (block_shape < max_block_shape)
        ):
            warnings.warn(
                f'Block shape {totuple(block_shape)} is small.  Increasing '
                f"'max_block_mem' will improve processing times.",
                category=HomonimWarning,
                stacklevel=2,
            )
        return totuple(block_shape)

    @assert_open
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.__exit__(exc_type, exc_val, exc_tb)

    @assert_open
    def _read(self, block_pair: BlockPair) -> tuple[RasterArray, RasterArray]:
        """Read source and reference image blocks defined by the given BlockPair
        (thread safe).
        """
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
    def _block_pairs(
        self, overlap: tuple[int, int] = (0, 0), max_block_mem: float = 0
    ) -> Iterable[BlockPair]:
        """Generate matched overlappping source - reference image blocks,
        given overlap (in proc_crs pixels), and max_block_mem defining the maximum
        block size (in the highest resolution image).
        """
        # find the proc_crs block shape
        block_shape = self._find_block_shape(max_block_mem=max_block_mem)
        if np.any(np.array(block_shape) <= overlap):
            raise HomonimError(
                f'Block shape {block_shape} is smaller than the overlap {overlap}.  '
                f"Increase 'max_block_mem'."
            )

        # initialise block formation variables
        if self.proc_crs is ProcCrs.ref:
            proc_win, proc_im, other_im = (self._ref_win, self._ref_im, self._src_im)
        else:
            proc_win, proc_im, other_im = (self._src_win, self._src_im, self._ref_im)
        block_ranges = [
            range(start, stop, step)
            for (start, stop), step in zip(
                proc_win.toranges(), block_shape, strict=True
            )
        ]

        # outer loop over bands so that all blocks in a band are yielded
        # consecutively - this is fastest for reading band interleaved images.
        for band_i in range(len(self._src_bands)):
            # inner loop over the upper left (row, col) corner of each non-overlapping
            # block
            for ul in product(*block_ranges):
                # Create 'proc' space windows:
                # - Consecutive proc_in_block's will overlap by exactly 'overlap'.
                # - Consecutive proc_out_block's will be exactly adjacent.
                proc_in_block = Window(
                    ul[1] - overlap[1],
                    ul[0] - overlap[0],
                    block_shape[1] + 2 * overlap[1],
                    block_shape[0] + 2 * overlap[0],
                )
                proc_in_block = intersection(proc_in_block, proc_win)
                proc_out_block = Window(*ul[::-1], *block_shape[::-1])
                proc_out_block = intersection(proc_out_block, proc_win)

                # Create matching 'other' space windows:
                # - other_in_block boundaries are expanded to ensure reprojecting
                #   between source/reference CRSs does not mask valid *_out_block
                #   pixels.  This means consecutive other_in_blocks may overlap by more
                #   than 'overlap'.
                # - Consecutive other_out_block's may overlap by a pixel.
                other_in_block = utils.expand_window_to_grid(
                    other_im.window(*proc_im.window_bounds(proc_in_block))
                )
                other_out_block = utils.round_window_to_grid(
                    other_im.window(*proc_im.window_bounds(proc_out_block))
                )

                # yield a BlockPair
                yield BlockPair.from_proc_other(
                    self.proc_crs,
                    band_i,
                    proc_in_block,
                    proc_out_block,
                    other_in_block,
                    other_out_block,
                )

    def close(self):
        """Close the source and reference images."""
        self._stack.close()
