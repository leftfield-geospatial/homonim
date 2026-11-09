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
from collections.abc import Callable
from functools import wraps
from math import ceil, prod
from typing import ParamSpec, TypeVar

import numpy as np
from rasterio._io import Affine
from rasterio.enums import ColorInterp, Resampling
from rasterio.io import DatasetReader, DatasetWriter
from rasterio.windows import Window
from tabulate import DataRow, Line, TableFormat, tabulate

from homonim.errors import HomonimError

logger = logging.getLogger(__name__)
tabulate.MIN_PADDING = 0

# tabulate format for comparison and parameter stats
table_format = TableFormat(
    lineabove=Line('', '-', ' ', ''),
    linebelowheader=Line('', '-', ' ', ''),
    linebetweenrows=None,
    linebelow=Line('', '-', ' ', ''),
    headerrow=DataRow('', ' ', ''),
    datarow=DataRow('', ' ', ''),
    padding=0,
    with_header_hide=['lineabove', 'linebelow'],
)

P = ParamSpec('P')
R = TypeVar('R')


def nan_equals(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Compare two numpy objects, returning True where elements of both are nan."""

    def _nan_equals(obj, scalar) -> np.ndarray:
        if np.isnan(scalar):
            return np.isnan(obj)
        else:
            return obj == scalar

    # use _nan_equals() to speed up the special cases where a or b is a scalar
    if np.ndim(a) == 0:
        return _nan_equals(b, a)
    elif np.ndim(b) == 0:
        return _nan_equals(a, b)
    else:
        return (a == b) | (np.isnan(a) & np.isnan(b))


def expand_window_to_grid(
    win: Window, expand_pixels: tuple[int, int] = (0, 0)
) -> Window:
    """Return the given RasterIO window expanded by expand_pixels to whole number
    extents.
    """
    col_off, col_frac = divmod(win.col_off - expand_pixels[1], 1)
    row_off, row_frac = divmod(win.row_off - expand_pixels[0], 1)
    width = ceil(win.width + 2 * expand_pixels[1] + col_frac)
    height = ceil(win.height + 2 * expand_pixels[0] + row_frac)
    return Window(int(col_off), int(row_off), width, height)


def round_window_to_grid(win: Window) -> Window:
    """Return the given RasterIO window with rounded extents."""
    ranges = win.toranges()
    row_range, col_range = [[round(s) for s in r] for r in ranges]
    return Window(
        col_off=col_range[0],
        row_off=row_range[0],
        width=col_range[1] - col_range[0],
        height=row_range[1] - row_range[0],
    )


def validate_threads(threads: int) -> int:
    """Validate and convert the threads parameter for concurent processing of image
    blocks.
    """
    # TODO: Memory increases ~linearly with number of threads, but does processing
    #  speed?  The bottleneck is often file IO & I am not sure >2 threads as a
    #  default is justified.
    if threads > os.cpu_count():
        raise HomonimError(
            f"'threads' should be less than or equal to the number of CPUs: "
            f'{os.cpu_count()}.'
        )
    return threads if threads > 0 else os.cpu_count()


def get_nonalpha_bands(im: DatasetReader | DatasetWriter) -> tuple[int, ...]:
    """Return a list of non-alpha band indices for the given Rasterio dataset."""
    bands = tuple(
        [bi + 1 for bi in range(im.count) if im.colorinterp[bi] != ColorInterp.alpha]
    )
    return bands


def assert_open(meth: Callable[P, R]) -> Callable[P, R]:
    """Method decorator for context managers that raises an error if a closed
    attribute is True.
    """

    @wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if args and getattr(args[0], 'closed', False):
            raise RuntimeError('Context is closed.')
        return meth(*args, **kwargs)

    return wrapper


def get_resampling(
    from_res: tuple[float, float],
    to_res: tuple[float, float],
    downsampling: Resampling,
    upsampling: Resampling,
) -> Resampling:
    """Return ``upsampling`` when ``from_res`` is a lower resolution than ``to_res``,
    otherwise return ``downsampling``.
    """
    from_area = abs(prod(from_res))
    to_area = abs(prod(to_res))
    return downsampling if from_area <= to_area else upsampling


def from_bounds(
    left: float, bottom: float, right: float, top: float, transform: Affine
) -> Window:
    """Return a window given its bounds and geo-referencing ``transform``.

    This is a replacement for ``rasterio.windows.from_bounds()`` that uses Python
    rather than NumPy floats for better precision.
    """
    cnrs = [(left, top), (right, top), (right, bottom), (left, bottom)]
    cols, rows = zip(*[~transform * cnr for cnr in cnrs], strict=True)
    row_start, row_stop = min(rows), max(rows)
    col_start, col_stop = min(cols), max(cols)
    return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
