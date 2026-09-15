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
from contextlib import contextmanager
from multiprocessing import cpu_count
from os import PathLike
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.enums import ColorInterp, Resampling
from rasterio.io import DatasetReader, DatasetWriter
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tabulate import DataRow, Line, TableFormat, tabulate

from homonim.enums import Model, ProcCrs
from homonim.errors import ImageFormatError

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


def nan_equals(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Compare two numpy objects, returning True where elements of both are nan."""

    def _nan_equals(obj, scalar) -> np.ndarray:
        if np.isnan(scalar):
            return np.isnan(obj)
        else:
            return obj == scalar

    # use _nan_equals() to speed up the special cases where a or b is a scalar
    if np.isscalar(a):
        return _nan_equals(b, a)
    elif np.isscalar(b):
        return _nan_equals(a, b)
    else:
        return (a == b) | (np.isnan(a) & np.isnan(b))


def expand_window_to_grid(
    win: Window, expand_pixels: tuple[int, int] = (0, 0)
) -> Window:
    """Return the given RasterIO window expanded by expand_pixels to whole number
    extents.
    """
    col_off, col_frac = np.divmod(win.col_off - expand_pixels[1], 1)
    row_off, row_frac = np.divmod(win.row_off - expand_pixels[0], 1)
    width = np.ceil(win.width + 2 * expand_pixels[1] + col_frac)
    height = np.ceil(win.height + 2 * expand_pixels[0] + row_frac)
    exp_win = Window(int(col_off), int(row_off), int(width), int(height))
    return exp_win


def round_window_to_grid(win: Window) -> Window:
    """Return the given RasterIO window with rounded extents."""
    row_range, col_range = win.toranges()
    row_range = np.round(row_range).astype('int')
    col_range = np.round(col_range).astype('int')
    return Window(
        col_off=col_range[0],
        row_off=row_range[0],
        width=np.diff(col_range)[0],
        height=np.diff(row_range)[0],
    )


def validate_threads(threads: int) -> int:
    """Validate the number of threads for concurent processing of image blocks."""
    # TODO: Memory increases ~linearly with number of threads, but does processing
    #  speed?  The bottleneck is often file IO & I am not sure >2 threads as a
    #  default is justified.
    _cpu_count = cpu_count()
    threads = _cpu_count if threads == 0 else threads
    if threads < 0 or threads > _cpu_count:
        raise ValueError(
            f"'threads' should be greater than or equal to zero, and less than the "
            f'number of processors ({_cpu_count})'
        )
    return threads


def north_up(im: DatasetReader) -> bool:
    """Return True if im is in a standard North-up orientation."""
    return (
        (np.sign(im.transform.a) == 1)
        and (np.sign(im.transform.e) == -1)
        and (im.transform.b == 0)
        and (im.transform.d == 0)
    )


def same_orientation_crs(
    src_im: DatasetReader, ref_im: DatasetReader, proc_crs: ProcCrs = None
) -> tuple[DatasetReader, DatasetReader]:
    """Reproject src_im and ref_im (as necessary) so they are both oriented north-up,
    and in the same CRS.  Reproject the proc_crs image into the CRS of the other when
    their CRSs are not the same.
    """
    # Note: without transform etc arguments, WarpedVRT re-projects to north-up
    resampling = Resampling.bilinear
    same_crs = src_im.crs == ref_im.crs
    if not north_up(src_im) and (same_crs or proc_crs is not ProcCrs.src):
        src_im = WarpedVRT(src_im, crs=src_im.crs, resampling=resampling)
    if not north_up(ref_im) and (same_crs or proc_crs is ProcCrs.src):
        ref_im = WarpedVRT(ref_im, crs=ref_im.crs, resampling=resampling)
    if not same_crs and (proc_crs is ProcCrs.src):
        src_im = WarpedVRT(src_im, crs=ref_im.crs, resampling=resampling)
    if not same_crs and (proc_crs is not ProcCrs.src):
        ref_im = WarpedVRT(ref_im, crs=src_im.crs, resampling=resampling)
    return src_im, ref_im


@contextmanager
def same_orientation_crs_ctx(
    src_im: DatasetReader, ref_im: DatasetReader, proc_crs: ProcCrs = None
) -> tuple[DatasetReader, DatasetReader]:
    """Context manager wrapping same_orientation_crs()."""
    try:
        src_im, ref_im = same_orientation_crs(src_im, ref_im, proc_crs=proc_crs)
        yield (src_im, ref_im)
    finally:
        # TODO: this may be closing the dataset(s) if they are not wrapped in WarpedVRT
        #  by same_orientation_crs()
        src_im.close()
        ref_im.close()


def covers_bounds(
    im1: DatasetReader, im2: DatasetReader, expand_pixels: tuple[int, int] = (0, 0)
) -> bool:
    """Return True if the extents of im1 encompass those of im2 expanded by
    expand_pixels, otherwise False.
    """
    with same_orientation_crs_ctx(im1, im2) as (im1, im2):
        im1_win = im1.window(*im2.bounds)
    if not np.all(np.array(expand_pixels) == 0):
        im1_win = expand_window_to_grid(im1_win, expand_pixels)
    win_ul = np.array((im1_win.row_off, im1_win.col_off))
    win_shape = np.array((im1_win.height, im1_win.width))
    return False if np.any(win_ul < 0) or np.any(win_shape > im1.shape) else True


def get_nonalpha_bands(im: DatasetReader | DatasetWriter) -> tuple[int, ...]:
    """Return a list of non-alpha band indices for the given Rasterio dataset."""
    bands = tuple(
        [bi + 1 for bi in range(im.count) if im.colorinterp[bi] != ColorInterp.alpha]
    )
    return bands


def validate_param_image(param_file: str | PathLike):
    """Validate the given parameter image."""
    # TODO: move to stats module & modify to work with URI if necessary
    param_file = Path(param_file)
    if not param_file.exists():
        raise FileNotFoundError(f'{param_file} does not exist')

    with rio.open(param_file) as param_im:
        tags = param_im.tags()
        # check band count is a multiple of 3 and that expected metadata tags exist
        if (
            param_im.count == 0
            or divmod(param_im.count, 3)[1] != 0
            or not {'FUSE_MODEL', 'FUSE_KERNEL_SHAPE', 'FUSE_PROC_CRS', 'FUSE_REF_FILE'}
            <= set(tags)
        ):
            raise ImageFormatError(f'{param_file.name} is not a valid parameter image.')

        # check band descriptions end with the expected suffixes
        n_refl_bands = int(param_im.count / 3)
        suffixes = (
            ['gain'] * n_refl_bands + ['offset'] * n_refl_bands + ['r2'] * n_refl_bands
        )
        if not all(
            [
                desc.lower().endswith(suffix)
                for suffix, desc in zip(suffixes, param_im.descriptions, strict=True)
            ]
        ):
            raise ImageFormatError(f'{param_file.name} is not a valid parameter image.')
