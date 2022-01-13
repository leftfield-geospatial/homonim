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
from homonim.enums import ProcCrs


def nan_equals(a, b):
    """Compare two numpy objects, returning true if both a & b elements are nan"""
    return ((a == b) | (np.isnan(a) & np.isnan(b)))


def expand_window_to_grid(win, expand_pixels=(0, 0)):
    """
    Expands decimal window extents.

    For expand_pixel=(0,0) window extents are expanded to the nearest integers that include the original extents.

    Parameters
    ----------
    win: rasterio.windows.Window
         The window to expand.
    expand_pixels: numpy.array_like, List[float, float], tuple, optional
                   A two element iterable (rows, columns) specifying the number of pixels to expand the window by.

    Returns
    -------
    win: rasterio.windows.Window
         The expanded window
    """
    col_off, col_frac = np.divmod(win.col_off - expand_pixels[1], 1)
    row_off, row_frac = np.divmod(win.row_off - expand_pixels[0], 1)
    width = np.ceil(win.width + 2 * expand_pixels[1] + col_frac)
    height = np.ceil(win.height + 2 * expand_pixels[0] + row_frac)
    exp_win = Window(col_off.astype('int'), row_off.astype('int'), width.astype('int'), height.astype('int'))
    return exp_win


def round_window_to_grid(win):
    """
    Rounds decimal window extents to the nearest integers.

    Parameters
    ----------
    win: rasterio.windows.Window
         The window to round.

    Returns
    -------
    win: rasterio.windows.Window
         The rounded window.
    """
    row_range, col_range = win.toranges()
    row_range = np.round(row_range).astype('int')
    col_range = np.round(col_range).astype('int')
    return Window(col_off=col_range[0], row_off=row_range[0], width=np.diff(col_range)[0], height=np.diff(row_range)[0])

def check_kernel_shape(kernel_shape):
    kernel_shape = np.array(kernel_shape).astype(int)
    if not np.all(np.mod(kernel_shape, 2) == 1):
        raise ValueError("'kernel_shape' must be odd in both dimensions")
    if not np.all(kernel_shape >= 1):
        raise ValueError("'kernel_shape' must be a minimum of one in both dimensions")
    return kernel_shape

def overlap_for_kernel(kernel_shape):
    kernel_shape = np.array(kernel_shape).astype(int)
    return np.ceil(kernel_shape/2).astype('int')
