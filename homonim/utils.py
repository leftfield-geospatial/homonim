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

import numpy as np
import rasterio
from rasterio.windows import Window


def nan_equals(a, b):
    """Compare two numpy objects a & b, returning true where elements of both a & b are nan"""
    return (a == b) | (np.isnan(a) & np.isnan(b))


def expand_window_to_grid(win, expand_pixels=(0, 0)):
    """
    Expand rasterio window extents to nearest whole numbers i.e. for expand_pixels >= (0, 0), it will return a window
    that contains the original extents.

    Parameters
    ----------
    win: rasterio.windows.Window
        The window to expand.
    expand_pixels: tuple, optional
        A tuple specifying the number of (rows, columns) pixels to expand the window by.
        [default: (0, 0)]

    Returns
    -------
    win: rasterio.windows.Window
         The expanded window.
    """
    col_off, col_frac = np.divmod(win.col_off - expand_pixels[1], 1)
    row_off, row_frac = np.divmod(win.row_off - expand_pixels[0], 1)
    width = np.ceil(win.width + 2 * expand_pixels[1] + col_frac)
    height = np.ceil(win.height + 2 * expand_pixels[0] + row_frac)
    exp_win = Window(col_off.astype('int'), row_off.astype('int'), width.astype('int'), height.astype('int'))
    return exp_win


def round_window_to_grid(win):
    """
    Round window extents to the nearest whole numbers.

    Parameters
    ----------
    win: rasterio.windows.Window
        The window to round.

    Returns
    -------
    win: rasterio.windows.Window
        The rounded window with integer extents.
    """
    row_range, col_range = win.toranges()
    row_range = np.round(row_range).astype('int')
    col_range = np.round(col_range).astype('int')
    return Window(col_off=col_range[0], row_off=row_range[0], width=np.diff(col_range)[0], height=np.diff(row_range)[0])


def check_kernel_shape(kernel_shape):
    """
    Check a kernel_shape (height, width) tuple for validity.  Raises ValueError if kernel_shape is invalid.

    Parameters
    ----------
    kernel_shape: tuple
        The kernel (height, width) in pixels.

    Returns
    -------
    kernel_shape: numpy.array
        The validated kernel_shape as a numpy array.
    """
    kernel_shape = np.array(kernel_shape).astype(int)
    if not np.all(np.mod(kernel_shape, 2) == 1):
        raise ValueError("kernel shape must be odd in both dimensions.")
    if not np.all(kernel_shape >= 1):
        raise ValueError("kernel shape must be a minimum of one in both dimensions.")
    return kernel_shape


def overlap_for_kernel(kernel_shape):
    """
    Return the block overlap for a kernel shape.

    Parameters
    ----------
    kernel_shape: tuple
        The kernel (height, width) in pixels.

    Returns
    -------
    overlap: numpy.array
        The overlap (height, width) in integer pixels as a numpy.array.
    """
    kernel_shape = np.array(kernel_shape).astype(int)
    return np.ceil(kernel_shape / 2).astype('int')
