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
from collections import OrderedDict
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.enums import Resampling, ColorInterp
from rasterio.windows import Window, get_data_window
from rasterio.vrt import WarpedVRT

from homonim.enums import Method

logger = logging.getLogger(__name__)


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


def validate_kernel_shape(kernel_shape, method=Method.gain_blk_offset):
    """
    Check a kernel_shape (height, width) tuple for validity.  Raises ValueError if kernel_shape is invalid.

    Parameters
    ----------
    kernel_shape: tuple
        The kernel (height, width) in pixels.
    method: Method, optional
        The modelling method kernel_shape will be used with.

    Returns
    -------
    kernel_shape: numpy.array
        The validated kernel_shape as a numpy array.
    """
    kernel_shape = np.array(kernel_shape).astype(int)
    if not np.all(np.mod(kernel_shape, 2) == 1):
        raise ValueError("kernel shape must be odd in both dimensions.")
    if method == Method.gain_offset:
        if not np.product(kernel_shape) >= 25:
            raise ValueError("kernel shape should contain at least 25 elements for the gain-offset method.")
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
    # Block overlap should be at least half the kernel 'shape' to ensure full kernel coverage at block edges, and a
    # minimum of (1, 1) to avoid including extrapolated (rather than interpolated) pixels when up-sampling.
    kernel_shape = np.array(kernel_shape).astype(int)
    return np.ceil(kernel_shape / 2).astype('int')


def validate_threads(threads):
    """Parse number of threads parameter."""
    _cpu_count = cpu_count()
    threads = _cpu_count if threads == 0 else threads
    if threads > _cpu_count:
        raise ValueError(f"'threads' is limited to the number of processors ({_cpu_count})")
    return threads


def create_debug_filename(filename):
    """Create a debug image filename, given the homogenised image filename"""
    filename = pathlib.Path(filename)
    return filename.parent.joinpath(f'{filename.stem}_DEBUG{filename.suffix}')


def build_overviews(filename):
    """
    Build internal overviews for an existing image file.

    Parameters
    ----------
    filename: str, pathlib.Path
              Path to the image file to build overviews for.
    """
    filename = pathlib.Path(filename)

    if not filename.exists():
        raise Exception(f'{filename} does not exist')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'r+') as im:
        # limit overviews so that the highest level has at least 2**8=256 pixels along the shortest dimension,
        # and so there are no more than 8 levels.
        max_ovw_levels = int(np.min(np.log2(im.shape)))
        num_ovw_levels = np.min([8, max_ovw_levels - 8])
        ovw_levels = [2 ** m for m in range(1, num_ovw_levels + 1)]
        im.build_overviews(ovw_levels, Resampling.average)


def debug_stats(dbg_filename, method, r2_inpaint_thresh):
    dbg_filename = pathlib.Path(dbg_filename)
    if not dbg_filename.exists():
        raise FileNotFoundError(f'{dbg_filename} does not exist')

    with rio.open(dbg_filename, 'r') as im:
        band_dict = {}
        band_desc = im.descriptions
        if len(np.unique(band_desc)) != im.count:
            band_desc = [f'Band {i + 1}' for i in range(im.count)]
        _mask = im.dataset_mask()
        win = get_data_window(_mask, nodata=0)
        mask = im.dataset_mask(window=win).astype('bool', copy=False)
        for band_i in range(im.count):
            band_array = im.read(indexes=band_i + 1, window=win, out_dtype='float32')
            band_vec = band_array[mask]

            def stats(v):
                vm = np.ma.masked_invalid(v)
                return OrderedDict(Mean=vm.mean(), Std=vm.std(), Min=vm.min(), Max=vm.max())

            stats_dict = stats(band_vec)
            if (method == Method.gain_offset) and (band_i >= im.count * 2 / 3):
                inpaint_portion = np.nansum(band_vec < r2_inpaint_thresh) / len(~np.isnan(band_vec))
                stats_dict['Inpaint (%)'] = inpaint_portion * 100

            band_dict[band_desc[band_i]] = stats_dict

        band_df = pd.DataFrame.from_dict(band_dict, orient='index')
        band_str = band_df.to_string(float_format="{:.2f}".format, index=True, justify="center",
                                     index_names=False)
        return band_dict, band_str


def covers_bounds(im1, im2, expand_pixels=(0, 0)):
    """
    Determines if the spatial extents of one image cover another image

    Parameters
    ----------
    im1: rasterio.DatasetReader
        An open rasterio dataset.
    im2: rasterio.DatasetReader
        Another open rasterio dataset.
    expand_pixels: Tuple[int, int], optional
        Expand the im2 bounds by this many pixels.

    Returns
    -------
    covers_bounds: bool
        True if im1 covers im2 else False.
    """
    # use WarpedVRT to get the datasets in the same crs
    _im2 = WarpedVRT(im2, crs=im1.crs) if im1.crs != im2.crs else im2
    im1_win = im1.window(*_im2.bounds)
    if not np.all(np.array(expand_pixels) == 0):
        im1_win = expand_window_to_grid(im1_win, expand_pixels)
    win_ul = np.array((im1_win.row_off, im1_win.col_off))
    win_shape = np.array((im1_win.height, im1_win.width))
    return False if np.any(win_ul < 0) or np.any(win_shape > im1.shape) else True


def get_nonalpha_bands(im):
    """
    Return a list of non-alpha band indices from a rasterio dataset.

    Parameters
    ----------
    im: rasterio.DatasetReader
        Retrieve band indices from this dataset.

    Returns
    -------
    bands: list[int, ]
        The list of 1-based band indices.
    """
    bands = tuple([bi + 1 for bi in range(im.count) if im.colorinterp[bi] != ColorInterp.alpha])
    return bands