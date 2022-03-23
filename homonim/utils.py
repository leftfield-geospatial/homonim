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
from homonim.enums import Method
from homonim.errors import ImageFormatError
from rasterio.enums import Resampling, ColorInterp
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, get_data_window

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
    if method == Method.gain_offset and not np.product(kernel_shape) >= 25:
        raise ValueError("kernel shape area should contain at least 25 elements for the gain-offset method.")
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


def create_homo_postfix(proc_crs, method, kernel_shape, driver='GTiff'):
    """Create a filename postfix, including extension, for the homogenised image file"""
    ext_dict = rio.drivers.raster_driver_extensions()
    ext_idx = list(ext_dict.values()).index(driver)
    ext = list(ext_dict.keys())[ext_idx]
    post_fix = f'_HOMO_c{proc_crs.name.upper()}_m{method.upper()}_k{kernel_shape[0]}_{kernel_shape[1]}.{ext}'
    return post_fix


def create_param_filename(filename: pathlib.Path):
    """Create a debug image filename, given the homogenised image filename"""
    filename = pathlib.Path(filename)
    return filename.parent.joinpath(f'{filename.stem}_PARAM{filename.suffix}')


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


def combine_profiles(in_profile, config_profile):
    """
    Update an input rasterio profile with a configuration profile.

    Parameters
    ----------
    in_profile: dict
        The input/initial rasterio profile to update.  Driver-specific items are in the root dict.
    config_profile: dict
        The configuration profile.  Driver specific options are contained in a nested dict, with 'creation_options' key.
        E.g. see homonim.fuse.RasterFuse.default_out_profile.

    Returns
    -------
    out_profile: dict
        The combined profile.
    """

    if in_profile['driver'].lower() != config_profile['driver'].lower():
        # copy only non driver specific keys from input profile when the driver is different to the configured val
        copy_keys = ['driver', 'width', 'height', 'count', 'dtype', 'crs', 'transform']
        out_profile = {copy_key: in_profile[copy_key] for copy_key in copy_keys}
    else:
        out_profile = in_profile.copy()  # copy the whole input profile

    def nested_update(self_dict, other_dict):
        """Update self_dict with a flattened version of other_dict"""
        for other_key, other_value in other_dict.items():
            if isinstance(other_value, dict):
                # flatten the driver specific nested dict into the root dict
                nested_update(self_dict, other_value)
            elif other_value is not None:
                self_dict[other_key] = other_value
        return self_dict

    # update out_profile with a flattened config_profile
    return nested_update(out_profile, config_profile)

def validate_param_image(param_filename):
    """Check file is a valid parameter image"""
    if not param_filename.exists():
        raise FileNotFoundError(f'{param_filename} does not exist')

    with rio.open(param_filename) as param_im:
        tags = param_im.tags()
        # check band count is a multiple of 3 and that expected metadata tags exist
        if (param_im.count == 0 or divmod(param_im.count, 3)[1] != 0 or
                not {'HOMO_METHOD', 'HOMO_MODEL_CONF', 'HOMO_PROC_CRS'} <= set(tags)):
            raise ImageFormatError(f'{param_filename.name} is not a valid parameter image.')

        # check band descriptions end with the expected suffixes
        n_refl_bands = int(param_im.count / 3)
        suffixes = ['gain'] * n_refl_bands + ['offset'] * n_refl_bands + ['r2'] * n_refl_bands
        if not all([desc.lower().endswith(suffix) for suffix, desc in zip(suffixes, param_im.descriptions)]):
            raise ImageFormatError(f'{param_filename.name} is not a valid parameter image.')
