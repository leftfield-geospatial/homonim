"""
    Homonim: Correction of aerial and satellite imagery to surface reflectance
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
from multiprocessing import cpu_count
from typing import Tuple, Dict, Union
from contextlib import contextmanager

import numpy as np
import rasterio as rio
from rasterio.enums import ColorInterp, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tabulate import TableFormat, Line, DataRow, tabulate

from homonim.enums import Model, ProcCrs
from homonim.errors import ImageFormatError

logger = logging.getLogger(__name__)
tabulate.MIN_PADDING = 0

##
table_format = TableFormat(
    lineabove=Line("", "-", " ", ""),
    linebelowheader=Line("", "-", " ", ""),
    linebetweenrows=None,
    linebelow=Line("", "-", " ", ""),
    headerrow=DataRow("", " ", ""),
    datarow=DataRow("", " ", ""),
    padding=0,
    with_header_hide=["lineabove", "linebelow"]
)  # yapf: disable
""" Tabulate format for comparison and parameter stats. """


def nan_equals(a: Union[np.ndarray, float], b: Union[np.ndarray, float]) -> np.ndarray:
    """ Compare two numpy objects a & b, returning true where elements of both a & b are nan. """
    return (a == b) | (np.isnan(a) & np.isnan(b))


def expand_window_to_grid(win: Window, expand_pixels: Tuple[int, int] = (0, 0)) -> Window:
    """
    Expand rasterio window extents to the nearest whole numbers i.e. for ``expand_pixels`` >= (0, 0), it will return a
    window that contains the original extents.

    Parameters
    ----------
    win: rasterio.windows.Window
        Window to expand.
    expand_pixels: tuple, optional
        Tuple specifying the number of (rows, columns) pixels to expand the window by.

    Returns
    -------
    rasterio.windows.Window
        Expanded window.
    """
    col_off, col_frac = np.divmod(win.col_off - expand_pixels[1], 1)
    row_off, row_frac = np.divmod(win.row_off - expand_pixels[0], 1)
    width = np.ceil(win.width + 2 * expand_pixels[1] + col_frac)
    height = np.ceil(win.height + 2 * expand_pixels[0] + row_frac)
    exp_win = Window(col_off.astype('int'), row_off.astype('int'), width.astype('int'), height.astype('int'))
    return exp_win


def round_window_to_grid(win: Window) -> Window:
    """
    Round window extents to the nearest whole numbers.

    Parameters
    ----------
    win: rasterio.windows.Window
        Window to round.

    Returns
    -------
    rasterio.windows.Window
        Rounded window with integer extents.
    """
    row_range, col_range = win.toranges()
    row_range = np.round(row_range).astype('int')
    col_range = np.round(col_range).astype('int')
    return Window(col_off=col_range[0], row_off=row_range[0], width=np.diff(col_range)[0], height=np.diff(row_range)[0])


def validate_kernel_shape(kernel_shape: Tuple[int, int], model: Model = Model.gain_blk_offset) -> Tuple[int, int]:
    """
    Check a kernel shape (height, width) tuple for validity.  Raises ValueError if ``kernel_shape`` is invalid.

    Parameters
    ----------
    kernel_shape: tuple of int
        Kernel (height, width) in pixels.
    model: Model, optional
        The model type ``kernel_shape`` will be used with.

    Returns
    -------
    tuple of int
        The validated kernel_shape as a numpy array.
    """
    kernel_shape = np.array(kernel_shape).astype(int)
    if not np.all(np.mod(kernel_shape, 2) == 1):
        raise ValueError('`kernel_shape` must be odd in both dimensions.')
    if model == Model.gain_offset:
        if np.product(kernel_shape) < 2:
            raise ValueError('`kernel_shape` area should contain at least 2 elements for the gain-offset model.')
        elif np.product(kernel_shape) < 25:
            logger.warning('A `kernel_shape` of at least 25 elements is recommended for the gain-offset model.')
    if not np.all(kernel_shape >= 1):
        raise ValueError('`kernel_shape` must be a minimum of one in both dimensions.')
    return tuple(kernel_shape)


def overlap_for_kernel(kernel_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Return the block overlap for a kernel shape.

    Parameters
    ----------
    kernel_shape: tuple of int
        Kernel (height, width) in pixels.

    Returns
    -------
    tuple of int
        Overlap (height, width) in integer pixels.
    """
    # Block overlap should be at least half the kernel 'shape' to ensure full kernel coverage at block edges, and a
    # minimum of (1, 1) to avoid including extrapolated (rather than interpolated) pixels when up-sampling.
    kernel_shape = np.array(kernel_shape).astype(int)
    return tuple(np.ceil(kernel_shape / 2).astype('int'))


def validate_threads(threads: int) -> int:
    """ Parse number of threads parameter. """
    _cpu_count = cpu_count()
    threads = _cpu_count if threads == 0 else threads
    if threads > _cpu_count:
        raise ValueError(f"'threads' is limited to the number of processors ({_cpu_count})")
    return threads


def create_out_postfix(proc_crs: ProcCrs, model: Model, kernel_shape: Tuple[int, int], driver: str = 'GTiff') -> str:
    """ Create a filename postfix, including extension, for the corrected image file. """
    ext_dict = rio.drivers.raster_driver_extensions()
    ext_idx = list(ext_dict.values()).index(driver)
    ext = list(ext_dict.keys())[ext_idx]
    post_fix = f'_FUSE_c{proc_crs.name.upper()}_m{model.upper()}_k{kernel_shape[0]}_{kernel_shape[1]}.{ext}'
    return post_fix


def create_param_filename(filename: Union[str, pathlib.Path]) -> pathlib.Path:
    """ Create a debug image filename, given the corrected image filename. """
    filename = pathlib.Path(filename)
    return filename.parent.joinpath(f'{filename.stem}_PARAM{filename.suffix}')


def north_up(im: rio.DatasetReader) -> bool:
    """ Return true if im is in a standard North-up orientation. """
    return(
        (np.sign(im.transform.a) == 1) and (np.sign(im.transform.e) == -1) and (im.transform.b == 0) and
        (im.transform.d == 0)
    )


def same_orientation_crs(
    src_im: rio.DatasetReader, ref_im: rio.DatasetReader, proc_crs: ProcCrs = None,
) -> Tuple[rio.DatasetReader, rio.DatasetReader]:  # yapf: disable
    """
    Re-project source and reference (as necessary) so that are they both oriented north-up, and in the same CRS.
    Re-project the proc_crs image (usually lower resolution) into the CRS of the other when the CRSs are not
    the same.
    """
    # Note: without transform etc arguments, WarpedVRT re-projects to north-up
    resampling = Resampling.bilinear
    same_crs = src_im.crs.to_proj4() == ref_im.crs.to_proj4()
    if (not north_up(src_im)) and (same_crs or proc_crs != ProcCrs.src):
        src_im = WarpedVRT(src_im, crs=src_im.crs, resampling=resampling)
    if (not north_up(ref_im)) and (same_crs or proc_crs == ProcCrs.src):
        ref_im = WarpedVRT(ref_im, crs=ref_im.crs, resampling=resampling)
    if (not same_crs) and (proc_crs == ProcCrs.src):
        src_im = WarpedVRT(src_im, crs=ref_im.crs, resampling=resampling)
    if (not same_crs) and (proc_crs != ProcCrs.src):
        ref_im = WarpedVRT(ref_im, crs=src_im.crs, resampling=resampling)
    return src_im, ref_im


@contextmanager
def same_orientation_crs_ctx(
    src_im: rio.DatasetReader, ref_im: rio.DatasetReader, proc_crs: ProcCrs = None,
) -> Tuple[rio.DatasetReader, rio.DatasetReader]:  # yapf: disable
    """
    Context manager to re-project source and reference (as necessary) so that are they both oriented north-up,
    and in the same CRS.
    """
    try:
        src_im, ref_im = same_orientation_crs(src_im, ref_im, proc_crs=proc_crs)
        yield (src_im, ref_im)
    finally:
        src_im.close()
        ref_im.close()


def covers_bounds(im1: rio.DatasetReader, im2: rio.DatasetReader, expand_pixels: Tuple[int, int] = (0, 0)) -> bool:
    """
    Determines if the spatial extents of one image cover another image

    Parameters
    ----------
    im1: rasterio.DatasetReader
        An open rasterio dataset.
    im2: rasterio.DatasetReader
        Another open rasterio dataset.
    expand_pixels: tuple of int, optional
        Expand the im2 bounds by this many pixels.

    Returns
    -------
    bool
        True if im1 covers im2 else False.
    """
    with same_orientation_crs_ctx(im1, im2) as (im1, im2):
        im1_win = im1.window(*im2.bounds)
    if not np.all(np.array(expand_pixels) == 0):
        im1_win = expand_window_to_grid(im1_win, expand_pixels)
    win_ul = np.array((im1_win.row_off, im1_win.col_off))
    win_shape = np.array((im1_win.height, im1_win.width))
    return False if np.any(win_ul < 0) or np.any(win_shape > im1.shape) else True


def get_nonalpha_bands(im: Union[rio.DatasetReader, rio.io.DatasetWriter]) -> Tuple[int, ...]:
    """
    Return a list of non-alpha band indices from a rasterio dataset.

    Parameters
    ----------
    im: rasterio.DatasetReader
        Retrieve band indices from this dataset.

    Returns
    -------
    list of int
        List of 1-based band indices.
    """
    bands = tuple([bi + 1 for bi in range(im.count) if im.colorinterp[bi] != ColorInterp.alpha])
    return bands


def combine_profiles(in_profile: Dict, config_profile: Dict) -> Dict:
    """
    Update an input rasterio profile with a configuration profile.

    Parameters
    ----------
    in_profile: dict
        Input/initial rasterio profile to update.  Driver-specific items are in the root dict.
    config_profile: dict
        Configuration profile.  Driver specific options are contained in a nested dict,
        with the ``creation_options`` key. E.g. see :meth:`homonim.RasterFuse.create_out_profile`.

    Returns
    -------
    dict
        Combined profile.
    """

    if in_profile['driver'].lower() != config_profile['driver'].lower():
        # copy only non-driver specific keys from input profile when the driver is different to the configured val
        copy_keys = ['driver', 'width', 'height', 'count', 'dtype', 'crs', 'transform']
        out_profile = {copy_key: in_profile[copy_key] for copy_key in copy_keys}
    else:
        out_profile = in_profile.copy()  # copy the whole input profile

    def nested_update(self_dict, other_dict):
        """ Update self_dict with a flattened version of other_dict. """
        for other_key, other_value in other_dict.items():
            if isinstance(other_value, dict):
                # flatten the driver specific nested dict into the root dict
                nested_update(self_dict, other_value)
            # elif other_value is not None:
            else:
                self_dict[other_key] = other_value
        return self_dict

    # update out_profile with a flattened config_profile
    return nested_update(out_profile, config_profile)


def validate_param_image(param_filename: Union[str, pathlib.Path]):
    """ Check file is a valid parameter image. """
    param_filename = pathlib.Path(param_filename)
    if not param_filename.exists():
        raise FileNotFoundError(f'{param_filename} does not exist')

    with rio.open(param_filename) as param_im:
        tags = param_im.tags()
        # check band count is a multiple of 3 and that expected metadata tags exist
        if (param_im.count == 0 or divmod(param_im.count, 3)[1] != 0 or
                not {'FUSE_MODEL', 'FUSE_KERNEL_SHAPE', 'FUSE_PROC_CRS', 'FUSE_REF_FILE'} <= set(tags)):
            raise ImageFormatError(f'{param_filename.name} is not a valid parameter image.')

        # check band descriptions end with the expected suffixes
        n_refl_bands = int(param_im.count / 3)
        suffixes = ['gain'] * n_refl_bands + ['offset'] * n_refl_bands + ['r2'] * n_refl_bands
        if not all([desc.lower().endswith(suffix) for suffix, desc in zip(suffixes, param_im.descriptions)]):
            raise ImageFormatError(f'{param_filename.name} is not a valid parameter image.')
