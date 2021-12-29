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

import multiprocessing

import numpy as np
from rasterio import Affine
from rasterio import transform
from rasterio import windows
from rasterio.crs import CRS
from rasterio.enums import MaskFlags
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window

from homonim.errors import ImageProfileError


def nan_equals(a, b, equal_nan=True):
    if not equal_nan:
        return (a == b)
    else:
        return ((a == b) | (np.isnan(a) & np.isnan(b)))


def expand_window_to_grid(win, expand_pixels=(0, 0)):
    """
    Expands float window extents to be integers that include the original extents

    Parameters
    ----------
    win : rasterio.windows.Window
        the window to expand

    Returns
    -------
    exp_win: rasterio.windows.Window
        the expanded window
    """
    col_off, col_frac = np.divmod(win.col_off - expand_pixels[1], 1)
    row_off, row_frac = np.divmod(win.row_off - expand_pixels[0], 1)
    width = np.ceil(win.width + 2 * expand_pixels[1] + col_frac)
    height = np.ceil(win.height + 2 * expand_pixels[0] + row_frac)
    exp_win = Window(col_off.astype('int'), row_off.astype('int'), width.astype('int'), height.astype('int'))
    return exp_win


def round_window_to_grid(win):
    """
    Rounds float window extents to nearest integer

    Parameters
    ----------
    win : rasterio.windows.Window
        the window to round

    Returns
    -------
    exp_win: rasterio.windows.Window
        the rounded window
    """
    row_range, col_range = win.toranges()
    row_range = np.round(row_range).astype('int')
    col_range = np.round(col_range).astype('int')
    return Window(col_off=col_range[0], row_off=row_range[0], width=np.diff(col_range)[0], height=np.diff(row_range)[0])


class RasterArray(transform.TransformMethodsMixin, windows.WindowMethodsMixin):
    """
    A class for wrapping and re-projecting a geo-referenced numpy array.
    Internally masking is done using a nodata value, not a separately stored mask.
    By default internal data type is float32 and the nodata value is nan.
    """
    default_nodata = float('nan')
    default_dtype = 'float32'

    def __init__(self, array, crs, transform, nodata=default_nodata, window=None):
        # array = np.array(array)
        if (array.ndim < 2) or (array.ndim > 3):
            raise ValueError('"array" must be have 2 or 3 dimensions with bands along the first dimension')
        self._array = array

        if window is not None:
            if (window.height, window.width) != array.shape[-2:]:
                raise ValueError('"window" and "array" width and height must match')

        if isinstance(crs, CRS):
            self._crs = crs
        else:
            raise TypeError('"crs" must be an instance of rasterio.CRS')

        if isinstance(transform, Affine):
            if window is not None:
                self._transform = windows.transform(window, transform)
            else:
                self._transform = transform
        else:
            raise TypeError('"transform" must be an instance of rasterio.Affine')

        self._nodata = nodata
        self._nodata_mask = None

    @classmethod
    def from_profile(cls, array, profile, window=None):
        if not ('crs' and 'transform' and 'nodata' in profile):
            raise ImageProfileError('"profile" should include "crs", "transform" and "nodata" keys')
        if array is None:  # create array filled with nodata
            if not ('width' and 'height' and 'count' and 'dtype' in profile):
                raise ImageProfileError('"profile" should include "width", "height", "count" and "dtype" keys')
            array_shape = (profile['count'], profile['height'], profile['width'])
            array = np.full(array_shape, fill_value=profile['nodata'], dtype=profile['dtype'])
        return cls(array, profile['crs'], profile['transform'], nodata=profile['nodata'], window=window)

    @classmethod
    def from_rio_dataset(cls, rio_dataset, indexes=None, window=None, boundless=False):
        # check bands if bands have masks (i.e. internal/side-car mask or alpha channel, as opposed to nodata value)
        index_list = [indexes] if np.isscalar(indexes) else indexes
        is_masked = any([MaskFlags.per_dataset in rio_dataset.mask_flag_enums[bi - 1] for bi in index_list])

        # force nodata to default if masked or dataset nodata is None
        nodata = cls.default_nodata if (is_masked or rio_dataset.nodata is None) else rio_dataset.nodata
        array = rio_dataset.read(indexes=indexes, window=window, boundless=boundless, out_dtype=cls.default_dtype,
                                 fill_value=nodata)
        if is_masked:
            # read mask from dataset and apply it to array
            mask = rio_dataset.dataset_mask(window=window, boundless=boundless).astype('bool', copy=False)
            array[~mask] = nodata

        return cls(array, rio_dataset.crs, rio_dataset.transform, nodata=nodata, window=window)

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, value):
        if np.all(value.shape[-2:] == self._array.shape[-2:]):
            self._array = value
        else:
            raise ValueError('"value" and current width and height must match')

    @property
    def crs(self):
        return self._crs

    @property
    def width(self):
        return self.shape[-1]

    @property
    def height(self):
        return self.shape[-2]

    @property
    def shape(self):
        return self._array.shape[-2:]

    @property
    def count(self):
        return self._array.shape[0] if self.array.ndim == 3 else 1

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def transform(self):
        return self._transform

    @property
    def res(self):
        return np.abs((self._transform.a, self._transform.e))

    @property
    def bounds(self):
        return windows.bounds(windows.Window(0, 0, self.width, self.height), self._transform)

    @property
    def profile(self):
        return dict(crs=self._crs, transform=self._transform, nodata=self._nodata, count=self.count,
                    width=self.width, height=self.height, bounds=self.bounds, dtype=self.dtype)

    @property
    def proj_profile(self):
        return dict(crs=self._crs, transform=self._transform, shape=self.shape)

    @property
    def mask(self):
        """ 2D boolean mask corresponding to valid pixels in array """
        if self._nodata is None:
            return np.full(self.shape, True)
        mask = ~nan_equals(self.array, self.nodata)
        if self._array.ndim > 2:
            mask = np.all(mask, axis=0)
        return mask

    @mask.setter
    def mask(self, value):
        """ 2D boolean mask corresponding to valid pixels in array """
        if self._array.ndim == 2:
            self._array[~value] = self._nodata
        else:
            self._array[:, ~value] = self._nodata

    @property
    def nodata(self):
        """ nodata value """
        return self._nodata

    @nodata.setter
    def nodata(self, value):
        """ nodata value """
        if value is None or self._nodata is None:
            # if value is None, remove the mask, if current nodata is None,
            # there is no mask to apply the new value to array
            self._nodata = value
        elif not (nan_equals(value, self._nodata)):
            # if value is different to current nodata, set mask area in array to value
            nodata_mask = ~self.mask
            if self._array.ndim == 3:
                self._array[:, nodata_mask] = value
            else:
                self._array[nodata_mask] = value
            self._nodata = value

    def copy(self, deep=True):
        array = self._array.copy() if deep else self._array
        return RasterArray.from_profile(array, self.profile)

    def slice_array(self, *bounds):
        window = self.window(*bounds)
        window = round_window_to_grid(window)
        if self._array.ndim == 2:
            return self._array[window.toslices()]
        else:
            return self._array[(slice(self._array.shape[0]), *window.toslices())]

    def reproject(self, crs=None, transform=None, shape=None, nodata=default_nodata, dtype=default_dtype,
                  resampling=Resampling.lanczos):

        if transform and not shape:
            raise ValueError('If "transform" is specified, "shape" must also be specified')

        if isinstance(resampling, str):
            resampling = Resampling[resampling]

        crs = crs or self._crs
        shape = shape or self._array.shape
        dtype = dtype or self._array.dtype

        if self.array.ndim > 2:
            _dst_array = np.zeros((self._array.shape[0], *shape), dtype=dtype)
        else:
            _dst_array = np.zeros(shape, dtype=dtype)

        _, _dst_transform = reproject(
            self._array,
            destination=_dst_array,
            src_crs=self._crs,
            src_transform=self._transform,
            src_nodata=self._nodata,
            dst_crs=crs,
            dst_transform=transform,
            dst_nodata=nodata,
            num_threads=multiprocessing.cpu_count(),
            resampling=resampling,
        )
        return RasterArray(_dst_array, crs=crs, transform=_dst_transform, nodata=nodata)

##
