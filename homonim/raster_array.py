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
from multiprocessing.dummy import Value

import numpy as np
import rasterio as rio
from rasterio import windows
from rasterio import transform
from rasterio.crs import CRS
from rasterio import Affine
from rasterio.warp import reproject, Resampling
from homonim import get_logger, hom_dtype, hom_nodata
import multiprocessing
from rasterio.enums import ColorInterp

def nan_equals(a, b, equal_nan=True):
    if not equal_nan:
        return (a == b)
    else:
        return ((a == b) | (np.isnan(a) & np.isnan(b)))

class RasterArray(object):
    """A class for wrapping a geo-referenced numpy array"""
    _default_nodata = hom_nodata
    _default_dtype = hom_dtype
    def __init__(self, array, crs, transform, nodata=None, window=None):
        # array = np.array(array)
        if (array.ndim < 2) or (array.ndim > 3):
            raise ValueError('`array` must be a 2D or 3D numpy array')
        self._array = array

        if window is not None:
            if (window.height, window.width) != array.shape[-2:]:
                raise ValueError('window and array dimensions must match')

        if isinstance(crs, CRS):
            self._crs = crs
        else:
            raise TypeError('crs must be an instance of rasterio.CRS')

        if isinstance(transform, Affine):
            if window is not None:
                self._transform = windows.transform(window, transform)
            else:
                self._transform = transform
        else:
            raise TypeError('transform must be an instance of rasterio.Affine')

        self._nodata = nodata
        self._nodata_mask = None

    @classmethod
    def from_profile(cls, array, profile, window=None):
        if not ('crs' and 'transform' and 'nodata' in profile):
            raise Exception('profile should include crs, transform and nodata keys')
        return cls(array, profile['crs'], profile['transform'], nodata=profile['nodata'], window=window)

    @classmethod
    def from_rio_dataset(cls, rio_dataset, indexes=None, window=None, boundless=False):
        array = rio_dataset.read(indexes=indexes, window=window, boundless=boundless, out_dtype=cls._default_dtype)
        is_alpha = [band_cinterp == ColorInterp.alpha for band_cinterp in rio_dataset.colorinterp]
        if rio_dataset.nodata is not None and not any(is_alpha):
            nodata = rio_dataset.nodata
        else:
            nodata = cls._default_nodata
            mask = rio_dataset.dataset_mask()
            array[~mask] = nodata
        return cls(array, rio_dataset.crs, rio_dataset.transform, nodata=nodata, window=window)

    @property
    def array(self):
        return self._array

    # @array.setter
    # def array(self, array):
    #     if array.shape != self._array.shape:
    #         raise ValueError('Array must be same shape as RasterArray.array')
    #     if (array.ndim < 2) or (array.ndim > 3):
    #         raise ValueError('Array must a 2 or 3D numpy array')
    #     self._array = array

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
        return dict(crs=self._crs, transform=self._transform, nodata=self._nodata, count=self.count, width=self.width,
                    height=self.height, bounds=self.bounds)

    @property
    def proj_profile(self):
        return dict(crs=self._crs, transform=self._transform, shape=self._array.shape)

    @property
    def mask(self):
        """ 2D boolean mask corresponding to valid pixels in array """
        if self._nodata is None:
            return np.full(self.shape, True)
        mask = ~nan_equals(self.array, self.nodata)
        if self._array.ndim > 2:
            mask =  np.all(mask, axis=0)
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

    def reproject(self, crs=None, transform=None, shape=None, nodata=_default_nodata, dtype=_default_dtype,
                  resampling=Resampling.lanczos):

        if transform and not shape:
            raise ValueError('If transform is specified, shape must also be specified')

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

