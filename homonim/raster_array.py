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

class RasterArray(object):
    """A class for wrapping a geo-referenced numpy array"""
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
        return self._array.shape

    @property
    def count(self):
        return self.shape[0] if len(self.shape) == 3 else 1

    @property
    def transform(self):
        return self._transform

    @property
    def nodata(self):
        return self._nodata

    @property
    def res(self):
        return np.abs((self.transform.a, self.transform.e))

    @property
    def bounds(self):
        return windows.bounds(windows.Window(0, 0, self.width, self.height), self.transform)

    @property
    def profile(self):
        return dict(crs=self.crs, transform=self.transform, nodata=self.nodata, count=self.count, width=self.width,
                    height=self.height, bounds=self.bounds)

    @property
    def proj_profile(self):
        return dict(crs=self.crs, transform=self.transform, shape=self.shape)

    @property
    def mask(self):
        if self._nodata is None:
            return None

        if self._array.ndim == 3:
            mask =  np.all(self._array != self._nodata, axis=0).astype('uint8')
        else:
            mask = (self._array != self._nodata).astype('uint8')

        return RasterArray(mask, crs=self.crs, transform=self.transform, nodata=None)

    # @property
    # def nodata_mask(self):
    #     if self._nodata_mask is None and self._nodata is not None:
    #         self._nodata_mask = (self._array == self._nodata)
    #     return self._nodata_mask
    #
    # @nodata.setter
    # def nodata(self, value):
    #     if value != self._nodata:
    #         self.array[self.nodata_mask] = value
    #         self._nodata = value



    def reproject(self, crs=None, transform=None, shape=None, nodata=hom_nodata, resampling=Resampling.lanczos):

        if transform and not shape:
            raise ValueError('if transform is specified, shape must also be specified')

        if isinstance(resampling, str):
            resampling = Resampling[resampling]

        crs = crs or self._crs
        shape = shape or self._array.shape

        if self.array.ndim > 2:
            _dst_array = np.zeros((self._array.shape[0], *shape), dtype=self._array.dtype)
        else:
            _dst_array = np.zeros(shape, dtype=self._array.dtype)

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

    # def reproject_to_profile(self, profile, resampling=Resampling.lanczos):
    #     reproject_keys = ['crs', 'transform', 'nodata', 'height', 'width']
    #     if any([key not in profile for key in reproject_keys]):
    #         raise ValueError(f'profile should include {reproject_keys}')
    #     return self.reproject(crs=profile['crs'], transform=profile['transform'],
    #                         shape=(profile['height'], profile['width']), nodata=profile['nodata'],
    #                         resampling=resampling)
