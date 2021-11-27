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
    def __init__(self, array, crs, transform, nodata=hom_nodata):
        # array = np.array(array)
        if (array.ndim < 2) or (array.ndim > 3):
            raise ValueError('Array must a 2 or 3D numpy array')
        self._array = array

        if isinstance(crs, CRS):
            self._crs = crs
        else:
            raise TypeError('crs must be an instance of rasterio.CRS')

        if isinstance(transform, Affine):
            self._transform = transform
        else:
            raise TypeError('transform must be an instance of rasterio.Affine')

        self._nodata = nodata

    @classmethod
    def from_profile(cls, array, window=None, **profile):
        if not ('crs' and 'transform' and 'nodata' in profile):
            raise Exception('**kwargs should include crs, transform and nodata')

        if window is not None:
            profile['transform'] = windows.transform(window, profile['transform'])

        return cls(array, profile['crs'], profile['transform'], nodata=profile['nodata'])

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
        return self._array.shape[-1]

    @property
    def height(self):
        return self._array.shape[-2]

    @property
    def shape(self):
        return self._array.shape

    @property
    def count(self):
        if self._array.ndim == 3:
            return self._array.shape[0]
        else:
            return 1

    @property
    def transform(self):
        return self._transform

    @property
    def nodata(self):
        return self._nodata

    @property
    def res(self):
        return np.abs((self._transform.a, self._transform.e))

    @property
    def bounds(self):
        return windows.bounds(windows.Window(0, 0, self.width, self.height), self._transform)

    @property
    def profile(self):
        return dict(crs=self.crs, transform=self.transform, count=self.count, width=self.width, height=self.height,
                    bounds=self.bounds, nodata=self.nodata)

    @property
    def mask(self):
        return RasterArray((self._array != self._nodata).astype('uint8'), crs=self.crs, transform=self.transform,
                           nodata=None)

    def reproject(self, dst_crs=None, dst_transform=None, dst_shape=None, dst_nodata=hom_nodata, resampling=Resampling.lanczos):

        if isinstance(resampling, str):
            resampling = Resampling[resampling]
        # TODO: other argument combo checking
        if dst_transform and not dst_shape:
            raise Exception('if dst_transform is specified, dst_shape must also be specified')

        dst_crs = dst_crs or self._crs
        dst_transform = dst_transform or self._transform
        dst_shape = dst_shape or self._array.shape

        if self.array.ndim > 2:
            _dst_array = np.zeros((self._array.shape[0], *dst_shape), dtype=self._array.dtype)
        else:
            _dst_array = np.zeros(dst_shape, dtype=self._array.dtype)

        _, _ = reproject(
            self._array,
            destination=_dst_array,
            src_crs=self._crs,
            src_transform=self._transform,
            src_nodata=self._nodata,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            dst_nodata=dst_nodata,
            num_threads=multiprocessing.cpu_count(),
            resampling=resampling,
        )
        return RasterArray(_dst_array, crs=dst_crs, transform=dst_transform, nodata=dst_nodata)

    # def reproject_like_rarray(self, dst_array, resampling=Resampling.lanczos):
    #     self.reproject(dst_crs=dst_array.crs, dst_transform=dst_array.transform, dst_shape=dst_array.shape,
    #                    dst_nodata=dst_array.nodata, resampling=resampling)

    def reproject_like(self, dst_profile, resampling=Resampling.lanczos):
        return self.reproject(dst_crs=dst_profile['crs'], dst_transform=dst_profile['transform'],
                       dst_shape=(dst_profile['height'], dst_profile['width']), dst_nodata=dst_profile['nodata'],
                       resampling=resampling)
