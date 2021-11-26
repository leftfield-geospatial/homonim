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

class RasterArray(windows.WindowMethodsMixin, transform.TransformMethodsMixin):
    """A class for wrapping a geo-referenced numpy array"""
    def __init__(self, array, crs, transform, nodata=hom_nodata):
        array = np.array(array)
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
    def from_window(cls, array, window, **kwargs):
        if not ('crs' and 'transform' and 'nodata' in kwargs):
            raise Exception('**kwargs should include crs, transform and nodata')
        return cls(array, kwargs['crs'], windows.transform(window, kwargs['transform']), nodata=kwargs['nodata'])

    @classmethod
    def from_profile(cls, array, **kwargs):
        if not ('crs' and 'transform' and 'nodata' in kwargs):
            raise Exception('**kwargs should include crs, transform and nodata')
        return cls(array, kwargs['crs'], kwargs['transform'], nodata=kwargs['nodata'])

    @property
    def array(self):
        return self._array

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

    def reproject(self, dst_crs=None, dst_transform=None, dst_shape=None, dst_nodata=hom_nodata, resampling=Resampling.lanczos):

        # TODO: other argument combo checking
        if dst_transform and not dst_shape:
            raise Exception('if dst_transform is specified, dst_shape must also be specified')

        dst_crs = dst_crs or self._crs
        dst_transform = dst_transform or self._transform
        dst_shape = dst_shape or self._array.shape

        if self.array.ndim > 2:
            dst_array = np.zeros((self._array.shape[0], *dst_shape), dtype=self._array.dtype)
        else:
            dst_array = np.zeros(dst_shape, dtype=self._array.dtype)

        _, _ = reproject(
            self._array,
            destination=dst_array,
            src_crs=self._crs,
            src_transform=self._transform,
            src_nodata=self._nodata,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            dst_nodata=dst_nodata,
            num_threads=multiprocessing.cpu_count(),
            resampling=resampling,
        )
        return dst_array
