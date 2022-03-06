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
import pathlib

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

from homonim import root_path
from homonim.errors import ImageProfileError, ImageFormatError
from homonim.raster_array import RasterArray


@pytest.fixture
def landsat_filename():
    return root_path.joinpath('data/test_example/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.tif')


@pytest.fixture
def modis_filename():
    return root_path.joinpath('data/test_example/reference/MODIS-006-MCD43A4-2015_09_15_B143.tif')


@pytest.fixture
def modis_ds(modis_filename):
    return rio.open(modis_filename, 'r')


@pytest.fixture
def byte_array():
    array = np.array(range(1, 101), dtype='uint8').reshape(20, 5)
    array[:, [0, -1]] = 255
    array[[0, -1], :] = 255
    return array


@pytest.fixture
def byte_profile(byte_array):
    profile = {
        'crs': CRS({'init': 'epsg:3857'}),
        'transform': Affine.identity() * Affine.translation(1e-10, 1e-10),
        'count': 1 if byte_array.ndim < 3 else byte_array.shape[0],
        'dtype': rio.uint8,
        'driver': 'GTiff',
        'width': byte_array.shape[-1],
        'height': byte_array.shape[-2],
        'nodata': 255
    }
    return profile


@pytest.fixture
def float_array():
    array = np.array(range(1, 201), dtype='float32').reshape(20, 10)
    array[:, [0, 1, -2, -1]] = float('nan')
    array[[0, 1, -2, -1], :] = float('nan')
    return array


@pytest.fixture
def float_profile(float_array):
    profile = {
        'crs': CRS({'init': 'epsg:3857'}),
        'transform': Affine.identity(),
        'count': 1 if float_array.ndim < 3 else float_array.shape[0],
        'dtype': rio.float32,
        'driver': 'GTiff',
        'width': float_array.shape[-1],
        'height': float_array.shape[-2],
        'nodata': float('nan')
    }
    return profile


@pytest.fixture
def byte_ra(byte_array, byte_profile):
    return RasterArray(byte_array, byte_profile['crs'], byte_profile['transform'],
                       nodata=byte_profile['nodata'])


@pytest.fixture
def rgb_byte_ra(byte_array, byte_profile):
    return RasterArray(np.stack((byte_array,) * 3, axis=0), byte_profile['crs'], byte_profile['transform'],
                       nodata=byte_profile['nodata'])


@pytest.fixture
def byte_file(tmp_path, byte_array, byte_profile):
    byte_filename = tmp_path.joinpath('uint8.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(byte_filename, 'w', **byte_profile) as ds:
            ds.write(byte_array, indexes=1)
    return byte_filename


@pytest.fixture
def float_file(tmp_path, float_array, float_profile):
    float_filename = tmp_path.joinpath('float32.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(float_filename, 'w', **float_profile) as ds:
            ds.write(float_array, indexes=1)
    return float_filename


@pytest.fixture
def rgba_file(tmp_path, byte_array, byte_profile):
    rgba_array = np.stack((byte_array,) * 4, axis=0)
    rgba_array[3] = (rgba_array[0] != byte_profile['nodata']) * 255
    rgba_filename = tmp_path.joinpath('rgba.tif')
    rgba_profile = byte_profile.copy()
    rgba_profile.update(count=4, nodata=None,
                        colorinterp=[ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(rgba_filename, 'w', **rgba_profile) as ds:
            ds.write(rgba_array, indexes=range(1, 5))
    return rgba_filename


@pytest.fixture
def masked_file(tmp_path, byte_array, byte_profile):
    masked_filename = tmp_path.joinpath('masked.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(masked_filename, 'w', **byte_profile) as ds:
            ds.write(byte_array, indexes=1)
            ds.write_mask(byte_array != byte_profile['nodata'])
    return masked_filename


@pytest.fixture
def float_ra(float_array, float_profile):
    return RasterArray(float_array, float_profile['crs'], float_profile['transform'],
                       nodata=float_profile['nodata'])

