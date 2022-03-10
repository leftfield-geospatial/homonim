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
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window

from homonim import root_path
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
def float_100cm_array():
    array = np.array(range(1, 201), dtype='float32').reshape(20, 10)
    array[:, [0, -1]] = float('nan')
    array[[0, -1], :] = float('nan')
    return array


@pytest.fixture
def float_50cm_array(float_100cm_array):
    array = np.kron(float_100cm_array, np.ones((2, 2)))
    array[:, [0, 1, -2, -1]] = float('nan')
    array[[0, 1, -2, -1], :] = float('nan')
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
def float_100cm_profile(float_100cm_array):
    profile = {
        'crs': CRS({'init': 'epsg:3857'}),
        'transform': Affine.identity(),
        'count': 1 if float_100cm_array.ndim < 3 else float_100cm_array.shape[0],
        'dtype': rio.float32,
        'driver': 'GTiff',
        'width': float_100cm_array.shape[-1],
        'height': float_100cm_array.shape[-2],
        'nodata': float('nan')
    }
    return profile


@pytest.fixture
def float_50cm_profile(float_50cm_array):
    profile = {
        'crs': CRS({'init': 'epsg:3857'}),
        'transform': Affine.identity() * Affine.scale(0.5),
        'count': 1 if float_50cm_array.ndim < 3 else float_50cm_array.shape[0],
        'dtype': rio.float32,
        'driver': 'GTiff',
        'width': float_50cm_array.shape[-1],
        'height': float_50cm_array.shape[-2],
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
def float_100cm_ra(float_100cm_array, float_100cm_profile):
    return RasterArray(float_100cm_array, float_100cm_profile['crs'], float_100cm_profile['transform'],
                       nodata=float_100cm_profile['nodata'])


@pytest.fixture
def float_50cm_ra(float_50cm_array, float_50cm_profile):
    """
    A high resolution version of float_100cm_ra.
    Aligned with the float_100cm_ra pixel grid, so that re-projection back to float_100cm_ra space will give the float_100cm_ra
    mask, and ~data (resampling method dependent).
    """
    return RasterArray(float_50cm_array, float_50cm_profile['crs'], float_50cm_profile['transform'],
                       nodata=float_50cm_profile['nodata'])


@pytest.fixture
def float_45cm_profile(float_100cm_array, float_100cm_profile):
    scale = 0.45  # resolution scaling
    # pad scaled image with a border of ~1 float_100cm_ra pixel
    # shape = tuple(np.ceil(np.array(float_100cm_ra.shape) / scale + (2 / scale)).astype('int'))
    # transform = float_100cm_ra.transform * Affine.translation(-1, -1) * Affine.scale(scale)
    shape = tuple(np.round(np.array(float_100cm_array.shape) / scale + 1).astype('int'))
    transform = float_100cm_profile['transform'] * Affine.scale(scale) * Affine.translation(-.5, -.5)
    profile = float_100cm_profile.copy()
    profile.update(width=shape[1], height=shape[0], transform=transform)
    return profile


@pytest.fixture
def float_45cm_array(float_100cm_array, float_100cm_profile, float_45cm_profile):
    """
    A high resolution version of float_100cm_ra, but on a different pixel grid.
    """
    float_45cm_array = np.full((float_45cm_profile['height'], float_45cm_profile['width']),
                               float_45cm_profile['nodata'])
    _ = reproject(
        float_100cm_array,
        destination=float_45cm_array,
        src_crs=float_100cm_profile['crs'],
        dst_crs=float_45cm_profile['crs'],
        src_transform=float_100cm_profile['transform'],
        dst_transform=float_45cm_profile['transform'],
        src_nodata=float_100cm_profile['nodata'],
        resampling=Resampling.bilinear,
    )

    return float_45cm_array


@pytest.fixture
def float_45cm_ra(float_45cm_array, float_45cm_profile):
    return RasterArray(float_45cm_array, float_45cm_profile['crs'], float_45cm_profile['transform'],
                       nodata=float_45cm_profile['nodata'])


@pytest.fixture
def byte_file(tmp_path, byte_array, byte_profile):
    filename = tmp_path.joinpath('uint8.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **byte_profile) as ds:
            ds.write(byte_array, indexes=1)
    return filename


@pytest.fixture
def rgba_file(tmp_path, byte_array, byte_profile):
    array = np.stack((byte_array,) * 4, axis=0)
    array[3] = (array[0] != byte_profile['nodata']) * 255
    filename = tmp_path.joinpath('rgba.tif')
    profile = byte_profile.copy()
    profile.update(count=4, nodata=None,
                   colorinterp=[ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array, indexes=range(1, 5))
    return filename


@pytest.fixture
def masked_file(tmp_path, byte_array, byte_profile):
    filename = tmp_path.joinpath('masked.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **byte_profile) as ds:
            ds.write(byte_array, indexes=1)
            ds.write_mask(byte_array != byte_profile['nodata'])
    return filename


@pytest.fixture
def float_100cm_src_file(tmp_path, float_100cm_array, float_100cm_profile):
    filename = tmp_path.joinpath('float_100cm_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **float_100cm_profile) as ds:
            ds.write(float_100cm_array, indexes=1)
    return filename


@pytest.fixture
def float_100cm_ref_file(tmp_path, float_100cm_array, float_100cm_profile):
    shape = (np.array(float_100cm_array.shape) + 2).astype('int')
    transform = float_100cm_profile['transform'] * Affine.translation(-1, -1)
    profile = float_100cm_profile.copy()
    profile.update(transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_100cm_ref.tif')
    window = Window(1, 1, float_100cm_array.shape[1], float_100cm_array.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(float_100cm_array, indexes=1, window=window)
    return filename


@pytest.fixture
def float_50cm_src_file(tmp_path, float_50cm_array, float_50cm_profile):
    filename = tmp_path.joinpath('float_50cm_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **float_50cm_profile) as ds:
            ds.write(float_50cm_array, indexes=1)
    return filename


@pytest.fixture
def float_50cm_ref_file(tmp_path, float_50cm_array, float_50cm_profile):
    shape = (np.array(float_50cm_array.shape) + 2).astype('int')
    transform = float_50cm_profile['transform'] * Affine.translation(-1, -1)
    profile = float_50cm_profile.copy()
    profile.update(transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_50cm_ref.tif')
    window = Window(1, 1, float_50cm_array.shape[1], float_50cm_array.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(float_50cm_array, indexes=1, window=window)
    return filename


@pytest.fixture
def float_45cm_src_file(tmp_path, float_45cm_array, float_45cm_profile):
    filename = tmp_path.joinpath('float_45cm_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **float_45cm_profile) as ds:
            ds.write(float_45cm_array, indexes=1)
    return filename


@pytest.fixture
def float_45cm_ref_file(tmp_path, float_45cm_array, float_45cm_profile):
    shape = (np.array(float_45cm_array.shape) + 2).astype('int')
    transform = float_45cm_profile['transform'] * Affine.translation(-1, -1)
    profile = float_45cm_profile.copy()
    profile.update(transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_45cm_ref.tif')
    window = Window(1, 1, float_45cm_array.shape[1], float_45cm_array.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(float_45cm_array, indexes=1, window=window)
    return filename

@pytest.fixture
def float_100cm_rgb_file(tmp_path, float_100cm_array, float_100cm_profile):
    array = np.stack((float_100cm_array, ) * 3, axis=0)
    profile = float_100cm_profile.copy()
    profile.update(count=3)
    filename = tmp_path.joinpath('float_100cm_rgb.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array, indexes=[1, 2, 3])
    return filename
