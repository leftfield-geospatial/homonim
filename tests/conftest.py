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
import re
from collections import namedtuple
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import rasterio as rio
from click.testing import CliRunner
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, calculate_default_transform
from rasterio import windows

from homonim import root_path, utils
from homonim.enums import ProcCrs, Model
from homonim.fuse import RasterFuse
from homonim.raster_array import RasterArray

FuseCliParams = namedtuple(
    'FuseCliParams',
    ['src_file', 'ref_file', 'model', 'kernel_shape', 'proc_crs', 'corr_file', 'param_file', 'cli_str']
)  # yapf: disable
""" Named tuple to wrap fuse cli parameters and string. """


def str_contain_no_space(str1: str, str2: str) -> bool:
    """ Test if str2 contain str1, ignoring case and whitespace. """
    str1 = re.sub(r'\s+', '', str1.lower())
    str2 = re.sub(r'\s+', '', str2.lower())
    return str1 in str2


@pytest.fixture
def param_file() -> Path:
    """ Basic parameter image (16x16 tiles). """
    return root_path.joinpath('tests/data/parameter/float_100cm_rgb_FUSE_cREF_mGAIN-OFFSET_k5_5_PARAM.tif')


@pytest.fixture
def param_file_tile_10x20() -> Path:
    """ Basic parameter image (16x16 tiles). """
    return root_path.joinpath('tests/data/parameter/float_100cm_rgb_FUSE_cREF_mGAIN-OFFSET_k5_5_PARAM_tile_10x20.tif')


@pytest.fixture
def array_byte() -> np.ndarray:
    """ 2D byte gradient image with single pixel nodata=255 border. """
    array = np.array(range(1, 201), dtype='uint8').reshape(20, 10)
    array[:, [0, -1]] = 255
    array[[0, -1], :] = 255
    return array


@pytest.fixture
def array_100cm_float() -> np.ndarray:
    """ 2D float32 gradient image with single pixel nodata=nan border. """
    array = np.array(range(1, 201), dtype='float32').reshape(20, 10)
    array[:, [0, -1]] = float('nan')
    array[[0, -1], :] = float('nan')
    return array


@pytest.fixture
def array_50cm_float(array_100cm_float) -> np.ndarray:
    """ 2x upsampled array_100cm_float with double pixel nodata=nan border. """
    array = np.kron(array_100cm_float, np.ones((2, 2)))
    array[:, [0, 1, -2, -1]] = float('nan')
    array[[0, 1, -2, -1], :] = float('nan')
    return array


@pytest.fixture
def profile_byte(array_byte) -> Dict:
    """ rasterio profile dict for array_byte. """
    profile = {
        'crs': CRS({'init': 'epsg:3857'}),
        # North-up, with origin at (1, -1)
        'transform': Affine(1, 0, 0, 0, -1, 0) * Affine.translation(5, 5),
        'count': 1 if array_byte.ndim < 3 else array_byte.shape[0],
        'dtype': rio.uint8,
        'driver': 'GTiff',
        'width': array_byte.shape[-1],
        'height': array_byte.shape[-2],
        'nodata': 255
    }
    return profile


@pytest.fixture
def profile_100cm_float(array_100cm_float) -> Dict:
    """ rasterio profile dict for array_100cm_float. """
    profile = {
        'crs': CRS({'init': 'epsg:3857'}),
        # North-up, origin at (5, -5)
        'transform': Affine(1, 0, 0, 0, -1, 0) * Affine.translation(5, 5),
        'count': 1 if array_100cm_float.ndim < 3 else array_100cm_float.shape[0],
        'dtype': rio.float32,
        'driver': 'GTiff',
        'width': array_100cm_float.shape[-1],
        'height': array_100cm_float.shape[-2],
        'nodata': float('nan')
    }
    return profile


@pytest.fixture
def profile_50cm_float(array_50cm_float) -> Dict:
    """ rasterio profile dict for array_50cm_float. """
    profile = {
        'crs': CRS({'init': 'epsg:3857'}),
        # North-up, origin at (5, -5)
        'transform': Affine(1, 0, 0, 0, -1, 0) * Affine.translation(5, 5) * Affine.scale(0.5),
        'count': 1 if array_50cm_float.ndim < 3 else array_50cm_float.shape[0],
        'dtype': rio.float32,
        'driver': 'GTiff',
        'width': array_50cm_float.shape[-1],
        'height': array_50cm_float.shape[-2],
        'nodata': float('nan')
    }
    return profile


@pytest.fixture
def ra_byte(array_byte, profile_byte) -> RasterArray:
    """ Raster array with single band of byte. """
    return RasterArray(array_byte, profile_byte['crs'], profile_byte['transform'], nodata=profile_byte['nodata'])


@pytest.fixture
def ra_rgb_byte(array_byte, profile_byte) -> RasterArray:
    """ Raster array with three bands of byte. """
    return RasterArray(
        np.stack((array_byte,) * 3, axis=0), profile_byte['crs'], profile_byte['transform'],
        nodata=profile_byte['nodata']
    )


@pytest.fixture
def ra_100cm_float(array_100cm_float, profile_100cm_float) -> RasterArray:
    """ Raster array with single band of float32 at 100cm pixel resolution. """
    return RasterArray(
        array_100cm_float, profile_100cm_float['crs'], profile_100cm_float['transform'],
        nodata=profile_100cm_float['nodata']
    )


@pytest.fixture
def ra_50cm_float(array_50cm_float, profile_50cm_float) -> RasterArray:
    """ Raster array with single band of float32 at 50cm pixel resolution. 2x upsampled version of ra_100cm_float. """
    return RasterArray(
        array_50cm_float, profile_50cm_float['crs'], profile_50cm_float['transform'],
        nodata=profile_50cm_float['nodata']
    )


@pytest.fixture
def profile_45cm_float(array_100cm_float, profile_100cm_float) -> Dict:
    """
    rasterio profile dict for array_45cm_float, shifted by half 45cm pixel from ra_100cm_float, and padded with
    one pixel.
    """
    scale = 0.45  # resolution scaling
    shape = tuple(np.round(np.array(array_100cm_float.shape) / scale + 1).astype('int'))
    # scale and shift the profile_100cm_float['transform']
    transform = profile_100cm_float['transform'] * Affine.scale(scale) * Affine.translation(-.5, -.5)
    profile = profile_100cm_float.copy()
    profile.update(width=shape[1], height=shape[0], transform=transform)
    return profile


@pytest.fixture
def array_45cm_float(array_100cm_float, profile_100cm_float, profile_45cm_float) -> np.ndarray:
    """ 1/.45 upsampled array_100cm_float. """
    float_45cm_array = np.full(
        (profile_45cm_float['height'], profile_45cm_float['width']), profile_45cm_float['nodata']
    )
    _ = reproject(
        array_100cm_float,
        destination=float_45cm_array,
        src_crs=profile_100cm_float['crs'],
        dst_crs=profile_45cm_float['crs'],
        src_transform=profile_100cm_float['transform'],
        dst_transform=profile_45cm_float['transform'],
        src_nodata=profile_100cm_float['nodata'],
        resampling=Resampling.bilinear,
    )  # yapf: disable

    return float_45cm_array


@pytest.fixture
def ra_45cm_float(array_45cm_float, profile_45cm_float) -> RasterArray:
    """Raster array with single band of float32 at 45cm pixel resolution. upsampled version of ra_100cm_float, but
    on a different pixel grid"""
    return RasterArray(
        array_45cm_float, profile_45cm_float['crs'], profile_45cm_float['transform'],
        nodata=profile_45cm_float['nodata']
    )


@pytest.fixture
def file_byte(tmp_path: Path, array_byte, profile_byte) -> Path:
    """ Single band byte geotiff. """
    filename = tmp_path.joinpath('uint8.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile_byte) as ds:
            ds.write(array_byte, indexes=1)
    return filename


@pytest.fixture
def file_rgba(tmp_path: Path, array_byte, profile_byte) -> Path:
    """ RGB + alpha band byte geotiff. """
    array = np.stack((array_byte,) * 4, axis=0)
    array[3] = (array[0] != profile_byte['nodata']) * 255
    filename = tmp_path.joinpath('rgba.tif')
    profile = profile_byte.copy()
    profile.update(
        count=4, nodata=None, colorinterp=[ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
    )
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array, indexes=range(1, 5))
    return filename


@pytest.fixture
def file_bgr(tmp_path: Path, array_byte, profile_byte) -> Path:
    """ BGR byte geotiff. """
    array = np.stack((array_byte,) * 3, axis=0)
    filename = tmp_path.joinpath('bgr.tif')
    profile = profile_byte.copy()
    profile.update(count=3, nodata=None,)
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.colorinterp = [ColorInterp.blue, ColorInterp.green, ColorInterp.red]
            ds.write(array, indexes=range(1, 4))
    return filename


@pytest.fixture
def file_masked(tmp_path: Path, array_byte, profile_byte) -> Path:
    """ Single band byte geotiff with internal mask (i.e. w/o nodata). """
    filename = tmp_path.joinpath('masked.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile_byte) as ds:
            ds.write(array_byte, indexes=1)
            ds.write_mask(array_byte != profile_byte['nodata'])
    return filename


@pytest.fixture
def src_file_100cm_float(tmp_path: Path, array_100cm_float, profile_100cm_float) -> Path:
    """ Single band float32 geotiff with 100cm pixel resolution. """
    filename = tmp_path.joinpath('float_100cm_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile_100cm_float) as ds:
            ds.write(array_100cm_float, indexes=1)
    return filename


@pytest.fixture
def ref_file_100cm_float(tmp_path: Path, array_100cm_float, profile_100cm_float) -> Path:
    """
    Single band float32 geotiff with 100cm pixel resolution, the same as src_file_100cm_float, but padded with an
    extra pixel.
    """
    shape = (np.array(array_100cm_float.shape) + 2).astype('int')
    transform = profile_100cm_float['transform'] * Affine.translation(-1, -1)
    profile = profile_100cm_float.copy()
    profile.update(transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_100cm_ref.tif')
    window = windows.Window(1, 1, array_100cm_float.shape[1], array_100cm_float.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array_100cm_float, indexes=1, window=window)
    return filename


@pytest.fixture
def src_file_50cm_float(tmp_path: Path, array_50cm_float, profile_50cm_float) -> Path:
    """ Single band float32 geotiff with 50cm pixel resolution. """
    filename = tmp_path.joinpath('float_50cm_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile_50cm_float) as ds:
            ds.write(array_50cm_float, indexes=1)
    return filename


@pytest.fixture
def ref_file_50cm_float(tmp_path: Path, array_50cm_float, profile_50cm_float) -> Path:
    """Single band float32 geotiff with 50cm pixel resolution, the same as src_file_50cm_float, but padded with an
    extra pixel"""
    shape = (np.array(array_50cm_float.shape) + 2).astype('int')
    transform = profile_50cm_float['transform'] * Affine.translation(-1, -1)
    profile = profile_50cm_float.copy()
    profile.update(transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_50cm_ref.tif')
    window = windows.Window(1, 1, array_50cm_float.shape[1], array_50cm_float.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array_50cm_float, indexes=1, window=window)
    return filename


@pytest.fixture
def src_file_45cm_float(tmp_path: Path, array_45cm_float, profile_45cm_float) -> Path:
    """ Single band float32 geotiff with 45cm pixel resolution. """
    filename = tmp_path.joinpath('float_45cm_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile_45cm_float) as ds:
            ds.write(array_45cm_float, indexes=1)
    return filename


@pytest.fixture
def ref_file_45cm_float(tmp_path: Path, array_45cm_float, profile_45cm_float) -> Path:
    """Single band float32 geotiff with 45cm pixel resolution, the same as src_file_45cm_float, but padded with an
    extra pixel"""
    shape = (np.array(array_45cm_float.shape) + 2).astype('int')
    transform = profile_45cm_float['transform'] * Affine.translation(-1, -1)
    profile = profile_45cm_float.copy()
    profile.update(transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_45cm_ref.tif')
    window = windows.Window(1, 1, array_45cm_float.shape[1], array_45cm_float.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array_45cm_float, indexes=1, window=window)
    return filename


@pytest.fixture
def file_rgb_100cm_float(tmp_path: Path, array_100cm_float, profile_100cm_float) -> Path:
    """ 3 band float32 geotiff with 100cm pixel resolution. """
    array = np.stack([i * array_100cm_float for i in range(1, 4)], axis=0)
    profile = profile_100cm_float.copy()
    profile.update(count=3)
    filename = tmp_path.joinpath('float_100cm_rgb.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array, indexes=[1, 2, 3])
    return filename


@pytest.fixture
def file_rgb_50cm_float(tmp_path: Path, array_50cm_float, profile_50cm_float) -> Path:
    """ 3 band float32 geotiff with 50cm pixel resolution, same extent as file_rgb_100cm_float. """
    array = np.stack([i * array_50cm_float for i in range(1, 4)], axis=0)
    profile = profile_50cm_float.copy()
    profile.update(count=3)
    filename = tmp_path.joinpath('float_50cm_rgb.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array, indexes=[1, 2, 3])
    return filename


@pytest.fixture
def src_file_sup_100cm_float(tmp_path: Path, array_100cm_float, profile_100cm_float) -> Path:
    """ Single band float32 geotiff with 100cm pixel resolution.  South-up orientation. """
    transform = (
        profile_100cm_float['transform'] * Affine.scale(1, -1) * Affine.translation(0, -array_100cm_float.shape[0])
    )
    profile = profile_100cm_float.copy()
    profile.update(transform=transform)
    filename = tmp_path.joinpath('float_100cm_sup_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(np.flipud(array_100cm_float), indexes=1)
    return filename


@pytest.fixture
def src_file_rot_100cm_float(tmp_path: Path, array_100cm_float, profile_100cm_float) -> Path:
    """ Single band float32 geotiff with 100cm pixel resolution.  West-up orientation. """
    # Rotate the north-up (-ve scale y axis) transform counter-clock-wise by 90 degrees.  Now both x and y axes are
    # positive scale.  Then shift the origin (at BL of image) to (5, -15), so that the bounds are the same as for
    # src_file_100cm_float (i.e. 5,-25,15,-5).
    transform = Affine(1, 0, 0, 0, -1, 0) * Affine.rotation(90) * Affine.translation(5, -5 - array_100cm_float.shape[1])
    profile = profile_100cm_float.copy()
    profile.update(transform=transform, width=array_100cm_float.shape[0], height=array_100cm_float.shape[1])
    filename = tmp_path.joinpath('float_100cm_rot_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(np.rot90(array_100cm_float), indexes=1)
    return filename


@pytest.fixture
def src_file_wgs84_100cm_float(tmp_path: Path, array_100cm_float, profile_100cm_float) -> Path:
    """ Single band float32 geotiff with 100cm pixel resolution. WGS84 `projection`.  """
    to_crs = CRS.from_epsg('4326')
    bounds = windows.bounds(windows.Window(0, 0, *array_100cm_float.shape[::-1]), profile_100cm_float['transform'])
    transform, _, _ = calculate_default_transform(
        profile_100cm_float['crs'], to_crs, *array_100cm_float.shape[::-1], *bounds
    )
    profile = profile_100cm_float.copy()
    profile.update(crs=to_crs, transform=transform)
    filename = tmp_path.joinpath('float_100cm_wgs84_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(array_100cm_float, indexes=1)
    return filename


@pytest.fixture
def src_file_wgs84_sup_100cm_float(tmp_path: Path, array_100cm_float, profile_100cm_float) -> Path:
    """ Single band float32 geotiff with 100cm pixel resolution. WGS84 `projection` and South-up orientation.  """
    to_crs = CRS.from_epsg('4326')
    bounds = windows.bounds(windows.Window(0, 0, *array_100cm_float.shape[::-1]), profile_100cm_float['transform'])
    transform, _, _ = calculate_default_transform(
        profile_100cm_float['crs'], to_crs, *array_100cm_float.shape[::-1], *bounds
    )
    transform *= Affine.scale(1, -1) * Affine.translation(0, -array_100cm_float.shape[0])    # south up
    profile = profile_100cm_float.copy()
    profile.update(crs=to_crs, transform=transform)
    filename = tmp_path.joinpath('float_100cm_wgs84_sup_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(np.flipud(array_100cm_float), indexes=1)
    return filename


@pytest.fixture
def ref_file_sup_100cm_float(tmp_path: Path, array_100cm_float, profile_100cm_float) -> Path:
    """
    Single band float32 geotiff with 100cm pixel resolution, the same as src_file_100cm_float, but padded with an
    extra pixel, and South-up orientation.
    """
    shape = (np.array(array_100cm_float.shape) + 2).astype('int')
    transform = profile_100cm_float['transform'] * Affine.translation(-1, -1)   # padding
    transform *= Affine.scale(1, -1) * Affine.translation(0, -shape[0])   # South-up
    profile = profile_100cm_float.copy()
    profile.update(transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_100cm_sup_ref.tif')
    window = windows.Window(1, 1, array_100cm_float.shape[1], array_100cm_float.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(np.flipud(array_100cm_float), indexes=1, window=window)
    return filename


@pytest.fixture
def ref_file_rot_100cm_float(tmp_path: Path, array_100cm_float, profile_100cm_float) -> Path:
    """
    Single band float32 geotiff with 100cm pixel resolution, the same as src_file_100cm_float, but padded with an
    extra pixel, and West-up orientation.
    """
    # Rotate the north-up (-ve scale y axis) transform counter-clock-wise by 90 degrees.  Now both x and y axes are
    # positive scale.  Then shift the origin (at BL of image) to (5, -15), so that the bounds are the same as for
    # src_file_100cm_float (i.e. 5,-25,15,-5).
    transform = Affine(1, 0, 0, 0, -1, 0) * Affine.rotation(90) * Affine.translation(5, -5 - array_100cm_float.shape[1])
    transform *= Affine.translation(-1, -1)  # padding
    profile = profile_100cm_float.copy()
    profile.update(transform=transform, width=array_100cm_float.shape[0] + 2, height=array_100cm_float.shape[1] + 2)
    window = windows.Window(1, 1, array_100cm_float.shape[0], array_100cm_float.shape[1])
    filename = tmp_path.joinpath('float_100cm_rot_ref.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(np.rot90(array_100cm_float), indexes=1, window=window)
    return filename


@pytest.fixture
def ref_file_wgs84_100cm_float(tmp_path: Path, array_100cm_float, profile_100cm_float) -> Path:
    """
    Single band float32 geotiff with 100cm pixel resolution, the same as src_file_100cm_float, but padded with an
    extra pixel, and in WGS84.
    """
    shape = (np.array(array_100cm_float.shape) + 2).astype('int')
    to_crs = CRS.from_epsg('4326')
    bounds = windows.bounds(windows.Window(-1, -1, *shape[::-1]), profile_100cm_float['transform'])
    transform, _, _ = calculate_default_transform(
        profile_100cm_float['crs'], to_crs, *shape, *bounds
    )
    profile = profile_100cm_float.copy()
    profile.update(crs=to_crs, transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_100cm_wgs84_ref.tif')
    window = windows.Window(1, 1, array_100cm_float.shape[1], array_100cm_float.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(array_100cm_float, indexes=1, window=window)
    return filename


@pytest.fixture
def ref_file_wgs84_sup_float(tmp_path: Path, array_100cm_float, profile_100cm_float) -> Path:
    """
    Single band float32 geotiff with 100cm pixel resolution, the same as src_file_100cm_float, but padded with an
    extra pixel, in WGS84, and oriented South-up.
    """
    shape = (np.array(array_100cm_float.shape) + 2).astype('int')
    to_crs = CRS.from_epsg('4326')
    bounds = windows.bounds(windows.Window(-1, -1, *shape[::-1]), profile_100cm_float['transform'])
    transform, _, _ = calculate_default_transform(
        profile_100cm_float['crs'], to_crs, *shape, *bounds
    )
    transform *= Affine.scale(1, -1) * Affine.translation(0, -shape[0])    # south up
    profile = profile_100cm_float.copy()
    profile.update(crs=to_crs, transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_100cm_wgs84_sup_ref.tif')
    window = windows.Window(1, 1, array_100cm_float.shape[1], array_100cm_float.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(np.flipud(array_100cm_float), indexes=1, window=window)
    return filename


@pytest.fixture()
def modis_ref_file() -> Path:
    return root_path.joinpath(r'tests/data/reference/modis_nbar.tif')


@pytest.fixture()
def landsat_ref_file() -> Path:
    return root_path.joinpath(r'tests/data/reference/landsat8_byte.tif')


@pytest.fixture()
def s2_ref_file() -> Path:
    return root_path.joinpath(
        r'tests/data/reference/sentinel2_b432_byte.tif'
    )


@pytest.fixture()
def landsat_src_file() -> Path:
    return root_path.joinpath(r'tests/data/reference/landsat8_byte.vrt')


@pytest.fixture()
def ngi_src_files() -> Tuple[Path, ...]:
    source_root = root_path.joinpath('tests/data/source/')
    return tuple([fn for fn in source_root.glob('ngi_rgb_byte_*.tif')])


@pytest.fixture()
def ngi_src_file() -> Path:
    return root_path.joinpath(r'tests/data/source/ngi_rgb_byte_1.tif')


@pytest.fixture
def runner() -> CliRunner:
    """ click runner for command line execution. """
    return CliRunner()


@pytest.fixture
def default_fuse_cli_params(tmp_path: Path, ref_file_100cm_float, src_file_50cm_float) -> FuseCliParams:
    """ FuseCliParams with default parameter values. """
    ref_file = ref_file_100cm_float
    src_file = src_file_50cm_float
    model = Model.gain_blk_offset
    kernel_shape = (5, 5)
    proc_crs = ProcCrs.ref
    post_fix = utils.create_out_postfix(proc_crs, model, kernel_shape, RasterFuse.create_out_profile()['driver'])
    corr_file = tmp_path.joinpath(src_file.stem + post_fix)
    param_file = utils.create_param_filename(corr_file)

    cli_str = (f'fuse -od {tmp_path} {src_file} {ref_file}')
    return FuseCliParams(src_file, ref_file, model, kernel_shape, proc_crs, corr_file, param_file, cli_str)


@pytest.fixture
def basic_fuse_cli_params(tmp_path: Path, ref_file_100cm_float, src_file_100cm_float) -> FuseCliParams:
    """ FuseCliParams with basic parameter values. """
    ref_file = ref_file_100cm_float
    src_file = src_file_100cm_float
    model = Model.gain_blk_offset
    kernel_shape = (3, 3)
    proc_crs = ProcCrs.ref
    post_fix = utils.create_out_postfix(proc_crs, model, kernel_shape, RasterFuse.create_out_profile()['driver'])
    corr_file = tmp_path.joinpath(src_file.stem + post_fix)
    param_file = utils.create_param_filename(corr_file)

    cli_str = (
        f'fuse -m {model.value} -k {kernel_shape[0]} {kernel_shape[1]} -od {tmp_path} -pc {proc_crs.value} '
        f'{src_file} {ref_file}'
    )
    return FuseCliParams(src_file, ref_file, model, kernel_shape, proc_crs, corr_file, param_file, cli_str)


@pytest.fixture
def default_fuse_rgb_cli_params(tmp_path: Path, file_rgb_100cm_float, file_rgb_50cm_float) -> FuseCliParams:
    """ FuseCliParams with default parameter values. """
    ref_file = file_rgb_100cm_float
    src_file = file_rgb_50cm_float
    model = Model.gain_blk_offset
    kernel_shape = (5, 5)
    proc_crs = ProcCrs.ref
    post_fix = utils.create_out_postfix(proc_crs, model, kernel_shape, RasterFuse.create_out_profile()['driver'])
    corr_file = tmp_path.joinpath(src_file.stem + post_fix)
    param_file = utils.create_param_filename(corr_file)

    cli_str = (f'fuse -od {tmp_path} {src_file} {ref_file}')
    return FuseCliParams(src_file, ref_file, model, kernel_shape, proc_crs, corr_file, param_file, cli_str)
