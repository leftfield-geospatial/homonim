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
from typing import Dict

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
def byte_array() -> np.ndarray:
    """ 2D byte gradient image with single pixel nodata=255 border. """
    array = np.array(range(1, 101), dtype='uint8').reshape(20, 5)
    array[:, [0, -1]] = 255
    array[[0, -1], :] = 255
    return array


@pytest.fixture
def float_100cm_array() -> np.ndarray:
    """ 2D float32 gradient image with single pixel nodata=nan border. """
    array = np.array(range(1, 201), dtype='float32').reshape(20, 10)
    array[:, [0, -1]] = float('nan')
    array[[0, -1], :] = float('nan')
    return array


@pytest.fixture
def float_50cm_array(float_100cm_array: np.ndarray) -> np.ndarray:
    """ 2x upsampled float_100cm_array with double pixel nodata=nan border. """
    array = np.kron(float_100cm_array, np.ones((2, 2)))
    array[:, [0, 1, -2, -1]] = float('nan')
    array[[0, 1, -2, -1], :] = float('nan')
    return array


@pytest.fixture
def byte_profile(byte_array: np.ndarray) -> Dict:
    """ rasterio profile dict for byte_array. """
    profile = {
        'crs': CRS({'init': 'epsg:3857'}),
        # North-up, in S hemisphere
        'transform': Affine(1, 0, 0, 0, -1, 0) * Affine.translation(1, 1),
        'count': 1 if byte_array.ndim < 3 else byte_array.shape[0],
        'dtype': rio.uint8,
        'driver': 'GTiff',
        'width': byte_array.shape[-1],
        'height': byte_array.shape[-2],
        'nodata': 255
    }
    return profile


@pytest.fixture
def float_100cm_profile(float_100cm_array: np.ndarray) -> Dict:
    """ rasterio profile dict for float_100cm_array. """
    profile = {
        'crs': CRS({'init': 'epsg:3857'}),
        # North-up, in S hemisphere
        'transform': Affine(1, 0, 0, 0, -1, 0) * Affine.translation(5, 5),
        'count': 1 if float_100cm_array.ndim < 3 else float_100cm_array.shape[0],
        'dtype': rio.float32,
        'driver': 'GTiff',
        'width': float_100cm_array.shape[-1],
        'height': float_100cm_array.shape[-2],
        'nodata': float('nan')
    }
    return profile


@pytest.fixture
def float_50cm_profile(float_50cm_array: np.ndarray) -> Dict:
    """ rasterio profile dict for float_50cm_array. """
    profile = {
        'crs': CRS({'init': 'epsg:3857'}),
        # North-up, and in S hemisphere
        'transform': Affine(1, 0, 0, 0, -1, 0) * Affine.translation(5, 5) * Affine.scale(0.5),
        'count': 1 if float_50cm_array.ndim < 3 else float_50cm_array.shape[0],
        'dtype': rio.float32,
        'driver': 'GTiff',
        'width': float_50cm_array.shape[-1],
        'height': float_50cm_array.shape[-2],
        'nodata': float('nan')
    }
    return profile


@pytest.fixture
def byte_ra(byte_array: np.ndarray, byte_profile: Dict) -> RasterArray:
    """ Raster array with single band of byte. """
    return RasterArray(byte_array, byte_profile['crs'], byte_profile['transform'], nodata=byte_profile['nodata'])


@pytest.fixture
def rgb_byte_ra(byte_array: np.ndarray, byte_profile: Dict) -> RasterArray:
    """ Raster array with three bands of byte. """
    return RasterArray(
        np.stack((byte_array, ) * 3, axis=0), byte_profile['crs'], byte_profile['transform'],
        nodata=byte_profile['nodata']
    )


@pytest.fixture
def float_100cm_ra(float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> RasterArray:
    """ Raster array with single band of float32 at 100cm pixel resolution. """
    return RasterArray(
        float_100cm_array, float_100cm_profile['crs'], float_100cm_profile['transform'],
        nodata=float_100cm_profile['nodata']
    )


@pytest.fixture
def float_50cm_ra(float_50cm_array: np.ndarray, float_50cm_profile: Dict) -> RasterArray:
    """ Raster array with single band of float32 at 50cm pixel resolution. 2x upsampled version of float_100cm_ra. """
    return RasterArray(
        float_50cm_array, float_50cm_profile['crs'], float_50cm_profile['transform'],
        nodata=float_50cm_profile['nodata']
    )


@pytest.fixture
def float_45cm_profile(float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> Dict:
    """
    rasterio profile dict for float_45cm_array, shifted by half 45cm pixel from float_100cm_ra, and padded with
    one pixel.
    """
    scale = 0.45  # resolution scaling
    shape = tuple(np.round(np.array(float_100cm_array.shape) / scale + 1).astype('int'))
    # scale and shift the float_100cm_profile['transform']
    transform = float_100cm_profile['transform'] * Affine.scale(scale) * Affine.translation(-.5, -.5)
    profile = float_100cm_profile.copy()
    profile.update(width=shape[1], height=shape[0], transform=transform)
    return profile


@pytest.fixture
def float_45cm_array(float_100cm_array: np.ndarray, float_100cm_profile: Dict, float_45cm_profile: Dict) -> np.ndarray:
    """ 1/.45 upsampled float_100cm_array. """
    float_45cm_array = np.full(
        (float_45cm_profile['height'], float_45cm_profile['width']), float_45cm_profile['nodata']
    )
    _ = reproject(
        float_100cm_array,
        destination=float_45cm_array,
        src_crs=float_100cm_profile['crs'],
        dst_crs=float_45cm_profile['crs'],
        src_transform=float_100cm_profile['transform'],
        dst_transform=float_45cm_profile['transform'],
        src_nodata=float_100cm_profile['nodata'],
        resampling=Resampling.bilinear,
    )  # yapf: disable

    return float_45cm_array


@pytest.fixture
def float_45cm_ra(float_45cm_array: np.ndarray, float_45cm_profile: Dict) -> RasterArray:
    """Raster array with single band of float32 at 45cm pixel resolution. upsampled version of float_100cm_ra, but
    on a different pixel grid"""
    return RasterArray(
        float_45cm_array, float_45cm_profile['crs'], float_45cm_profile['transform'],
        nodata=float_45cm_profile['nodata']
    )


@pytest.fixture
def byte_file(tmp_path: Path, byte_array: np.ndarray, byte_profile: Dict) -> Path:
    """ Single band byte geotiff. """
    filename = tmp_path.joinpath('uint8.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **byte_profile) as ds:
            ds.write(byte_array, indexes=1)
    return filename


@pytest.fixture
def rgba_file(tmp_path: Path, byte_array: np.ndarray, byte_profile: Dict) -> Path:
    """ RGB + alpha band byte geotiff. """
    array = np.stack((byte_array, ) * 4, axis=0)
    array[3] = (array[0] != byte_profile['nodata']) * 255
    filename = tmp_path.joinpath('rgba.tif')
    profile = byte_profile.copy()
    profile.update(
        count=4, nodata=None, colorinterp=[ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
    )
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array, indexes=range(1, 5))
    return filename


@pytest.fixture
def masked_file(tmp_path: Path, byte_array: np.ndarray, byte_profile: Dict) -> Path:
    """ Single band byte geotiff with internal mask (i.e. w/o nodata). """
    filename = tmp_path.joinpath('masked.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **byte_profile) as ds:
            ds.write(byte_array, indexes=1)
            ds.write_mask(byte_array != byte_profile['nodata'])
    return filename


@pytest.fixture
def float_100cm_src_file(tmp_path: Path, float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> Path:
    """ Single band float32 geotiff with 100cm pixel resolution. """
    filename = tmp_path.joinpath('float_100cm_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **float_100cm_profile) as ds:
            ds.write(float_100cm_array, indexes=1)
    return filename


@pytest.fixture
def float_100cm_ref_file(tmp_path: Path, float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> Path:
    """
    Single band float32 geotiff with 100cm pixel resolution, the same as float_100cm_src_file, but padded with an
    extra pixel.
    """
    shape = (np.array(float_100cm_array.shape) + 2).astype('int')
    transform = float_100cm_profile['transform'] * Affine.translation(-1, -1)
    profile = float_100cm_profile.copy()
    profile.update(transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_100cm_ref.tif')
    window = windows.Window(1, 1, float_100cm_array.shape[1], float_100cm_array.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(float_100cm_array, indexes=1, window=window)
    return filename


@pytest.fixture
def float_50cm_src_file(tmp_path: Path, float_50cm_array: np.ndarray, float_50cm_profile: Dict) -> Path:
    """ Single band float32 geotiff with 50cm pixel resolution. """
    filename = tmp_path.joinpath('float_50cm_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **float_50cm_profile) as ds:
            ds.write(float_50cm_array, indexes=1)
    return filename


@pytest.fixture
def float_50cm_ref_file(tmp_path: Path, float_50cm_array: np.ndarray, float_50cm_profile: Dict) -> Path:
    """Single band float32 geotiff with 50cm pixel resolution, the same as float_50cm_src_file, but padded with an
    extra pixel"""
    shape = (np.array(float_50cm_array.shape) + 2).astype('int')
    transform = float_50cm_profile['transform'] * Affine.translation(-1, -1)
    profile = float_50cm_profile.copy()
    profile.update(transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_50cm_ref.tif')
    window = windows.Window(1, 1, float_50cm_array.shape[1], float_50cm_array.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(float_50cm_array, indexes=1, window=window)
    return filename


@pytest.fixture
def float_45cm_src_file(tmp_path: Path, float_45cm_array: np.ndarray, float_45cm_profile: Dict) -> Path:
    """ Single band float32 geotiff with 45cm pixel resolution. """
    filename = tmp_path.joinpath('float_45cm_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **float_45cm_profile) as ds:
            ds.write(float_45cm_array, indexes=1)
    return filename


@pytest.fixture
def float_45cm_ref_file(tmp_path: Path, float_45cm_array: np.ndarray, float_45cm_profile: Dict) -> Path:
    """Single band float32 geotiff with 45cm pixel resolution, the same as float_45cm_src_file, but padded with an
    extra pixel"""
    shape = (np.array(float_45cm_array.shape) + 2).astype('int')
    transform = float_45cm_profile['transform'] * Affine.translation(-1, -1)
    profile = float_45cm_profile.copy()
    profile.update(transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_45cm_ref.tif')
    window = windows.Window(1, 1, float_45cm_array.shape[1], float_45cm_array.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(float_45cm_array, indexes=1, window=window)
    return filename


@pytest.fixture
def float_100cm_rgb_file(tmp_path: Path, float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> Path:
    """ 3 band float32 geotiff with 100cm pixel resolution. """
    array = np.stack((float_100cm_array, ) * 3, axis=0)
    profile = float_100cm_profile.copy()
    profile.update(count=3)
    filename = tmp_path.joinpath('float_100cm_rgb.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array, indexes=[1, 2, 3])
    return filename


@pytest.fixture
def float_50cm_rgb_file(tmp_path: Path, float_50cm_array: np.ndarray, float_50cm_profile: Dict) -> Path:
    """ 3 band float32 geotiff with 50cm pixel resolution, same extent as float_100cm_rgb_file. """
    array = np.stack((float_50cm_array, ) * 3, axis=0)
    profile = float_50cm_profile.copy()
    profile.update(count=3)
    filename = tmp_path.joinpath('float_50cm_rgb.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array, indexes=[1, 2, 3])
    return filename


@pytest.fixture
def float_100cm_sup_src_file(tmp_path: Path, float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> Path:
    """ Single band float32 geotiff with 100cm pixel resolution.  South-up orientation. """
    transform = (
        float_100cm_profile['transform'] * Affine.scale(1, -1) * Affine.translation(0, -float_100cm_array.shape[0])
    )
    profile = float_100cm_profile.copy()
    profile.update(transform=transform)
    filename = tmp_path.joinpath('float_100cm_sup_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(np.flipud(float_100cm_array), indexes=1)
    return filename


@pytest.fixture
def float_100cm_wgs84_src_file(tmp_path: Path, float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> Path:
    """ Single band float32 geotiff with 100cm pixel resolution. WGS84 `projection`.  """
    to_crs = CRS.from_epsg('4326')
    bounds = windows.bounds(windows.Window(0, 0, *float_100cm_array.shape[::-1]), float_100cm_profile['transform'])
    transform, _, _ = calculate_default_transform(
        float_100cm_profile['crs'], to_crs, *float_100cm_array.shape[::-1], *bounds
    )
    profile = float_100cm_profile.copy()
    profile.update(crs=to_crs, transform=transform)
    filename = tmp_path.joinpath('float_100cm_wgs84_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(float_100cm_array, indexes=1)
    return filename


@pytest.fixture
def float_100cm_wgs84_sup_src_file(tmp_path: Path, float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> Path:
    """ Single band float32 geotiff with 100cm pixel resolution. WGS84 `projection` and South-up orientation.  """
    to_crs = CRS.from_epsg('4326')
    bounds = windows.bounds(windows.Window(0, 0, *float_100cm_array.shape[::-1]), float_100cm_profile['transform'])
    transform, _, _ = calculate_default_transform(
        float_100cm_profile['crs'], to_crs, *float_100cm_array.shape[::-1], *bounds
    )
    transform *= Affine.scale(1, -1) * Affine.translation(0, -float_100cm_array.shape[0])    # south up
    profile = float_100cm_profile.copy()
    profile.update(crs=to_crs, transform=transform)
    filename = tmp_path.joinpath('float_100cm_wgs84_sup_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(np.flipud(float_100cm_array), indexes=1)
    return filename


@pytest.fixture
def float_100cm_sup_ref_file(tmp_path: Path, float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> Path:
    """
    Single band float32 geotiff with 100cm pixel resolution, the same as float_100cm_src_file, but padded with an
    extra pixel, and South-up orientation.
    """
    shape = (np.array(float_100cm_array.shape) + 2).astype('int')
    transform = float_100cm_profile['transform'] * Affine.translation(-1, -1)   # padding
    transform *= Affine.scale(1, -1) * Affine.translation(0, -shape[0])   # South-up
    profile = float_100cm_profile.copy()
    profile.update(transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_100cm_sup_ref.tif')
    window = windows.Window(1, 1, float_100cm_array.shape[1], float_100cm_array.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(np.flipud(float_100cm_array), indexes=1, window=window)
    return filename


@pytest.fixture
def float_100cm_wgs84_ref_file(tmp_path: Path, float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> Path:
    """
    Single band float32 geotiff with 100cm pixel resolution, the same as float_100cm_src_file, but padded with an
    extra pixel, and in WGS84.
    """
    shape = (np.array(float_100cm_array.shape) + 2).astype('int')
    to_crs = CRS.from_epsg('4326')
    bounds = windows.bounds(windows.Window(-1, -1, *shape[::-1]), float_100cm_profile['transform'])
    transform, _, _ = calculate_default_transform(
        float_100cm_profile['crs'], to_crs, *shape, *bounds
    )
    profile = float_100cm_profile.copy()
    profile.update(crs=to_crs, transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_100cm_wgs84_ref.tif')
    window = windows.Window(1, 1, float_100cm_array.shape[1], float_100cm_array.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(float_100cm_array, indexes=1, window=window)
    return filename


@pytest.fixture
def float_100cm_wgs84_sup_ref_file(tmp_path: Path, float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> Path:
    """
    Single band float32 geotiff with 100cm pixel resolution, the same as float_100cm_src_file, but padded with an
    extra pixel, in WGS84, and oriented South-up.
    """
    shape = (np.array(float_100cm_array.shape) + 2).astype('int')
    to_crs = CRS.from_epsg('4326')
    bounds = windows.bounds(windows.Window(-1, -1, *shape[::-1]), float_100cm_profile['transform'])
    transform, _, _ = calculate_default_transform(
        float_100cm_profile['crs'], to_crs, *shape, *bounds
    )
    transform *= Affine.scale(1, -1) * Affine.translation(0, -shape[0])    # south up
    profile = float_100cm_profile.copy()
    profile.update(crs=to_crs, transform=transform, width=shape[1], height=shape[0])
    filename = tmp_path.joinpath('float_100cm_wgs84_sup_ref.tif')
    window = windows.Window(1, 1, float_100cm_array.shape[1], float_100cm_array.shape[0])
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'w', **profile) as ds:
        ds.write(np.flipud(float_100cm_array), indexes=1, window=window)
    return filename


@pytest.fixture
def runner() -> CliRunner:
    """ click runner for command line execution. """
    return CliRunner()


@pytest.fixture
def default_fuse_cli_params(tmp_path: Path, float_100cm_ref_file: Path, float_50cm_src_file: Path) -> FuseCliParams:
    """ FuseCliParams with default parameter values. """
    ref_file = float_100cm_ref_file
    src_file = float_50cm_src_file
    model = Model.gain_blk_offset
    kernel_shape = (5, 5)
    proc_crs = ProcCrs.ref
    post_fix = utils.create_out_postfix(proc_crs, model, kernel_shape, RasterFuse.create_out_profile()['driver'])
    corr_file = tmp_path.joinpath(src_file.stem + post_fix)
    param_file = utils.create_param_filename(corr_file)

    cli_str = (f'fuse -od {tmp_path} {src_file} {ref_file}')
    return FuseCliParams(src_file, ref_file, model, kernel_shape, proc_crs, corr_file, param_file, cli_str)


@pytest.fixture
def basic_fuse_cli_params(tmp_path: Path, float_100cm_ref_file: Path, float_100cm_src_file: Path) -> FuseCliParams:
    """ FuseCliParams with basic parameter values. """
    ref_file = float_100cm_ref_file
    src_file = float_100cm_src_file
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
