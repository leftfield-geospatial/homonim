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

from collections import namedtuple
import re

import numpy as np
import pytest
import rasterio as rio
from click.testing import CliRunner
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window

from homonim import root_path
from homonim import utils
from homonim.enums import ProcCrs, Method
from homonim.fuse import RasterFuse
from homonim.raster_array import RasterArray

"""Named tuple to contain fuse cli parameters and string"""
FuseCliParams = namedtuple('FuseCliParams', ['src_file', 'ref_file', 'method', 'kernel_shape', 'proc_crs', 'homo_file',
                                             'param_file', 'cli_str'])

def str_contain_nos(str1, str2):
    """Test if str2 contain str1, ignoring case and whitespace"""
    str1 = re.sub(r'\s+', '', str1.lower())
    str2 = re.sub(r'\s+', '', str2.lower())
    return str1 in str2


@pytest.fixture()
def param_file():
    """Basic parameter image"""
    return root_path.joinpath('data/test_example/param/float_100cm_rgb_HOMO_cREF_mGAIN-OFFSET_k5_5_PARAM.tif')


@pytest.fixture
def byte_array():
    """2D byte gradient image with single pixel nodata=255 border"""
    array = np.array(range(1, 101), dtype='uint8').reshape(20, 5)
    array[:, [0, -1]] = 255
    array[[0, -1], :] = 255
    return array


@pytest.fixture
def float_100cm_array():
    """2D float32 gradient image with single pixel nodata=nan border"""
    array = np.array(range(1, 201), dtype='float32').reshape(20, 10)
    array[:, [0, -1]] = float('nan')
    array[[0, -1], :] = float('nan')
    return array


@pytest.fixture
def float_50cm_array(float_100cm_array):
    """2x upsampled float_100cm_array with double pixel nodata=nan border"""
    array = np.kron(float_100cm_array, np.ones((2, 2)))
    array[:, [0, 1, -2, -1]] = float('nan')
    array[[0, 1, -2, -1], :] = float('nan')
    return array


@pytest.fixture
def byte_profile(byte_array):
    """rasterio profile dict for byte_array"""
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
    """rasterio profile dict for float_100cm_array"""
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
    """rasterio profile dict for float_50cm_array"""
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
    """Raster array with single band of byte"""
    return RasterArray(byte_array, byte_profile['crs'], byte_profile['transform'],
                       nodata=byte_profile['nodata'])


@pytest.fixture
def rgb_byte_ra(byte_array, byte_profile):
    """Raster array with three bands of byte"""
    return RasterArray(np.stack((byte_array,) * 3, axis=0), byte_profile['crs'], byte_profile['transform'],
                       nodata=byte_profile['nodata'])


@pytest.fixture
def float_100cm_ra(float_100cm_array, float_100cm_profile):
    """Raster array with single band of float32 at 100cm pixel resolution"""
    return RasterArray(float_100cm_array, float_100cm_profile['crs'], float_100cm_profile['transform'],
                       nodata=float_100cm_profile['nodata'])


@pytest.fixture
def float_50cm_ra(float_50cm_array, float_50cm_profile):
    """Raster array with single band of float32 at 50cm pixel resolution. 2x upsampled version of float_100cm_ra"""
    return RasterArray(float_50cm_array, float_50cm_profile['crs'], float_50cm_profile['transform'],
                       nodata=float_50cm_profile['nodata'])


@pytest.fixture
def float_45cm_profile(float_100cm_array, float_100cm_profile):
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
def float_45cm_array(float_100cm_array, float_100cm_profile, float_45cm_profile):
    """1/.45 upsampled float_100cm_array"""
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
    """Raster array with single band of float32 at 45cm pixel resolution. upsampled version of float_100cm_ra, but
    on a different pixel grid"""
    return RasterArray(float_45cm_array, float_45cm_profile['crs'], float_45cm_profile['transform'],
                       nodata=float_45cm_profile['nodata'])


@pytest.fixture
def byte_file(tmp_path, byte_array, byte_profile):
    """Single band byte geotiff"""
    filename = tmp_path.joinpath('uint8.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **byte_profile) as ds:
            ds.write(byte_array, indexes=1)
    return filename


@pytest.fixture
def rgba_file(tmp_path, byte_array, byte_profile):
    """RGB + alpha band byte geotiff"""
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
    """Single band byte geotiff with internal mask (i.e. w/o nodata)"""
    filename = tmp_path.joinpath('masked.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **byte_profile) as ds:
            ds.write(byte_array, indexes=1)
            ds.write_mask(byte_array != byte_profile['nodata'])
    return filename


@pytest.fixture
def float_100cm_src_file(tmp_path, float_100cm_array, float_100cm_profile):
    """Single band float32 geotiff with 100cm pixel resolution"""
    filename = tmp_path.joinpath('float_100cm_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **float_100cm_profile) as ds:
            ds.write(float_100cm_array, indexes=1)
    return filename


@pytest.fixture
def float_100cm_ref_file(tmp_path, float_100cm_array, float_100cm_profile):
    """Single band float32 geotiff with 100cm pixel resolution, the same as float_100cm_src_file, but padded with an
    extra pixel"""
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
    """Single band float32 geotiff with 50cm pixel resolution"""
    filename = tmp_path.joinpath('float_50cm_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **float_50cm_profile) as ds:
            ds.write(float_50cm_array, indexes=1)
    return filename


@pytest.fixture
def float_50cm_ref_file(tmp_path, float_50cm_array, float_50cm_profile):
    """Single band float32 geotiff with 50cm pixel resolution, the same as float_50cm_src_file, but padded with an
    extra pixel"""
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
    """Single band float32 geotiff with 45cm pixel resolution"""
    filename = tmp_path.joinpath('float_45cm_src.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **float_45cm_profile) as ds:
            ds.write(float_45cm_array, indexes=1)
    return filename


@pytest.fixture
def float_45cm_ref_file(tmp_path, float_45cm_array, float_45cm_profile):
    """Single band float32 geotiff with 45cm pixel resolution, the same as float_45cm_src_file, but padded with an
    extra pixel"""
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
    """3 band float32 geotiff with 100cm pixel resolution"""
    array = np.stack((float_100cm_array,) * 3, axis=0)
    profile = float_100cm_profile.copy()
    profile.update(count=3)
    filename = tmp_path.joinpath('float_100cm_rgb.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array, indexes=[1, 2, 3])
    return filename


@pytest.fixture
def float_50cm_rgb_file(tmp_path, float_50cm_array, float_50cm_profile):
    """3 band float32 geotiff with 50cm pixel resolution, same extent as float_100cm_rgb_file"""
    array = np.stack((float_50cm_array,) * 3, axis=0)
    profile = float_50cm_profile.copy()
    profile.update(count=3)
    filename = tmp_path.joinpath('float_50cm_rgb.tif')
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
        with rio.open(filename, 'w', **profile) as ds:
            ds.write(array, indexes=[1, 2, 3])
    return filename


@pytest.fixture
def runner():
    """click runner for command line execution"""
    return CliRunner()


@pytest.fixture
def default_fuse_cli_params(tmp_path, float_100cm_ref_file, float_50cm_src_file):
    """FuseCliParams with default parameter values"""
    ref_file = float_100cm_ref_file
    src_file = float_50cm_src_file
    method = Method.gain_blk_offset
    kernel_shape = (5, 5)
    proc_crs = ProcCrs.ref
    post_fix = utils.create_homo_postfix(proc_crs, method, kernel_shape, RasterFuse.default_out_profile['driver'])
    homo_file = tmp_path.joinpath(src_file.stem + post_fix)
    param_file = utils.create_param_filename(homo_file)

    cli_str = (f'fuse -od {tmp_path} {src_file} {ref_file}')
    return FuseCliParams(src_file, ref_file, method, kernel_shape, proc_crs, homo_file, param_file, cli_str)


@pytest.fixture
def basic_fuse_cli_params(tmp_path, float_100cm_ref_file, float_100cm_src_file):
    """FuseCliParams with basic parameter values"""
    ref_file = float_100cm_ref_file
    src_file = float_100cm_src_file
    method = Method.gain_blk_offset
    kernel_shape = (3, 3)
    proc_crs = ProcCrs.ref
    post_fix = utils.create_homo_postfix(proc_crs, method, kernel_shape, RasterFuse.default_out_profile['driver'])
    homo_file = tmp_path.joinpath(src_file.stem + post_fix)
    param_file = utils.create_param_filename(homo_file)

    cli_str = (f'fuse -m {method.value} -k {kernel_shape[0]} {kernel_shape[1]} -od {tmp_path} -pc {proc_crs.value} '
               f'{src_file} {ref_file}')
    return FuseCliParams(src_file, ref_file, method, kernel_shape, proc_crs, homo_file, param_file, cli_str)
