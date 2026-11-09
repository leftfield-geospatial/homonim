# Copyright Leftfield Geospatial
#
# This file is part of Homonim.
#
# Homonim is free software: you can redistribute it and/or modify it under the terms
# of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# Homonim is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with
# Homonim. If not, see <https://www.gnu.org/licenses/>.
import re
from inspect import getsourcefile
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from click.testing import CliRunner
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from homonim.raster_array import RasterArray

# path to the test data root
data_path = Path(getsourcefile(lambda: None)).parent.joinpath('data')

# default CRS for georeferenced fixtures
_webmerc_crs = CRS.from_epsg(3857)


def str_contain_no_space(str1: str, str2: str) -> bool:
    """Return True if str2 contains str1, ignoring case and whitespace, otherwise
    False.
    """
    str1 = re.sub(r'\s+', '', str1.lower())
    str2 = re.sub(r'\s+', '', str2.lower())
    return str1 in str2


def get_nodata(dtype) -> int | float:
    """Return a nodata value for the given dtype."""
    return np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else float('nan')


def create_profile(
    array: np.ndarray, transform: Affine, crs: CRS = None, **kwargs
) -> dict:
    """Return a RasterIO GTiff profile for the given array."""
    crs = crs or _webmerc_crs
    return dict(
        driver='GTiff',
        crs=crs,
        transform=transform,
        width=array.shape[-1],
        height=array.shape[-2],
        count=1 if array.ndim == 2 else array.shape[0],
        dtype=array.dtype.name,
        nodata=get_nodata(array.dtype),
        **kwargs,
    )


@pytest.fixture
def param_file() -> Path:
    """Parameter image for a gain-offset model with gain~=1 and offset~=0
    (16x16 tiles).
    """
    return data_path.joinpath(
        'parameter/float_100cm_rgb_FUSE_cREF_mGAIN-OFFSET_k5_5_PARAM.tif'
    )


@pytest.fixture
def param_file_tile_10x20() -> Path:
    """Parameter image for a gain-offset model with gain~=1 and offset~=0
    (10x20 tiles).
    """
    return data_path.joinpath(
        'parameter/float_100cm_rgb_FUSE_cREF_mGAIN-OFFSET_k5_5_PARAM_tile_10x20.tif'
    )


@pytest.fixture
def array_byte() -> np.ndarray:
    """2D uint8 gradient image with a single pixel nodata border."""
    array = np.array(range(1, 201), dtype='uint8').reshape(20, 10)
    nodata = get_nodata(array.dtype)
    array[:, [0, -1]] = nodata
    array[[0, -1], :] = nodata
    return array


@pytest.fixture
def array_100cm_float(array_byte: np.ndarray) -> np.ndarray:
    """2D float32 gradient image with a single pixel nodata border."""
    array = array_byte.astype('float32')
    array[array == get_nodata(array_byte.dtype)] = get_nodata(array.dtype)
    return array


@pytest.fixture
def array_50cm_float(array_100cm_float) -> np.ndarray:
    """array_100cm_float 2x upsampled with a double pixel nodata border."""
    array = np.kron(array_100cm_float, np.ones((2, 2), dtype=array_100cm_float.dtype))
    nodata = get_nodata(array.dtype)
    array[:, [0, 1, -2, -1]] = nodata
    array[[0, 1, -2, -1], :] = nodata
    return array


@pytest.fixture
def transform_100cm() -> Affine:
    """Georeferencing transform with 100cm resolution."""
    # North-up, with origin at (5, -5)
    return Affine(1, 0, 0, 0, -1, 0) * Affine.translation(5, 5)


@pytest.fixture
def transform_50cm(transform_100cm) -> Affine:
    """Georeferencing transform with the same origin as transform_100cm, and 50cm
    resolution.
    """
    return transform_100cm * Affine.scale(0.5)


@pytest.fixture
def transform_45cm(transform_100cm) -> dict:
    """Georeferencing transform with the origin of transform_100cm shifted by half a
    pixel, and 45cm resolution.
    """
    return transform_100cm * Affine.scale(0.45) * Affine.translation(-0.5, -0.5)


@pytest.fixture
def array_45cm_float(array_100cm_float, transform_100cm, transform_45cm) -> np.ndarray:
    """array_100cm_float 1/.45x upsampled."""
    scale = 0.45  # resolution scaling
    shape = [round(s / scale) + 1 for s in array_100cm_float.shape]
    dtype = array_100cm_float.dtype
    nodata = get_nodata(dtype)
    array = np.full(shape, nodata, dtype=dtype)

    _ = reproject(
        array_100cm_float,
        destination=array,
        src_crs=_webmerc_crs,
        dst_crs=_webmerc_crs,
        src_transform=transform_100cm,
        dst_transform=transform_45cm,
        src_nodata=nodata,
        resampling=Resampling.bilinear,
    )
    return array


@pytest.fixture
def ra_byte(array_byte, transform_100cm) -> RasterArray:
    """RasterArray with a single uint8 band."""
    return RasterArray(
        array_byte, _webmerc_crs, transform_100cm, nodata=get_nodata(array_byte.dtype)
    )


@pytest.fixture
def ra_rgb_byte(array_byte, transform_100cm) -> RasterArray:
    """RasterArray with three uint8 bands."""
    array = np.stack((array_byte,) * 3, axis=0)
    return RasterArray(
        array, _webmerc_crs, transform_100cm, nodata=get_nodata(array.dtype)
    )


@pytest.fixture
def ra_100cm_float(array_100cm_float, transform_100cm) -> RasterArray:
    """RasterArray with a single float32 band at 100cm resolution."""
    return RasterArray(
        array_100cm_float,
        _webmerc_crs,
        transform_100cm,
        nodata=get_nodata(array_100cm_float.dtype),
    )


@pytest.fixture
def ra_50cm_float(array_50cm_float, transform_50cm) -> RasterArray:
    """RasterArray with a single float32 band at 50cm resolution (ra_100cm_float 2x
    upsampled).
    """
    return RasterArray(
        array_50cm_float,
        _webmerc_crs,
        transform_50cm,
        nodata=get_nodata(array_50cm_float.dtype),
    )


@pytest.fixture
def file_byte(tmp_path: Path, array_byte, transform_100cm) -> Path:
    """Single band uint8 GeoTIFF."""
    file = tmp_path.joinpath('uint8.tif')
    profile = create_profile(array_byte, transform_100cm)
    with rio.open(file, 'w', **profile) as ds:
        ds.write(array_byte, indexes=1)
    return file


@pytest.fixture
def file_rgba(tmp_path: Path, array_byte, transform_100cm) -> Path:
    """RGB + alpha uint8 GeoTIFF (without nodata)."""
    file = tmp_path.joinpath('rgba.tif')
    array = np.stack((array_byte,) * 4, axis=0)
    array[3] = (array[0] != get_nodata(array_byte.dtype)) * 255
    profile = create_profile(array, transform_100cm)
    profile.update(nodata=None)

    with rio.open(file, 'w', **profile) as ds:
        ds.colorinterp = [
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        ]
        ds.write(array)
    return file


@pytest.fixture
def file_bgr(tmp_path: Path, array_byte, transform_100cm) -> Path:
    """BGR uint8 GeoTIFF."""
    file = tmp_path.joinpath('bgr.tif')
    array = np.stack((array_byte,) * 3, axis=0)
    profile = create_profile(array, transform_100cm)

    with rio.open(file, 'w', **profile) as ds:
        ds.colorinterp = [ColorInterp.blue, ColorInterp.green, ColorInterp.red]
        ds.write(array)
    return file


@pytest.fixture
def file_masked(tmp_path: Path, array_byte, transform_100cm) -> Path:
    """Single band uint8 GeoTIFF with internal mask (without nodata)."""
    file = tmp_path.joinpath('masked.tif')
    profile = create_profile(array_byte, transform_100cm)
    profile.update(nodata=None)

    with rio.open(file, 'w', **profile) as ds:
        ds.write(array_byte, indexes=1)
        ds.write_mask(array_byte != get_nodata(array_byte.dtype))
    return file


@pytest.fixture
def src_file_100cm_float(tmp_path: Path, array_100cm_float, transform_100cm) -> Path:
    """Single band float32 GeoTIFF with 100cm resolution."""
    file = tmp_path.joinpath('float_100cm_src.tif')
    profile = create_profile(array_100cm_float, transform_100cm)
    with rio.open(file, 'w', **profile) as ds:
        ds.write(array_100cm_float, indexes=1)
    return file


@pytest.fixture
def ref_file_100cm_float(tmp_path: Path, array_100cm_float, transform_100cm) -> Path:
    """Single band float32 GeoTIFF with 100cm resolution, the same as
    src_file_100cm_float, but padded with an extra nodata pixel.
    """
    file = tmp_path.joinpath('float_100cm_ref.tif')
    nodata = get_nodata(array_100cm_float.dtype)
    array = np.pad(array_100cm_float, 1, 'constant', constant_values=nodata)
    transform = transform_100cm * Affine.translation(-1, -1)

    profile = create_profile(array, transform)
    with rio.open(file, 'w', **profile) as ds:
        ds.write(array, indexes=1)
    return file


@pytest.fixture
def src_file_50cm_float(tmp_path: Path, array_50cm_float, transform_50cm) -> Path:
    """Single band float32 GeoTIFF with 50cm resolution."""
    file = tmp_path.joinpath('float_50cm_src.tif')
    profile = create_profile(array_50cm_float, transform_50cm)
    with rio.open(file, 'w', **profile) as ds:
        ds.write(array_50cm_float, indexes=1)
    return file


@pytest.fixture
def ref_file_50cm_float(tmp_path: Path, array_50cm_float, transform_50cm) -> Path:
    """Single band float32 GeoTIFF with 50cm resolution, the same as
    src_file_50cm_float, but padded with an extra nodata pixel.
    """
    file = tmp_path.joinpath('float_50cm_ref.tif')
    nodata = get_nodata(array_50cm_float.dtype)
    array = np.pad(array_50cm_float, 1, 'constant', constant_values=nodata)
    transform = transform_50cm * Affine.translation(-1, -1)

    profile = create_profile(array, transform)
    with rio.open(file, 'w', **profile) as ds:
        ds.write(array, indexes=1)
    return file


@pytest.fixture
def src_file_45cm_float(tmp_path: Path, array_45cm_float, transform_45cm) -> Path:
    """Single band float32 GeoTIFF with 45cm resolution."""
    file = tmp_path.joinpath('float_45cm_src.tif')
    profile = create_profile(array_45cm_float, transform_45cm)
    with rio.open(file, 'w', **profile) as ds:
        ds.write(array_45cm_float, indexes=1)
    return file


@pytest.fixture
def ref_file_45cm_float(tmp_path: Path, array_45cm_float, transform_45cm) -> Path:
    """Single band float32 GeoTIFF with 45cm resolution, the same as
    src_file_45cm_float, but padded with an extra nodata pixel.
    """
    file = tmp_path.joinpath('float_45cm_ref.tif')
    nodata = get_nodata(array_45cm_float.dtype)
    array = np.pad(array_45cm_float, 1, 'constant', constant_values=nodata)
    transform = transform_45cm * Affine.translation(-1, -1)

    profile = create_profile(array, transform)
    with rio.open(file, 'w', **profile) as ds:
        ds.write(array, indexes=1)
    return file


@pytest.fixture
def file_rgb_100cm_float(tmp_path: Path, array_100cm_float, transform_100cm) -> Path:
    """RGB float32 GeoTIFF with 100cm resolution."""
    file = tmp_path.joinpath('float_100cm_rgb.tif')
    array = np.stack(
        [i * array_100cm_float for i in range(1, 4)],
        axis=0,
        dtype=array_100cm_float.dtype,
    )
    profile = create_profile(array, transform_100cm)
    with rio.open(file, 'w', **profile) as ds:
        ds.write(array)
    return file


@pytest.fixture
def file_rgb_50cm_float(tmp_path: Path, array_50cm_float, transform_50cm) -> Path:
    """RGB float32 GeoTIFF with 50cm resolution."""
    file = tmp_path.joinpath('float_50cm_rgb.tif')
    array = np.stack(
        [i * array_50cm_float for i in range(1, 4)],
        axis=0,
        dtype=array_50cm_float.dtype,
    )
    profile = create_profile(array, transform_50cm)
    with rio.open(file, 'w', **profile) as ds:
        ds.write(array)
    return file


@pytest.fixture
def src_file_sup_100cm_float(
    tmp_path: Path, array_100cm_float, transform_100cm
) -> Path:
    """Single band float32 GeoTIFF with 100cm resolution and a South-up orientation."""
    file = tmp_path.joinpath('float_100cm_sup_src.tif')
    transform = (
        transform_100cm
        * Affine.scale(1, -1)
        * Affine.translation(0, -array_100cm_float.shape[0])
    )
    profile = create_profile(array_100cm_float, transform)
    with rio.open(file, 'w', **profile) as ds:
        ds.write(np.flipud(array_100cm_float), indexes=1)
    return file


@pytest.fixture
def src_file_wgs84_100cm_float(
    tmp_path: Path, array_100cm_float, transform_100cm
) -> Path:
    """Single band float32 GeoTIFF with 100cm resolution in WGS84."""
    file = tmp_path.joinpath('float_100cm_wgs84_src.tif')
    nodata = get_nodata(array_100cm_float.dtype)
    array = np.full_like(array_100cm_float, nodata)
    crs = CRS.from_epsg('4326')
    _, transform = reproject(
        array_100cm_float,
        destination=array,
        src_crs=_webmerc_crs,
        dst_crs=crs,
        src_transform=transform_100cm,
        src_nodata=nodata,
        resampling=Resampling.bilinear,
    )

    profile = create_profile(array, transform, crs=crs)
    with rio.open(file, 'w', **profile) as ds:
        ds.write(array, indexes=1)
    return file


@pytest.fixture
def ref_file_wgs84_100cm_float(
    tmp_path: Path, array_100cm_float, transform_100cm
) -> Path:
    """Single band float32 GeoTIFF with 100cm resolution, the same as
    src_file_wgs84_100cm_float, but padded with an extra nodata pixel.
    """
    file = tmp_path.joinpath('float_100cm_wgs84_ref.tif')
    nodata = get_nodata(array_100cm_float.dtype)
    array = np.full_like(array_100cm_float, nodata)
    crs = CRS.from_epsg('4326')
    _, transform = reproject(
        array_100cm_float,
        destination=array,
        src_crs=_webmerc_crs,
        dst_crs=crs,
        src_transform=transform_100cm,
        src_nodata=nodata,
        resampling=Resampling.bilinear,
    )
    array = np.pad(array, 1, 'constant', constant_values=nodata)
    transform *= Affine.translation(-1, -1)

    profile = create_profile(array, transform, crs=crs)
    with rio.open(file, 'w', **profile) as ds:
        ds.write(array, indexes=1)
    return file


@pytest.fixture
def modis_ref_file() -> Path:
    return data_path.joinpath('reference/modis_nbar.tif')


@pytest.fixture
def landsat_ref_file() -> Path:
    return data_path.joinpath('reference/landsat8_byte.tif')


@pytest.fixture
def s2_ref_file() -> Path:
    return data_path.joinpath('reference/sentinel2_b432_byte.tif')


@pytest.fixture
def landsat_src_file() -> Path:
    return data_path.joinpath('reference/landsat8_byte.vrt')


@pytest.fixture
def ngi_src_files() -> tuple[Path, ...]:
    source_root = data_path.joinpath('source/')
    return tuple(sorted(source_root.glob('ngi_rgb_byte_*.tif')))


@pytest.fixture
def ngi_src_file() -> Path:
    return data_path.joinpath('source/ngi_rgb_byte_1.tif')


@pytest.fixture
def runner() -> CliRunner:
    """click runner for command line execution."""
    return CliRunner()
