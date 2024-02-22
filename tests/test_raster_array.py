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

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling, MaskFlags
from rasterio.transform import Affine, from_bounds
from rasterio.windows import Window
from rasterio.warp import transform_bounds

from homonim.errors import ImageProfileError, ImageFormatError
from homonim.raster_array import RasterArray
from homonim import utils


def test_read_only_properties(array_byte, profile_byte):
    """ Test RasterArray read-only properties. """
    basic_ra = RasterArray(array_byte, profile_byte['crs'], profile_byte['transform'], nodata=profile_byte['nodata'])
    assert basic_ra.shape == array_byte.shape
    assert basic_ra.width == array_byte.shape[-1]
    assert basic_ra.height == array_byte.shape[-2]
    assert basic_ra.count == 1 if array_byte.ndim < 3 else array_byte.ndim
    assert basic_ra.nodata == profile_byte['nodata']
    assert basic_ra.transform == profile_byte['transform']
    assert basic_ra.dtype == array_byte.dtype
    assert basic_ra.res == (profile_byte['transform'].a, -profile_byte['transform'].e)
    assert basic_ra.crs == profile_byte['crs']


def test_array_property(array_byte, profile_byte):
    """ Test array get/set. """
    byte_ra = RasterArray(array_byte, profile_byte['crs'], profile_byte['transform'], nodata=profile_byte['nodata'])
    assert (byte_ra.array == array_byte).all()  # test get

    array = array_byte / 2
    byte_ra.array = array
    assert (byte_ra.array == array).all()  # test set with same num bands

    array = np.stack((array, array), axis=0)
    byte_ra.array = array
    assert (byte_ra.array == array).all()  # test set with more bands
    assert byte_ra.count == array.shape[0]


def test_nodata_mask(ra_byte):
    """ Test set/get nodata and mask properties. """
    mask = ra_byte.mask
    ra_byte.nodata = 254
    assert ra_byte.nodata == 254
    assert (ra_byte.mask == mask).all()  # test mask unchanged after setting nodata
    assert (ra_byte.mask_ra.array == mask).all()
    assert ra_byte.mask_ra.transform == ra_byte.transform

    # test mask changed after setting altered masked array
    array = ra_byte.array.copy()
    array[np.divide(mask.shape, 2).astype('int')] = ra_byte.nodata
    mask[np.divide(mask.shape, 2).astype('int')] = False
    ra_byte.array = array
    assert (ra_byte.mask == mask).all()

    # test mask unchanged after setting stacked array
    ra_byte.array = np.stack((ra_byte.array, ra_byte.array), axis=0)
    assert (ra_byte.mask == mask).all()

    # test altering the mask
    mask[(np.divide(mask.shape, 2) + 1).astype('int')] = False
    ra_byte.mask = mask
    assert (ra_byte.mask == mask).all()

    # test removing nodata
    ra_byte.nodata = None
    assert ra_byte.mask.all()


def test_array_set_shape(ra_byte):
    """ Test setting array with different rows/cols raises error. """
    with pytest.raises(ValueError):
        ra_byte.array = ra_byte.array.reshape(-1, 1)


def test_from_profile(array_byte, profile_byte):
    """ Test creating raster array from rasterio profile dict. """
    byte_ra = RasterArray.from_profile(array_byte, profile_byte)
    assert (byte_ra.array == array_byte).all()
    assert byte_ra.transform == profile_byte['transform']


def test_from_profile_noarray(profile_byte):
    """ Test creating raster array from rasterio profile dict w/o array. """
    byte_ra = RasterArray.from_profile(None, profile_byte)
    assert (byte_ra.array == byte_ra.nodata).all()


@pytest.mark.parametrize('missing_key', ['crs', 'transform', 'nodata', 'width', 'height', 'count', 'dtype'])
def test_from_profile_missingkey(profile_byte, missing_key: str):
    """ Test an error is raised when creating raster array from a rasterio profile dict that is missing a key. """
    profile = profile_byte
    profile.pop(missing_key)
    with pytest.raises(ImageProfileError):
        RasterArray.from_profile(None, profile)


def test_from_rio_dataset(file_byte):
    """ Test an error is raised when creating raster array from a rasterio profile dict that is missing a key. """
    with rio.open(file_byte, 'r') as ds:
        # check default
        ds_ra = RasterArray.from_rio_dataset(ds)
        assert ds_ra.shape == ds.shape
        assert ds_ra.count == ds.count
        assert ds_ra.nodata == ds.nodata
        assert ds_ra.dtype == RasterArray.default_dtype

        # create boundless raster array that extends beyond ds
        pad = [1, 1]
        indexes = ds.indexes[0]
        window = Window(-pad[1], -pad[0], ds.width + 2 * pad[1], ds.height + 2 * pad[0])
        ds_ra_boundless = RasterArray.from_rio_dataset(ds, indexes=indexes, window=window)
        assert ds_ra_boundless.shape == (window.height, window.width)
        assert ds_ra_boundless.count == 1

        # check boundless array contents and transform against ds_ra
        bounded_win = Window(pad[1], pad[0], ds_ra.width, ds_ra.height)
        assert (ds_ra_boundless.array[bounded_win.toslices()] == ds_ra.array).all()
        test_transform = ds_ra.transform * Affine.translation(-pad[1], -pad[0])
        assert (
            ds_ra_boundless.transform.xoff == test_transform.xoff and
            ds_ra_boundless.transform.yoff == test_transform.yoff
        )


@pytest.mark.parametrize('file, count', [('file_masked', 1), ('file_rgba', 3)])
def test_from_rio_dataset_masked(file: str, count: int, request: pytest.FixtureRequest):
    """ Test creating raster array from nodata and internally masked datasets. """
    file: Path = request.getfixturevalue(file)
    with rio.open(file, 'r') as ds:
        ds_mask = ds.dataset_mask().astype('bool', copy=False)
        ra = RasterArray.from_rio_dataset(ds)
        assert ra.count == count
        assert np.isnan(ra.nodata)
        assert (ra.mask == ds_mask).all()


@pytest.mark.parametrize('pad', [[1, 1], [-1, -1]])
def test_bounded_window_slices(file_byte, pad: Tuple[int, int]):
    """ Test RasterArray.bounded_window_slices() with bounded and boundless windows. """
    with rio.open(file_byte, 'r') as ds:
        window = Window(-pad[1], -pad[0], ds.width + 2 * pad[1], ds.height + 2 * pad[0])
        bounded_win, bounded_slices = RasterArray.bounded_window_slices(ds, window)
        # test that the returned window is bounded by the dataset
        assert (bounded_win.col_off == max(0, -pad[1]) and bounded_win.row_off == max(0, -pad[0]))
        assert (
            bounded_win.width == min(ds.width, ds.width + 2 * pad[1]) and
            bounded_win.height == min(ds.height, ds.height + 2 * pad[0])
        )


def test_slice_to_bounds(ra_byte):
    """ Test RasterArray.slice_to_bounds(). """
    # test valid bounded window
    window = Window(1, 1, ra_byte.width - 2, ra_byte.height - 2)
    bounds = ra_byte.window_bounds(window)
    slice_ra = ra_byte.slice_to_bounds(*bounds)
    assert slice_ra.bounds == pytest.approx(bounds)
    assert (slice_ra.array == ra_byte.array[window.toslices()]).all()

    # test invalid boundless window
    with pytest.raises(ValueError):
        ra_byte.slice_to_bounds(*ra_byte.window_bounds(Window(-1, -1, ra_byte.width, ra_byte.height)))


def test_to_rio_dataset(ra_byte, tmp_path: Path):
    """ Test writing raster array to dataset. """
    ds_filename = tmp_path.joinpath('temp.tif')
    with rio.open(ds_filename, 'w', driver='GTiff', **ra_byte.profile) as ds:
        ra_byte.to_rio_dataset(ds)
    with rio.open(ds_filename, 'r') as ds:
        test_array = ds.read(indexes=1)
    assert (test_array == ra_byte.array).all()


def test_to_rio_dataset_nodata_none(ra_byte, tmp_path: Path):
    """ Test writing raster array to dataset with nodata=None writes an internal mask. """
    ds_filename = tmp_path.joinpath('temp.tif')
    profile = ra_byte.profile
    profile.update(nodata=None)
    with rio.open(ds_filename, 'w', driver='GTiff', **profile) as ds:
        ra_byte.to_rio_dataset(ds)

    with rio.open(ds_filename, 'r') as ds:
        assert ds.nodata is None
        assert ds.mask_flag_enums[0] == [MaskFlags.per_dataset]
        test_mask = ds.dataset_mask().astype('bool')
        test_array = ds.read(indexes=1)
    assert (test_mask == ra_byte.mask).all()
    assert (test_array[test_mask] == ra_byte.array[ra_byte.mask]).all()


def test_to_rio_dataset_crop(ra_rgb_byte, tmp_path: Path):
    """ Test writing a raster array to a dataset where the dataset & raster array sizes differ. """
    ds_filename = tmp_path.joinpath('temp.tif')
    indexes = [1, 2, 3]
    # crop the raster array and write to full dataset
    crop_window = Window(1, 1, ra_rgb_byte.width - 2, ra_rgb_byte.height - 2)
    crop_ra = ra_rgb_byte.slice_to_bounds(*ra_rgb_byte.window_bounds(crop_window))
    with rio.open(ds_filename, 'w', driver='GTiff', **ra_rgb_byte.profile) as ds:
        crop_ra.to_rio_dataset(ds, indexes=indexes, window=crop_window)
    with rio.open(ds_filename, 'r') as ds:
        test_array = ds.read(indexes=indexes)
    assert (test_array[(np.array(indexes) - 1, *crop_window.toslices())] == crop_ra.array).all()

    # crop the dataset and write in the full raster array
    with rio.open(ds_filename, 'w', driver='GTiff', **crop_ra.profile) as ds:
        ra_rgb_byte.to_rio_dataset(ds, indexes=indexes)
    with rio.open(ds_filename, 'r') as ds:
        test_array = ds.read(indexes=indexes)
    assert (test_array == ra_rgb_byte.array[(np.array(indexes) - 1, *crop_window.toslices())]).all()


def test_to_rio_dataset_exceptions(ra_rgb_byte, tmp_path: Path):
    """ Test possible error conditions when writing a raster array to a dataset. """
    ds_filename = tmp_path.joinpath('temp.tif')
    with rio.open(ds_filename, 'w', driver='GTiff', **ra_rgb_byte.profile) as ds:
        with pytest.raises(ValueError):
            # window lies outside the bounds of raster array
            crop_window = Window(1, 1, ra_rgb_byte.width - 2, ra_rgb_byte.height - 2)
            crop_ra = ra_rgb_byte.slice_to_bounds(*ra_rgb_byte.window_bounds(crop_window))
            boundless_window = Window(-1, -1, ra_rgb_byte.width + 2, ra_rgb_byte.height + 2)
            crop_ra.to_rio_dataset(ds, indexes=[1, 2, 3], window=boundless_window)
        with pytest.raises(ValueError):
            # len(indexes) > number of dataset bands
            ra_rgb_byte.to_rio_dataset(ds, indexes=[1] * (ds.count + 1))
        with pytest.raises(ValueError):
            # indexes outside of valid range
            ra_rgb_byte.to_rio_dataset(ds, indexes=ds.count + 1)

    # dataset and raster array have different CRSs
    profile = ra_rgb_byte.profile
    profile.update(crs=CRS.from_epsg(4326))
    with rio.open(ds_filename, 'w', driver='GTiff', **profile) as ds:
        with pytest.raises(ImageFormatError):
            ra_rgb_byte.to_rio_dataset(ds, indexes=[1, 2, 3])

    # dataset and raster array have different resolutions
    profile = ra_rgb_byte.profile
    profile.update(transform=Affine.identity() * Affine.scale(0.5))
    with rio.open(ds_filename, 'w', driver='GTiff', **profile) as ds:
        with pytest.raises(ImageFormatError):
            ra_rgb_byte.to_rio_dataset(ds, indexes=[1, 2, 3])


def test_reprojection(ra_rgb_byte):
    """ Test raster array re-projection. """
    # reproject to WGS84 with default parameters, assuming ra_rgb_byte is North up
    to_crs = CRS.from_epsg(4326)
    reprj_ra = ra_rgb_byte.reproject(crs=to_crs, resampling=Resampling.nearest)
    assert (reprj_ra.crs == to_crs)
    abs_diff = np.abs(reprj_ra.array[:, reprj_ra.mask] - ra_rgb_byte.array[:, ra_rgb_byte.mask])
    assert abs_diff.mean() == pytest.approx(0, abs=.1)

    # reproject with rescaling to WGS84 using a specified transform & shape
    to_bounds = transform_bounds(ra_rgb_byte.crs, to_crs, *ra_rgb_byte.bounds)
    to_shape = tuple(np.array(ra_rgb_byte.shape) * 2)
    to_transform = from_bounds(*to_bounds, *to_shape[::-1])
    reprj_ra = ra_rgb_byte.reproject(
        crs=to_crs, transform=to_transform, shape=to_shape, resampling=Resampling.bilinear
    )
    assert (reprj_ra.crs == to_crs)
    assert (reprj_ra.transform == to_transform)
    assert (reprj_ra.shape == to_shape)
    assert (reprj_ra.bounds == pytest.approx(to_bounds, abs=1.e-9))
    assert (
        reprj_ra.array[:, reprj_ra.mask].mean() == pytest.approx(ra_rgb_byte.array[:, ra_rgb_byte.mask].mean(), abs=.1)
    )


@pytest.mark.parametrize('src_dtype, src_nodata, dst_dtype, dst_nodata', [
    ('float32', float('nan'), 'uint8', 1),
    ('float32', float('nan'), 'int8', 1),
    ('float32', float('nan'), 'uint16', 1),
    ('float32', float('nan'), 'int16', 1),
    ('float32', float('nan'), 'uint32', 1),
    ('float32', float('nan'), 'int32', 1),
    # ('float32', float('nan'), 'int64', 0),  # overflow
    ('float32', float('nan'), 'float32', float('nan')),
    ('float32', float('nan'), 'float64', float('nan')),
    ('float64', float('nan'), 'int32', 1),
    # ('float64', float('nan'), 'int64', 1),  # overflow
    ('float64', float('nan'), 'float32', float('nan')),
    ('float64', float('nan'), 'float64', float('nan')),
    ('int64', 1, 'int32', 1),
    ('int64', 1, 'int64', 1),
    ('int64', 1, 'float32', float('nan')),
    ('int64', 1, 'float64', float('nan')),
    ('float32', float('nan'), 'float32', None),  # nodata unchanged
])
def test_convert_array_dtype(profile_100cm_float: dict, src_dtype: str, src_nodata: float, dst_dtype: str, dst_nodata: float):
    """ Test dtype conversion with combinations covering rounding, clipping (with and w/o type promotion) and
    re-masking.
    """
    src_dtype_info = np.iinfo(src_dtype) if np.issubdtype(src_dtype, np.integer) else np.finfo(src_dtype)
    dst_dtype_info = np.iinfo(dst_dtype) if np.issubdtype(dst_dtype, np.integer) else np.finfo(dst_dtype)

    # create array that spans the src_dtype range, includes decimals, excludes -1..1 (to allow nodata == +-1),
    # and is padded with nodata
    array = np.geomspace(2, src_dtype_info.max, 50, dtype=src_dtype).reshape(5, 10)
    if src_dtype_info.min != 0:
        array = np.concatenate((array, np.geomspace(-2, src_dtype_info.min, 50, dtype=src_dtype).reshape(5, 10)))
    array = np.pad(array, (1, 1), constant_values=src_nodata)
    src_ra = RasterArray(
        array, crs=profile_100cm_float['crs'], transform=profile_100cm_float['transform'], nodata=src_nodata
    )

    # convert to dtype
    src_copy_ra = src_ra.copy()
    test_array = src_copy_ra._convert_array_dtype(dst_dtype, nodata=dst_nodata)

    # test converting did not change src_copy_ra
    assert utils.nan_equals(src_copy_ra.array, src_ra.array).all()
    assert (src_copy_ra.mask == src_ra.mask).all()

    # create rounded & clipped array in src_dtype to test against
    ref_array = array
    if np.issubdtype(dst_dtype, np.integer):
        ref_array = np.clip(np.round(ref_array), dst_dtype_info.min, dst_dtype_info.max)
    elif np.issubdtype(src_dtype, np.floating):
        # don't clip float but set out of range vals to +-inf (as np.astype does)
        ref_array[ref_array < dst_dtype_info.min] = float('-inf')
        ref_array[ref_array > dst_dtype_info.max] = float('inf')
        assert np.any(ref_array[src_ra.mask] % 1 != 0)  # check contains decimals

    assert test_array.dtype == dst_dtype
    if dst_nodata:
        test_mask = ~utils.nan_equals(test_array, dst_nodata)
        assert np.any(test_mask)
        assert (test_mask == src_ra.mask).all()
    # use approx test for case of (expected) precision loss e.g. float64->float32 or int64->float32
    assert test_array[src_ra.mask] == pytest.approx(ref_array[src_ra.mask], rel=1e-6)


def test_convert_array_dtype_error(ra_100cm_float: RasterArray):
    """ Test dtype conversion raises an error when the nodata value cannot be cast to the conversion dtype. """
    test_ra = ra_100cm_float.copy()
    with pytest.raises(ValueError) as ex:
        test_ra._convert_array_dtype('uint8', nodata=float('nan'))
    assert 'cast' in str(ex.value)
