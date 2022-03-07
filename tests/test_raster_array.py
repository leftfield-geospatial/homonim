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
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

from homonim.errors import ImageProfileError, ImageFormatError
from homonim.raster_array import RasterArray


def test_read_only_properties(byte_array, byte_profile):
    """Test read only properties"""
    basic_ra = RasterArray(byte_array, byte_profile['crs'], byte_profile['transform'],
                           nodata=byte_profile['nodata'])
    assert basic_ra.shape == byte_array.shape
    assert basic_ra.width == byte_array.shape[-1]
    assert basic_ra.height == byte_array.shape[-2]
    assert basic_ra.count == 1 if byte_array.ndim < 3 else byte_array.ndim
    assert basic_ra.nodata == byte_profile['nodata']
    assert basic_ra.transform == byte_profile['transform']
    assert basic_ra.dtype == byte_array.dtype
    assert basic_ra.res == (byte_profile['transform'].a, -byte_profile['transform'].e)
    assert basic_ra.crs == byte_profile['crs']


def test_array_property(byte_array, byte_profile):
    """Test array get/set"""
    byte_ra = RasterArray(byte_array, byte_profile['crs'], byte_profile['transform'],
                          nodata=byte_profile['nodata'])
    assert (byte_ra.array == byte_array).all()

    array = byte_array / 2
    byte_ra.array = array
    assert (byte_ra.array == array).all()

    array = np.stack((array, array), axis=0)
    byte_ra.array = array
    assert (byte_ra.array == array).all()
    assert byte_ra.count == array.shape[0]


def test_nodata_mask(byte_ra):
    mask = byte_ra.mask
    byte_ra.nodata = 254
    assert byte_ra.nodata == 254
    assert (byte_ra.mask == mask).all()
    assert (byte_ra.mask_ra.array == mask).all()
    assert byte_ra.mask_ra.transform == byte_ra.transform

    byte_ra.array = np.stack((byte_ra.array, byte_ra.array), axis=0)
    assert (byte_ra.mask == mask).all()

    mask[np.divide(mask.shape, 2).astype('int')] = False
    byte_ra.mask = mask
    assert (byte_ra.mask == mask).all()

    byte_ra.nodata = None
    assert byte_ra.mask.all()


def test_array_set_shape(byte_ra):
    with pytest.raises(ValueError):
        byte_ra.array = byte_ra.array.reshape(-1, 1)


def test_from_profile(byte_array, byte_profile):
    byte_ra = RasterArray.from_profile(byte_array, byte_profile)
    assert (byte_ra.array == byte_array).all()
    assert byte_ra.transform == byte_profile['transform']


def test_from_profile_noarray(byte_profile):
    byte_ra = RasterArray.from_profile(None, byte_profile)
    assert (byte_ra.array == byte_ra.nodata).all()


@pytest.mark.parametrize('missing_key', ['crs', 'transform', 'nodata', 'width', 'height', 'count', 'dtype'])
def test_from_profile_missingkey(byte_profile, missing_key):
    profile = byte_profile
    profile.pop(missing_key)
    with pytest.raises(ImageProfileError):
        RasterArray.from_profile(None, profile)


def test_from_rio_dataset(byte_file):
    with rio.open(byte_file, 'r') as ds:
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
        assert (ds_ra_boundless.transform.xoff == test_transform.xoff and
                ds_ra_boundless.transform.yoff == test_transform.yoff)


@pytest.mark.parametrize('file, count', [('masked_file', 1), ('rgba_file', 3)])
def test_from_rio_dataset_masked(file, count, request):
    file = request.getfixturevalue(file)
    with rio.open(file, 'r') as ds:
        ds_mask = ds.dataset_mask().astype('bool', copy=False)
        ra = RasterArray.from_rio_dataset(ds)
        assert ra.count == count
        assert np.isnan(ra.nodata)
        assert (ra.mask == ds_mask).all()


@pytest.mark.parametrize('pad', [[1, 1], [-1, -1]])
def test_bounded_window_slices(byte_file, pad):
    with rio.open(byte_file, 'r') as ds:
        window = Window(-pad[1], -pad[0], ds.width + 2 * pad[1], ds.height + 2 * pad[0])
        bounded_win, bounded_slices = RasterArray.bounded_window_slices(ds, window)
        assert (bounded_win.col_off == max(0, -pad[1]) and bounded_win.row_off == max(0, -pad[0]))
        assert (bounded_win.width == min(ds.width, ds.width + 2 * pad[1]) and
                bounded_win.height == min(ds.height, ds.height + 2 * pad[0]))


def test_slice_to_bounds(byte_ra: RasterArray):
    window = Window(1, 1, byte_ra.width - 2, byte_ra.height - 2)
    bounds = byte_ra.window_bounds(window)
    slice_ra = byte_ra.slice_to_bounds(*bounds)
    assert slice_ra.bounds == pytest.approx(bounds)
    assert (slice_ra.array == byte_ra.array[window.toslices()]).all()

    with pytest.raises(ValueError):
        byte_ra.slice_to_bounds(*byte_ra.window_bounds(Window(-1, -1, byte_ra.width, byte_ra.height)))


def test_to_rio_dataset(byte_ra: RasterArray, tmp_path):
    ds_filename = tmp_path.joinpath('temp.tif')
    with rio.open(ds_filename, 'w', driver='GTiff', **byte_ra.profile) as ds:
        byte_ra.to_rio_dataset(ds)
    with rio.open(ds_filename, 'r') as ds:
        test_ra = RasterArray.from_rio_dataset(ds)
    assert (test_ra.array == byte_ra.array).all()


def test_to_rio_dataset_crop(rgb_byte_ra: RasterArray, tmp_path):
    ds_filename = tmp_path.joinpath('temp.tif')
    indexes = [1, 2, 3]
    # crop the raster array and write to full dataset
    crop_window = Window(1, 1, rgb_byte_ra.width - 2, rgb_byte_ra.height - 2)
    crop_ra = rgb_byte_ra.slice_to_bounds(*rgb_byte_ra.window_bounds(crop_window))
    with rio.open(ds_filename, 'w', driver='GTiff', **rgb_byte_ra.profile) as ds:
        crop_ra.to_rio_dataset(ds, indexes=indexes, window=crop_window)
    with rio.open(ds_filename, 'r') as ds:
        test_ra = RasterArray.from_rio_dataset(ds)
    assert (test_ra.array[(np.array(indexes) - 1, *crop_window.toslices())] == crop_ra.array).all()

    # crop the dataset and write in the full raster array
    with rio.open(ds_filename, 'w', driver='GTiff', **crop_ra.profile) as ds:
        rgb_byte_ra.to_rio_dataset(ds, indexes=indexes)
    with rio.open(ds_filename, 'r') as ds:
        test_ra = RasterArray.from_rio_dataset(ds)
    assert (test_ra.array == rgb_byte_ra.array[(np.array(indexes) - 1, *crop_window.toslices())]).all()


def test_to_rio_dataset_exceptions(rgb_byte_ra: RasterArray, tmp_path):
    ds_filename = tmp_path.joinpath('temp.tif')
    with rio.open(ds_filename, 'w', driver='GTiff', **rgb_byte_ra.profile) as ds:
        with pytest.raises(ValueError):
            # window lies outside the bounds of raster array
            crop_window = Window(1, 1, rgb_byte_ra.width - 2, rgb_byte_ra.height - 2)
            crop_ra = rgb_byte_ra.slice_to_bounds(*rgb_byte_ra.window_bounds(crop_window))
            boundless_window = Window(-1, -1, rgb_byte_ra.width + 2, rgb_byte_ra.height + 2)
            crop_ra.to_rio_dataset(ds, indexes=[1, 2, 3], window=boundless_window)
        with pytest.raises(ValueError):
            # len(indexes) > number of dataset bands
            rgb_byte_ra.to_rio_dataset(ds, indexes=[1] * (ds.count + 1))
        with pytest.raises(ValueError):
            # indexes outside of valid range
            rgb_byte_ra.to_rio_dataset(ds, indexes=ds.count + 1)

    # dataset and raster array have different CRSs
    profile = rgb_byte_ra.profile
    profile.update(crs=CRS.from_epsg(4326))
    with rio.open(ds_filename, 'w', driver='GTiff', **profile) as ds:
        with pytest.raises(ImageFormatError):
            rgb_byte_ra.to_rio_dataset(ds, indexes=[1, 2, 3])

    # dataset and raster array have different resolutions
    profile = rgb_byte_ra.profile
    profile.update(transform=Affine.identity() * Affine.scale(0.5))
    with rio.open(ds_filename, 'w', driver='GTiff', **profile) as ds:
        with pytest.raises(ImageFormatError):
            rgb_byte_ra.to_rio_dataset(ds, indexes=[1, 2, 3])


def test_reprojection(rgb_byte_ra: RasterArray):
    # reproject to WGS84 with default parameters
    to_crs = CRS.from_epsg(4326)
    reproj_ra = rgb_byte_ra.reproject(crs=to_crs, resampling=Resampling.bilinear)
    assert (reproj_ra.crs == to_crs)
    assert (reproj_ra.array[:, reproj_ra.mask].mean() ==
            pytest.approx(rgb_byte_ra.array[:, rgb_byte_ra.mask].mean(), abs=.01))

    # reproject with rescaling to WGS84 using a specified transform & shape
    to_transform = Affine.identity() * Affine.scale(.5e-5)
    reproj_ra = rgb_byte_ra.reproject(crs=to_crs, transform=to_transform, shape=tuple(np.array(rgb_byte_ra.shape) * 2),
                                      resampling=Resampling.bilinear)
    assert (reproj_ra.crs == to_crs)
    assert (reproj_ra.transform == to_transform)
    assert (reproj_ra.array[:, reproj_ra.mask].mean() ==
            pytest.approx(rgb_byte_ra.array[:, rgb_byte_ra.mask].mean(), abs=.1))
