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

from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import MaskFlags, Resampling
from rasterio.transform import Affine, array_bounds, from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from homonim import utils
from homonim.errors import HomonimError, ImageFormatError
from homonim.raster_array import RasterArray


def test_read_only_properties(array_byte, profile_byte):
    """Test RasterArray read-only properties."""
    ra = RasterArray(
        array_byte,
        profile_byte['crs'],
        profile_byte['transform'],
        nodata=profile_byte['nodata'],
    )
    for name in ['crs', 'transform', 'width', 'height', 'count', 'dtype']:
        assert getattr(ra, name) == profile_byte[name]
    assert ra.shape == array_byte.shape
    assert ra.res == (ra.transform.a, -ra.transform.e)
    profile = profile_byte.copy()
    profile.pop('driver')
    assert ra.profile == profile
    assert ra.bounds == pytest.approx(array_bounds(*ra.shape, ra.transform))


def test_array_property(array_byte, profile_byte):
    """Test getting & setting the array property."""
    ra = RasterArray(
        array_byte,
        profile_byte['crs'],
        profile_byte['transform'],
        nodata=profile_byte['nodata'],
    )
    assert (ra.array == array_byte).all()

    # test set with 1 band
    array = array_byte / 2
    ra.array = array
    assert (ra.array == array).all()

    # test set with >1 bands
    array = np.stack((array, array), axis=0)
    ra.array = array
    assert (ra.array == array).all()
    assert ra.count == array.shape[0]


def test_nodata_mask(ra_byte):
    """Test getting and setting nodata, and its interaction with mask()."""
    mask = ra_byte.mask()
    nodata = 254
    ra_byte.nodata = nodata
    assert ra_byte.nodata == nodata
    # test mask unchanged after setting nodata
    assert (ra_byte.mask() == mask).all()

    # test setting the array updates the mask
    idx = slice(0, 10)
    mask[idx] = False
    ra_byte.array[idx] = ra_byte.nodata
    assert (ra_byte.mask() == mask).all()

    # test the mask after setting a multiband array
    ra_byte.array = np.stack((ra_byte.array,) * 2, axis=0)
    assert (ra_byte.mask() == mask).all()

    # test removing nodata
    ra_byte.nodata = None
    assert ra_byte.mask().all()


def test_array_set_shape(ra_byte):
    """Test setting array with different (row, col) dimensions raises an error."""
    with pytest.raises(ValueError, match='dimensions'):
        ra_byte.array = ra_byte.array.reshape(-1, 1)


def test_from_rio_dataset(file_byte: Path, array_byte: np.ndarray):
    """Test from_rio_dataset()."""
    with rio.open(file_byte, 'r') as ds:
        # test defaults
        ra = RasterArray.from_rio_dataset(ds)
        for name in ['crs', 'transform', 'shape', 'count', 'nodata']:
            assert getattr(ra, name) == getattr(ds, name)
        assert ra.dtype == RasterArray.default_dtype
        assert (ra.array == array_byte).all()

        # test parameters
        indexes = [1, 1]  # test >1 index with a single band file
        win = Window(0, 1, ds.width, ds.height - 2)
        nodata = 65535
        dtype = 'uint16'
        array = np.stack((array_byte,) * len(indexes), axis=0).astype(dtype)
        array = array[(..., *win.toslices())]
        array[array == ds.nodata] = nodata
        ra = RasterArray.from_rio_dataset(
            ds, indexes=indexes, window=win, nodata=nodata, dtype=dtype
        )

        assert ra.transform == ds.window_transform(win)
        assert ra.nodata == nodata
        assert ra.dtype == dtype
        assert (ra.array == array).all()


def test_from_rio_dataset_boundless(file_byte: Path, array_byte: np.ndarray):
    """Test from_rio_dataset() with a boundless window."""
    with rio.open(file_byte, 'r') as ds:
        win = Window(-1, -1, ds.width + 2, ds.height + 2)
        ra = RasterArray.from_rio_dataset(ds, window=win)
        assert ra.shape == (win.height, win.width)

        # test array contents and transform
        bounded_win = Window(1, 1, ds.width, ds.height)
        assert (ra.array[bounded_win.toslices()] == array_byte).all()
        assert ra.transform == ds.window_transform(win)
        # test array is nodata outside of dataset bounds
        mask = ra.mask()
        assert mask[bounded_win.toslices()].sum() == mask.sum()


@pytest.mark.parametrize('file, count', [('file_masked', 1), ('file_rgba', 3)])
def test_from_rio_dataset_masked(file: str, count: int, request: pytest.FixtureRequest):
    """Test from_rio_dataset() with internally and alpha masked datasets."""
    file: Path = request.getfixturevalue(file)
    with rio.open(file, 'r') as ds:
        ds_mask = ds.dataset_mask().astype('bool', copy=False)
        ra = RasterArray.from_rio_dataset(ds)
    assert ra.count == count
    assert np.isnan(ra.nodata)
    assert (ra.mask() == ds_mask).all()


def test_crop_to_window(ra_byte):
    """Test _crop_to_window()."""
    # test valid window
    win = Window(1, 1, ra_byte.width - 2, ra_byte.height - 2)
    crop_ra = ra_byte._crop_to_window(win)
    assert crop_ra.bounds == pytest.approx(ra_byte.window_bounds(win))
    assert (crop_ra.array == ra_byte.array[win.toslices()]).all()

    # test invalid windows
    with pytest.raises(ValueError, match='window'):
        ra_byte._crop_to_window(Window(-1, -1, ra_byte.width, ra_byte.height))
    with pytest.raises(ValueError, match='window'):
        ra_byte._crop_to_window(Window(1, 1, ra_byte.width, ra_byte.height))


def test_to_rio_dataset(ra_byte, tmp_path: Path):
    """Test to_rio_dataset()."""
    # dataset with same dtype, nodata & count as the RasterArray
    ds_file = tmp_path.joinpath('temp.tif')
    with rio.open(ds_file, 'w', driver='GTiff', **ra_byte.profile) as ds:
        ra_byte.to_rio_dataset(ds)

    with rio.open(ds_file, 'r') as ds:
        array = ds.read(indexes=1)
    assert (array == ra_byte.array).all()

    # dataset with different dtype, nodata & count to the RasterArray
    profile = ra_byte.profile.copy()
    profile.update(dtype='uint16', nodata=65535, count=2)
    indexes = 2
    exp_array = ra_byte.array.astype(profile['dtype'])
    exp_array[..., exp_array == ra_byte.nodata] = profile['nodata']
    with rio.open(ds_file, 'w', driver='GTiff', **profile) as ds:
        ra_byte.to_rio_dataset(ds, indexes=indexes)

    with rio.open(ds_file, 'r') as ds:
        array = ds.read(indexes=indexes)
    assert (array == exp_array).all()


def test_to_rio_dataset_nodata_none(ra_byte, tmp_path: Path):
    """Test to_rio_dataset() with nodata=None writes an internal mask."""
    ds_file = tmp_path.joinpath('temp.tif')
    profile = ra_byte.profile
    profile.update(nodata=None)
    with rio.open(ds_file, 'w', driver='GTiff', **profile) as ds:
        ra_byte.to_rio_dataset(ds)

    with rio.open(ds_file, 'r') as ds:
        assert ds.nodata is None
        assert ds.mask_flag_enums[0] == [MaskFlags.per_dataset]
        mask = ds.dataset_mask().astype('bool')
        array = ds.read(indexes=1)

    assert (mask == ra_byte.mask()).all()
    assert (array[mask] == ra_byte.array[mask]).all()


def test_to_rio_dataset_crop(ra_rgb_byte, tmp_path: Path):
    """Test to_rio_dataset() where the dataset & RasterArray bounds differ."""
    ds_file = tmp_path.joinpath('temp.tif')
    indexes = [1, 2, 3]
    # write a cropped RasterArray to a full extent dataset
    crop_win = Window(1, 1, ra_rgb_byte.width - 2, ra_rgb_byte.height - 2)
    crop_ra = ra_rgb_byte._crop_to_window(crop_win)
    with rio.open(ds_file, 'w', driver='GTiff', **ra_rgb_byte.profile) as ds:
        crop_ra.to_rio_dataset(ds, indexes=indexes, window=crop_win)
    with rio.open(ds_file, 'r') as ds:
        test_array = ds.read(indexes=indexes)
    assert (test_array[(..., *crop_win.toslices())] == crop_ra.array).all()

    # write a full extent RasterArray to a cropped dataset
    with rio.open(ds_file, 'w', driver='GTiff', **crop_ra.profile) as ds:
        ra_rgb_byte.to_rio_dataset(ds, indexes=indexes)
    with rio.open(ds_file, 'r') as ds:
        test_array = ds.read(indexes=indexes)
    assert (test_array == ra_rgb_byte.array[(..., *crop_win.toslices())]).all()


def test_to_rio_dataset_error(ra_rgb_byte, tmp_path: Path):
    """Test to_rio_dataset() error conditions."""
    ds_file = tmp_path.joinpath('temp.tif')
    # len(indexes) > number of dataset bands
    with rio.open(ds_file, 'w', driver='GTiff', **ra_rgb_byte.profile) as ds:
        with pytest.raises(ValueError, match='indexes'):
            ra_rgb_byte.to_rio_dataset(ds, indexes=[1] * (ds.count + 1))

    # dataset and RasterArray have different CRSs
    profile = ra_rgb_byte.profile
    profile.update(crs=CRS.from_epsg(4326))
    with rio.open(ds_file, 'w', driver='GTiff', **profile) as ds:
        with pytest.raises(ImageFormatError, match='CRS'):
            ra_rgb_byte.to_rio_dataset(ds)

    # dataset and RasterArray are not on the same pixel grid
    profile = ra_rgb_byte.profile
    profile.update(transform=Affine.identity() * Affine.translation(0.5, 0.5))
    with rio.open(ds_file, 'w', driver='GTiff', **profile) as ds:
        with pytest.raises(ImageFormatError, match='pixel grid'):
            ra_rgb_byte.to_rio_dataset(ds)

    profile.update(transform=Affine.identity() * Affine.scale(0.5, 0.5))
    with rio.open(ds_file, 'w', driver='GTiff', **profile) as ds:
        with pytest.raises(ImageFormatError, match='pixel grid'):
            ra_rgb_byte.to_rio_dataset(ds)


def test_reprojection(ra_rgb_byte):
    """Test RasterArray re-projection."""
    # reproject to WGS84 with default parameters, assuming ra_rgb_byte is North up
    to_crs = CRS.from_epsg(4326)
    reprj_ra = ra_rgb_byte.reproject(crs=to_crs, resampling=Resampling.nearest)
    assert reprj_ra.crs == to_crs
    abs_diff = np.abs(reprj_ra.array - ra_rgb_byte.array)
    assert np.nanmean(abs_diff) == pytest.approx(0, abs=0.1)
    assert (reprj_ra.mask() == ra_rgb_byte.mask()).all()

    # reproject with rescaling to WGS84 using a specified transform & shape
    to_bounds = transform_bounds(ra_rgb_byte.crs, to_crs, *ra_rgb_byte.bounds)
    to_shape = tuple(np.array(ra_rgb_byte.shape) * 2)
    to_transform = from_bounds(*to_bounds, *to_shape[::-1])
    reprj_ra = ra_rgb_byte.reproject(
        crs=to_crs,
        transform=to_transform,
        shape=to_shape,
        resampling=Resampling.bilinear,
    )
    assert reprj_ra.crs == to_crs
    assert reprj_ra.transform == to_transform
    assert reprj_ra.shape == to_shape
    assert reprj_ra.bounds == pytest.approx(to_bounds, abs=1e-9)
    assert reprj_ra.array[:, reprj_ra.mask()].mean() == pytest.approx(
        ra_rgb_byte.array[:, ra_rgb_byte.mask()].mean(), abs=0.1
    )


@pytest.mark.parametrize(
    'src_dtype, src_nodata, dst_dtype, dst_nodata',
    [
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
        # nodata unchanged
        ('float32', float('nan'), 'float32', None),
    ],
)
def test_convert_array_dtype(
    profile_100cm_float: dict,
    src_dtype: str,
    src_nodata: float,
    dst_dtype: str,
    dst_nodata: float,
):
    """Test dtype conversion with combinations covering rounding, clipping (with and
    w/o type promotion) and re-masking.
    """
    src_info = (
        np.iinfo(src_dtype)
        if np.issubdtype(src_dtype, np.integer)
        else np.finfo(src_dtype)
    )
    dst_info = (
        np.iinfo(dst_dtype)
        if np.issubdtype(dst_dtype, np.integer)
        else np.finfo(dst_dtype)
    )

    # create array that spans the src_dtype range, includes decimals, excludes -1..1
    # (to allow nodata == +-1), and is padded with nodata
    array = np.geomspace(2, src_info.max, 50, dtype=src_dtype).reshape(5, 10)
    if src_info.min != 0:
        array = np.concatenate(
            (np.geomspace(-2, src_info.min, 50, dtype=src_dtype).reshape(5, 10), array)
        )
    array = np.pad(array, (1, 1), constant_values=src_nodata)
    src_ra = RasterArray(
        array,
        crs=profile_100cm_float['crs'],
        transform=profile_100cm_float['transform'],
        nodata=src_nodata,
    )

    # convert to dtype
    src_copy_ra = src_ra.copy()
    test_array = src_copy_ra._convert_array_dtype(dst_dtype, nodata=dst_nodata)

    # test converting did not change src_copy_ra
    assert utils.nan_equals(src_copy_ra.array, src_ra.array).all()

    # create rounded & clipped array in src_dtype to test against
    src_mask = src_ra.mask()
    ref_array = array
    if np.issubdtype(dst_dtype, np.integer):
        ref_array = np.clip(np.round(ref_array), dst_info.min, dst_info.max)
    elif np.issubdtype(src_dtype, np.floating):
        # don't clip float but set out of range vals to +-inf (as np.astype does)
        ref_array[ref_array < dst_info.min] = float('-inf')
        ref_array[ref_array > dst_info.max] = float('inf')
        assert np.any(ref_array[src_mask] % 1 != 0)  # check contains decimals

    assert test_array.dtype == dst_dtype
    if dst_nodata:
        test_mask = ~utils.nan_equals(test_array, dst_nodata)
        assert np.any(test_mask)
        assert (test_mask == src_mask).all()
    # use approx test for case of (expected) precision loss e.g. float64->float32 or
    # int64->float32
    assert test_array[src_mask] == pytest.approx(ref_array[src_mask], rel=1e-6)


def test_convert_array_dtype_error(ra_100cm_float: RasterArray):
    """Test dtype conversion raises an error when the nodata value cannot be cast to
    the conversion dtype.
    """
    test_ra = ra_100cm_float.copy()
    with pytest.raises(HomonimError, match='cast') as ex:
        test_ra._convert_array_dtype('uint8', nodata=float('nan'))
