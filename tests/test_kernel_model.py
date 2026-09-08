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

import cv2
import numpy as np
import pytest
from rasterio.enums import Resampling

from homonim import utils
from homonim.enums import Model
from homonim.errors import HomonimError
from homonim.kernel_model import RefSpaceModel, SrcSpaceModel
from homonim.raster_array import RasterArray


@pytest.mark.parametrize(
    'model, kernel_shape',
    [
        (Model.gain, (1, 1)),
        (Model.gain, (3, 3)),
        (Model.gain_blk_offset, (1, 1)),
        (Model.gain_blk_offset, (5, 5)),
        (Model.gain_offset, (5, 5)),
    ],
)
def test_ref_basic_fit(
    ra_100cm_float, ra_50cm_float, model: Model, kernel_shape: tuple[int, int]
):
    """Test that ref-space models are fitted correctly against known parameters."""
    # mask_partial is only applied in RefSpaceModel.apply(), so we just set it to
    # False here
    kernel_model = RefSpaceModel(
        model, kernel_shape, mask_partial=False, r2_inpaint_thresh=0.25
    )
    src_ra = ra_50cm_float
    ref_ra = ra_100cm_float

    param_ra = kernel_model.fit(src_ra, ref_ra.copy())
    assert param_ra.shape == ref_ra.shape
    assert param_ra.transform == ref_ra.transform
    # given that ref_ra.mask == downsample(src_ra.mask)
    assert (ref_ra.mask == param_ra.mask).all()
    # test gain ~1
    assert param_ra.array[0, param_ra.mask] == pytest.approx(1, abs=1.0e-2)
    # test offset ~0
    assert param_ra.array[1, param_ra.mask] == pytest.approx(0, abs=1.0e-2)


@pytest.mark.parametrize(
    'model, kernel_shape',
    [
        (Model.gain, (1, 1)),
        (Model.gain, (3, 3)),
        (Model.gain_blk_offset, (1, 1)),
        (Model.gain_blk_offset, (5, 5)),
        (Model.gain_offset, (5, 5)),
    ],
)
def test_src_basic_fit(
    ra_100cm_float,
    ra_50cm_float,
    model: Model,
    kernel_shape: tuple[int, int],
):
    """Test that src-space models are fitted correctly against known parameters."""
    kernel_model = SrcSpaceModel(
        model, kernel_shape, mask_partial=False, r2_inpaint_thresh=0.25
    )
    src_ra = ra_100cm_float
    ref_ra = ra_50cm_float

    param_ra = kernel_model.fit(src_ra, ref_ra)
    assert param_ra.shape == src_ra.shape
    assert param_ra.transform == src_ra.transform
    # given that src_ra.mask == upsample(ref_ra.mask)
    assert (src_ra.mask == param_ra.mask).all()
    # gain ~1
    assert param_ra.array[0, param_ra.mask] == pytest.approx(1, abs=1e-2)
    # offset ~0
    assert param_ra.array[1, param_ra.mask] == pytest.approx(0, abs=1e-2)


def test_ref_basic_apply(ra_100cm_float, ra_50cm_float):
    """Test application of known ref-space parameters."""
    kernel_model = RefSpaceModel(Model.gain_blk_offset, (5, 5), mask_partial=False)
    src_ra = ra_50cm_float

    # create test parameters == 1
    param_ra = ra_100cm_float.copy()
    param_mask = param_ra.mask
    param_ra.array = np.ones((2, *param_ra.shape), dtype=param_ra.dtype)
    param_ra.mask = param_mask

    out_ra = kernel_model.apply(src_ra, param_ra)
    assert out_ra.transform == src_ra.transform
    assert out_ra.shape == src_ra.shape
    assert (src_ra.mask == out_ra.mask).all()
    assert out_ra.array[out_ra.mask] == pytest.approx(
        src_ra.array[out_ra.mask] + 1, abs=1e-2
    )


def test_src_basic_apply(ra_100cm_float):
    """Test application of known src-space parameters."""
    kernel_model = SrcSpaceModel(Model.gain_blk_offset, (5, 5), mask_partial=False)
    src_ra = ra_100cm_float

    # create test parameters == 1
    param_ra = ra_100cm_float.copy()
    param_mask = param_ra.mask
    param_ra.array = np.ones((2, *param_ra.shape), dtype=param_ra.dtype)
    param_ra.mask = param_mask

    out_ra = kernel_model.apply(src_ra, param_ra)
    assert out_ra.transform == src_ra.transform
    assert out_ra.shape == src_ra.shape
    assert (src_ra.mask == out_ra.mask).all()
    assert out_ra.array[out_ra.mask] == pytest.approx(
        src_ra.array[out_ra.mask] + 1, abs=1e-2
    )


@pytest.mark.parametrize(
    'model, find_r2',
    [
        (Model.gain, True),
        (Model.gain, False),
        (Model.gain_offset, True),
        (Model.gain_offset, False),
        (Model.gain_blk_offset, True),
        (Model.gain_blk_offset, False),
    ],
)
def test_find_r2(ra_100cm_float, ra_50cm_float, model: Model, find_r2: bool):
    """Test the R2 band is added correctly with find_r2==True."""
    kernel_model = RefSpaceModel(model, (5, 5), find_r2=find_r2, r2_inpaint_thresh=None)
    src_ra = ra_50cm_float
    ref_ra = ra_100cm_float
    param_ra = kernel_model.fit(src_ra, ref_ra)
    if find_r2:
        assert param_ra.count == 3
        assert np.nanmax(param_ra.array[2]) <= 1
    else:
        assert param_ra.count == 2


@pytest.mark.parametrize('kernel_shape', [(5, 5), (5, 7), (9, 9)])
def test_low_r2_inpainting(ra_50cm_float, kernel_shape: tuple[int, int]):
    """Test low R2 value in-painting."""
    # make src and ref the same so we have known parameters == 1
    src_ra = ra_50cm_float
    ref_ra = ra_50cm_float.copy()

    # find indices and masks to set a ref pixel to -100, so that r2 values are low
    # for all kernels covering that pixel
    low_r2_loc = np.floor(np.array(ref_ra.shape) / 2).astype('int')
    low_r2_ul = (low_r2_loc - np.floor(np.array(kernel_shape) / 2)).astype('int')
    low_r2_mask = np.zeros_like(ref_ra.mask).astype('bool')
    low_r2_mask[
        low_r2_ul[0] : low_r2_ul[0] + kernel_shape[0],
        low_r2_ul[1] : low_r2_ul[1] + kernel_shape[1],
    ] = True
    ref_ra.array[low_r2_loc[0], low_r2_loc[1]] = -100

    # fit models with and without inpainting
    no_inpaint_kernel_model = RefSpaceModel(
        Model.gain_offset,
        kernel_shape=kernel_shape,
        r2_inpaint_thresh=None,
        mask_partial=False,
        find_r2=True,
    )
    no_inpaint_param_ra = no_inpaint_kernel_model.fit(src_ra, ref_ra.copy())

    r2_inpaint_thresh = 0.5
    inpaint_kernel_model = RefSpaceModel(
        Model.gain_offset,
        kernel_shape=kernel_shape,
        r2_inpaint_thresh=r2_inpaint_thresh,
        mask_partial=False,
    )
    inpaint_param_ra = inpaint_kernel_model.fit(src_ra, ref_ra.copy())

    # test r2 values
    for param_ra in [no_inpaint_param_ra, inpaint_param_ra]:
        assert param_ra.array[2, ~low_r2_mask & ref_ra.mask] == pytest.approx(
            1, abs=1.0e-3
        )
        assert (param_ra.array[2, low_r2_mask] < r2_inpaint_thresh).all()

    # test r2 inpainting has improved parameters
    assert no_inpaint_param_ra.array[1, no_inpaint_param_ra.mask] != pytest.approx(
        0, abs=1.0e-1
    )
    assert inpaint_param_ra.array[1, inpaint_param_ra.mask] == pytest.approx(
        0, abs=1.0e-1
    )
    assert (
        inpaint_param_ra.array[0, inpaint_param_ra.mask].var()
        < no_inpaint_param_ra.array[0, no_inpaint_param_ra.mask].var()
    )


def test_nan_inpainting(ra_100cm_float: RasterArray):
    """Test nan gain and R2 are inpainted."""
    # create a source array that results in a a divide by 0 gain nan at gain_ij
    src_ra = ra_100cm_float.copy()
    gain_ij = (4, 3)
    src_slices = tuple([slice(coord - 1, coord + 2) for coord in gain_ij])
    src_ra.array[src_slices] = 1

    # create a reference array that results in a divide by 0 R2 nan at r2_ij
    ref_ra = ra_100cm_float.copy()
    r2_ij = (ref_ra.shape[0] - 4 - 1, ref_ra.shape[1] - 3 - 1)
    ref_slices = tuple([slice(coord - 1, coord + 2) for coord in r2_ij])
    ref_ra.array[ref_slices] = 1

    # fit models without and with inpainting
    no_inpaint_model = RefSpaceModel(
        Model.gain_offset,
        kernel_shape=(3, 3),
        r2_inpaint_thresh=None,
        mask_partial=False,
        find_r2=True,
    )
    no_inpaint_ra = no_inpaint_model.fit(src_ra, ref_ra)

    inpaint_model = RefSpaceModel(
        Model.gain_offset,
        kernel_shape=(3, 3),
        r2_inpaint_thresh=0.25,
        mask_partial=False,
        find_r2=True,
    )
    inpaint_ra = inpaint_model.fit(src_ra, ref_ra)

    # test inpainting of gain nan
    assert np.all(np.isnan(no_inpaint_ra.array[:, gain_ij[0], gain_ij[1]]))
    assert np.all(
        np.isnan(inpaint_ra.array[:, gain_ij[0], gain_ij[1]]) == [False, False, True]
    )

    # test inpainting of R2 nan
    slices = np.s_[2, r2_ij[0], r2_ij[1]]
    assert np.isnan(no_inpaint_ra.array[slices])
    assert np.isnan(inpaint_ra.array[slices])
    slices = np.s_[:2, r2_ij[0], r2_ij[1]]
    assert not np.any(
        utils.nan_equals(no_inpaint_ra.array[slices], inpaint_ra.array[slices])
    )


@pytest.mark.parametrize(
    'kernel_shape, mask_partial',
    [((1, 1), False), ((1, 1), True), ((3, 3), True), ((3, 5), True), ((5, 5), True)],
)
def test_ref_masking(
    ra_100cm_float, ra_50cm_float, kernel_shape: tuple[int, int], mask_partial: bool
):
    """Test ref-space partial masking."""
    kernel_model = RefSpaceModel(
        Model.gain_blk_offset, kernel_shape, mask_partial=mask_partial
    )
    src_ra = ra_50cm_float.copy()

    # create test parameters == 1
    param_ra = ra_100cm_float.copy()
    param_mask = param_ra.mask
    param_ra.array = np.ones((2, *param_ra.shape), dtype=param_ra.dtype)
    param_ra.mask = param_mask

    out_ra = kernel_model.apply(src_ra, param_ra)
    if not mask_partial:
        assert (src_ra.mask == out_ra.mask).all()
    else:
        # test output mask is contained by and smaller than src mask
        assert src_ra.mask.sum() > out_ra.mask.sum()
        assert src_ra.mask[out_ra.mask].all()

        # find and test against the expected mask
        # this test depends on RasterArray.reproject which is a compromise to allow
        # thorough testing here
        mask_ra = src_ra.mask_ra.reproject(
            **param_ra.proj_profile, nodata=None, resampling=Resampling.average
        )
        # ref pixels fully covered by src
        mask = (mask_ra.array >= 1).astype('uint8', copy=False)
        mask_ra.array = cv2.erode(mask, np.ones(np.add(kernel_shape, 2)))
        test_mask_ra = mask_ra.reproject(
            **src_ra.proj_profile, nodata=None, resampling=Resampling.nearest
        )
        assert (test_mask_ra.array == out_ra.mask).all()


@pytest.mark.parametrize(
    'kernel_shape, mask_partial',
    [((1, 1), False), ((1, 1), True), ((3, 3), True), ((3, 5), True), ((5, 5), True)],
)
def test_src_masking(
    ra_100cm_float,
    ra_50cm_float,
    kernel_shape: tuple[int, int],
    mask_partial: bool,
):
    """Test src-space partial masking."""
    kernel_model = SrcSpaceModel(
        Model.gain_blk_offset, kernel_shape, mask_partial=mask_partial
    )
    src_ra = ra_100cm_float
    ref_ra = ra_50cm_float

    # create test parameters
    param_ra = kernel_model.fit(src_ra, ref_ra)

    if not mask_partial:
        assert (src_ra.mask == param_ra.mask).all()
    else:
        # test output mask is contained by and smaller than src mask
        assert src_ra.mask.sum() > param_ra.mask.sum()
        assert src_ra.mask[param_ra.mask].all()
        # find and test against the expected mask
        test_mask = cv2.erode(
            src_ra.mask.astype('uint8'), np.ones(np.array(kernel_shape) + 2)
        )
        assert (test_mask == param_ra.mask).all()


def test_ref_force_proc_crs(ra_100cm_float, ra_50cm_float):
    """Test fitting models in ref space with low res src."""
    kernel_model = RefSpaceModel(Model.gain_blk_offset, (5, 5), mask_partial=False)
    src_ra = ra_100cm_float
    ref_ra = ra_50cm_float
    param_ra = kernel_model.fit(src_ra, ref_ra.copy())
    out_ra = kernel_model.apply(src_ra, param_ra)
    assert src_ra.array[src_ra.mask] == pytest.approx(out_ra.array[out_ra.mask], abs=2)


def test_src_force_proc_crs(ra_100cm_float, ra_50cm_float):
    """Test fitting models in src space with low res ref."""
    kernel_model = SrcSpaceModel(Model.gain_blk_offset, (5, 5), mask_partial=False)
    src_ra = ra_50cm_float
    ref_ra = ra_100cm_float
    param_ra = kernel_model.fit(src_ra, ref_ra)
    out_ra = kernel_model.apply(src_ra, param_ra)
    assert src_ra.array[src_ra.mask] == pytest.approx(out_ra.array[out_ra.mask], abs=2)


@pytest.mark.parametrize(
    'model, kernel_shape',
    [
        (Model.gain, (0, 0)),
        (Model.gain_blk_offset, (0, 0)),
        (Model.gain_offset, (4, 5)),
    ],
)
def test_kernel_shape_exception(model: Model, kernel_shape: tuple[int, int]):
    """Test bad kernel shape / model combination raises an exception."""
    with pytest.raises(HomonimError, match='kernel_shape'):
        _ = RefSpaceModel(model=model, kernel_shape=kernel_shape)


def test_r2_array_defaults(ra_100cm_float):
    """Test KernelModel._r2_array() with default (None) parameter values."""
    kernel_model = RefSpaceModel(
        Model.gain_offset,
        (5, 5),
        mask_partial=False,
        r2_inpaint_thresh=None,
        find_r2=False,
    )
    src_ra = ra_100cm_float
    ref_ra = ra_100cm_float
    param_ra = kernel_model.fit(src_ra, ref_ra.copy())
    r2_array = kernel_model._r2_array(ref_ra.array, ref_ra.array, param_ra.array)
    assert r2_array[~np.isnan(r2_array)] == pytest.approx(1, abs=1.0e-3)


def test_config():
    """Test KernelModel is configured correctly."""
    config = dict(
        r2_inpaint_thresh=0.1,
        mask_partial=True,
        downsampling=Resampling.bilinear,
        upsampling=Resampling.nearest,
    )
    kernel_model = RefSpaceModel(Model.gain, (5, 5), find_r2=True, **config)
    for key in config.keys():
        attr_name = '_' + key
        assert hasattr(kernel_model, attr_name)
        assert getattr(kernel_model, attr_name) == config[key]


@pytest.mark.parametrize('model', [Model.gain, Model.gain_offset])
def test_gain_divide_by_0(model: Model, ra_100cm_float: RasterArray):
    """Test that gain divide by 0 results in all parameters being nan."""
    kernel_model = RefSpaceModel(
        model,
        (3, 3),
        mask_partial=False,
        r2_inpaint_thresh=None,
        find_r2=True,
    )

    # create a source array that results in divide by 0 at src_ij
    src_ra = ra_100cm_float.copy()
    src_ij = (4, 3)
    src_slices = tuple([slice(coord - 1, coord + 2) for coord in src_ij])
    src_ra.array[src_slices] = 0

    # create a reference array that results in divide by 0 at ref_ij
    ref_ra = ra_100cm_float.copy()
    ref_ij = (ref_ra.shape[0] - 4 - 1, ref_ra.shape[0] - 3 - 1)
    ref_slices = tuple([slice(coord - 1, coord + 2) for coord in ref_ij])
    ref_ra.array[ref_slices] = 0

    param_ra = kernel_model.fit(src_ra, ref_ra)

    # test all params are nan at src_ij
    assert np.all(np.isnan(param_ra.array[:, src_ij[0], src_ij[1]]))


@pytest.mark.parametrize('model', Model)
def test_r2_divide_by_0(model: Model, ra_100cm_float: RasterArray):
    """Test that R2 divide by 0 results in nan R2."""
    kernel_model = RefSpaceModel(
        model,
        (3, 3),
        mask_partial=False,
        r2_inpaint_thresh=None,
        find_r2=True,
    )

    # create a reference array that results in divide by 0 at ref_ij
    ref_ra = ra_100cm_float.copy()
    ref_ij = (4, 3)
    ref_slices = tuple([slice(coord - 1, coord + 2) for coord in ref_ij])
    ref_ra.array[ref_slices] = 0

    param_ra = kernel_model.fit(ra_100cm_float, ref_ra)

    # test gain and offset are 0, and R2 is nan at ref_ij
    assert np.all(
        utils.nan_equals(param_ra.array[:, ref_ij[0], ref_ij[1]], [0, 0, np.nan])
    )
