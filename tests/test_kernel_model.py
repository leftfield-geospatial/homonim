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

from typing import Tuple

import cv2
import numpy as np
import pytest
from rasterio.enums import Resampling
from rasterio.transform import Affine

from homonim.enums import Method
from homonim.kernel_model import SrcSpaceModel, RefSpaceModel
from homonim.raster_array import RasterArray


@pytest.fixture
def high_res_align_float_ra(float_ra):
    """
    A higher resolution version of float_ra.
    Aligned with the float_ra pixel grid, so that re-projection back to float_ra space will give the float_ra
    mask, and ~data (resampling method dependent).
    """
    scale = 1 / 2  # resolution scaling
    # pad scaled image with a border of 1 float_ra pixel
    shape = tuple(np.ceil(np.array(float_ra.shape) / scale  + (2 / scale)).astype('int'))
    transform = float_ra.transform * Affine.translation(-1, -1) * Affine.scale(scale)
    return float_ra.reproject(transform=transform, shape=shape, resampling=Resampling.nearest)


@pytest.fixture
def high_res_unalign_float_ra(float_ra):
    """
    A higher resolution version of float_ra, but on a different pixel grid.
    """
    scale = 0.45  # resolution scaling
    # pad scaled image with a border of ~1 float_ra pixel
    shape = tuple(np.ceil(np.array(float_ra.shape) / scale + (2 / scale)).astype('int'))
    transform = float_ra.transform * Affine.translation(-1, -1) * Affine.scale(scale)
    return float_ra.reproject(transform=transform, shape=shape, resampling=Resampling.bilinear)


@pytest.mark.parametrize('method, kernel_shape', [
    (Method.gain, (1, 1)),
    (Method.gain, (3, 3)),
    (Method.gain_blk_offset, (1, 1)),
    (Method.gain_blk_offset, (5, 5)),
    (Method.gain_offset, (5, 5)),
    (Method.gain_offset, (5, 5)),
])
def test_ref_basic_fit(float_ra: RasterArray, high_res_align_float_ra: RasterArray, method: Method,
                       kernel_shape: Tuple[int, int]):
    """Test that models are fitted correctly using known parameters"""

    # mask_partial is only applied in RefSpaceModel.apply(), so we just set it False here
    kernel_model = RefSpaceModel(method, kernel_shape, mask_partial=False, r2_inpaint_thresh=0.25)
    src_ra = high_res_align_float_ra
    ref_ra = float_ra

    param_ra = kernel_model.fit(ref_ra.copy(), src_ra)
    assert (param_ra.shape == ref_ra.shape)
    assert (param_ra.transform == ref_ra.transform)
    assert (ref_ra.mask == param_ra.mask).all()
    assert param_ra.array[0, param_ra.mask] == pytest.approx(1, abs=1.e-2)
    assert param_ra.array[1, param_ra.mask] == pytest.approx(0, abs=1.e-2)


@pytest.mark.parametrize('method, kernel_shape', [
    (Method.gain, (1, 1)),
    (Method.gain, (3, 3)),
    (Method.gain_blk_offset, (1, 1)),
    (Method.gain_blk_offset, (5, 5)),
    (Method.gain_offset, (5, 5)),
    (Method.gain_offset, (5, 5)),
])
def test_src_basic_fit(float_ra: RasterArray, high_res_align_float_ra: RasterArray, method: Method,
                       kernel_shape: Tuple[int, int]):
    """Test models are fitted correctly in src space with known parameters"""
    kernel_model = SrcSpaceModel(method, kernel_shape, mask_partial=False, r2_inpaint_thresh=0.25)
    src_ra = float_ra
    ref_ra = high_res_align_float_ra

    param_ra = kernel_model.fit(ref_ra, src_ra)
    assert (param_ra.shape == src_ra.shape)
    assert (param_ra.transform == src_ra.transform)
    assert (src_ra.mask == param_ra.mask).all()
    assert param_ra.array[0, param_ra.mask] == pytest.approx(1, abs=1e-2)
    assert param_ra.array[1, param_ra.mask] == pytest.approx(0, abs=1e-2)


def test_ref_basic_apply(float_ra: RasterArray, high_res_align_float_ra: RasterArray):
    """Test application of known ref space parameters"""

    kernel_model = RefSpaceModel(Method.gain_blk_offset, (5, 5), mask_partial=False)
    src_ra = high_res_align_float_ra

    # create test parameters
    param_ra = float_ra.copy()
    param_mask = param_ra.mask
    param_ra.array = np.ones((2, *param_ra.shape), dtype=param_ra.dtype)
    param_ra.mask = param_mask

    out_ra = kernel_model.apply(src_ra, param_ra)
    assert (out_ra.transform == src_ra.transform)
    assert (out_ra.shape == src_ra.shape)
    assert (src_ra.mask == out_ra.mask).all()
    assert (out_ra.array[out_ra.mask] == pytest.approx(src_ra.array[out_ra.mask] + 1, abs=1e-2))


def test_src_basic_apply(float_ra: RasterArray):
    """Test application of known src space parameters"""

    kernel_model = SrcSpaceModel(Method.gain_blk_offset, (5, 5), mask_partial=False)
    src_ra = float_ra

    # create test parameters
    param_ra = float_ra.copy()
    param_mask = param_ra.mask
    param_ra.array = np.ones((2, *param_ra.shape), dtype=param_ra.dtype)
    param_ra.mask = param_mask

    out_ra = kernel_model.apply(src_ra, param_ra)
    assert (out_ra.transform == src_ra.transform)
    assert (out_ra.shape == src_ra.shape)
    assert (src_ra.mask == out_ra.mask).all()
    assert (out_ra.array[out_ra.mask] == pytest.approx(src_ra.array[out_ra.mask] + 1, abs=1e-2))


@pytest.mark.parametrize('kernel_shape, mask_partial', [
    ((1, 1), False),
    ((1, 1), True),
    ((3, 3), True),
    ((3, 7), True),
    ((5, 5), True),
])
def test_ref_masking(float_ra, high_res_align_float_ra, kernel_shape: Tuple[int, int],
                     mask_partial: bool):
    kernel_model = RefSpaceModel(Method.gain_blk_offset, kernel_shape, mask_partial=mask_partial)
    src_ra = high_res_align_float_ra.copy()

    # create test parameters
    param_ra = float_ra.copy()
    param_mask = param_ra.mask
    param_ra.array = np.ones((2, *param_ra.shape), dtype=param_ra.dtype)
    param_ra.mask = param_mask

    out_ra = kernel_model.apply(src_ra, param_ra)
    if not mask_partial:
        assert (src_ra.mask == out_ra.mask).all()
    else:
        assert (src_ra.mask.sum() > out_ra.mask.sum())
        assert src_ra.mask[out_ra.mask].all()

        # find and test against the expected mask
        mask_ra = src_ra.mask_ra.reproject(**param_ra.proj_profile, nodata=None, resampling=Resampling.average)
        mask = (mask_ra.array >= 1).astype('uint8', copy=False)  # ref pixels fully covered by src
        mask_ra.array = cv2.erode(mask, np.ones(np.add(kernel_shape, 2)))
        test_mask_ra = mask_ra.reproject(**src_ra.proj_profile, nodata=0, resampling=Resampling.nearest)
        assert (test_mask_ra.array == out_ra.mask).all()


@pytest.mark.parametrize('kernel_shape, mask_partial', [
    ((1, 1), False),
    ((1, 1), True),
    ((3, 3), True),
    ((3, 7), True),
    ((5, 5), True),
])
def test_src_masking(float_ra, high_res_align_float_ra, high_res_unalign_float_ra, kernel_shape: Tuple[int, int],
                     mask_partial: bool):
    kernel_model = SrcSpaceModel(Method.gain_blk_offset, kernel_shape, mask_partial=mask_partial)
    src_ra = float_ra
    ref_ra = high_res_align_float_ra

    # create test parameters
    param_ra = kernel_model.fit(ref_ra, src_ra)

    if not mask_partial:
        assert (src_ra.mask == param_ra.mask).all()
    else:
        assert (src_ra.mask.sum() > param_ra.mask.sum())
        assert src_ra.mask[param_ra.mask].all()
        test_mask = cv2.erode(src_ra.mask.astype('uint8'), np.ones(np.array(kernel_shape) + 2))
        assert (test_mask == param_ra.mask).all()

# TODO:
# - different src and ref masks (check param mask is as expected)
# - src and ref not aligned on same grid (check we don't lose data, and that params are close to expected similar to above)
#    actually this is not really KernelModel's problem, more relevant to RasterPair
# - r2 inpainting (above e.g. has r2~1 so does not do inpainting, not sure really how to test this... make artifical bad r2 area)
# - mask-partial works as expected for src and ref space (if possible check explicity with synthetic data)
# - separate fit and apply tests if possible.  apply does mask_partial in for ref-space model.  fit does it for src space model.
# - make fixtures for what make sense to make fixtures of, the way we make ref and src above is redundant, make one high res and one low res
# - test proc_crs being forced to opp of its auto val
