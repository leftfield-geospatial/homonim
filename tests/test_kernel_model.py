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

from homonim.kernel_model import SrcSpaceModel, RefSpaceModel, KernelModel
from homonim.raster_array import RasterArray
from homonim.enums import Method, ProcCrs
import numpy as np
import rasterio as rio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window
import pytest
import cv2
from typing import Tuple

@pytest.fixture()
def high_res_float_ra(float_ra):
    scale = 1/3          # resolution scaling
    shape = tuple(np.round(np.array(float_ra.shape)/scale).astype('int') + 2)
    transform = float_ra.transform  * Affine.scale(scale) * Affine.translation(-1, -1)
    return float_ra.reproject(transform=transform, shape=shape, resampling=Resampling.nearest)

@pytest.fixture()
def high_res_offset_float_ra(float_ra):
    scale = np.pi/10          # resolution scaling
    shape = tuple(np.round(np.array(float_ra.shape)/scale).astype('int') + 2)  # pad scaled image with 2/scale pixels
    transform = float_ra.transform  * Affine.scale(scale) * Affine.translation(-1, -1)
    return float_ra.reproject(transform=transform, shape=shape, resampling=Resampling.bilinear)

@pytest.mark.parametrize('method, kernel_shape', [
    (Method.gain, (1, 1)),
    (Method.gain, (3, 3)),
    (Method.gain_blk_offset, (1, 1)),
    (Method.gain_blk_offset, (5, 5)),
    (Method.gain_offset, (5, 5)),
    (Method.gain_offset, (5, 5)),
])
def test_ref_basic_fit(float_ra: RasterArray, high_res_float_ra: RasterArray, method: Method,
                       kernel_shape: Tuple[int, int]):
    """Test that models are fitted correctly using known parameters"""

    # mask_partial is only applied in RefSpaceModel.apply(), so we just set it False here
    kernel_model = RefSpaceModel(method, kernel_shape, mask_partial=False, r2_inpaint_thresh=0.25)
    src_ra = high_res_float_ra
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
def test_src_basic_fit(float_ra: RasterArray, high_res_float_ra: RasterArray, method: Method,
                              kernel_shape: Tuple[int, int]):
    """Test models are fitted correctly in src space with known parameters"""
    kernel_model = SrcSpaceModel(method, kernel_shape, mask_partial=False, r2_inpaint_thresh=0.25)
    src_ra = float_ra
    ref_ra = high_res_float_ra

    param_ra = kernel_model.fit(ref_ra, src_ra)
    assert (param_ra.shape == src_ra.shape)
    assert (param_ra.transform == src_ra.transform)
    assert (src_ra.mask == param_ra.mask).all()
    assert param_ra.array[0, param_ra.mask] == pytest.approx(1, abs=1e-2)
    assert param_ra.array[1, param_ra.mask] == pytest.approx(0, abs=1e-2)


def test_ref_basic_apply(float_ra: RasterArray, high_res_float_ra: RasterArray):
    """Test application of known ref space parameters"""

    kernel_model = RefSpaceModel(Method.gain_blk_offset, (5, 5), mask_partial=False)
    src_ra = high_res_float_ra

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
def test_ref_masking(float_ra, high_res_float_ra, high_res_offset_float_ra, kernel_shape: Tuple[int, int],
                          mask_partial: bool):
    kernel_model = RefSpaceModel(Method.gain_blk_offset, kernel_shape, mask_partial=mask_partial)
    src_ra = high_res_float_ra

    # create test parameters
    param_ra = float_ra.copy()
    param_mask = param_ra.mask
    param_ra.array = np.ones((2, *param_ra.shape), dtype=param_ra.dtype)
    param_ra.mask = param_mask

    out_ra = kernel_model.apply(src_ra, param_ra)
    if not mask_partial:
        assert (src_ra.mask == out_ra.mask).all()
    else:
        scale = np.round(np.divide(param_ra.res, src_ra.res))
        se = np.ones((scale * (np.array(kernel_shape) + 1) + 1).astype(int))
        test_mask = cv2.erode(src_ra.mask.astype('uint8'), se)
        assert (src_ra.mask.sum() > out_ra.mask.sum())
        assert src_ra.mask[out_ra.mask].all()
        assert (test_mask == out_ra.mask).all()


@pytest.mark.parametrize('kernel_shape, mask_partial', [
    ((1, 1), False),
    ((1, 1), True),
    ((3, 3), True),
    ((3, 7), True),
    ((5, 5), True),
])
def test_src_masking(float_ra, high_res_float_ra, high_res_offset_float_ra, kernel_shape: Tuple[int, int],
                          mask_partial: bool):
    kernel_model = SrcSpaceModel(Method.gain_blk_offset, kernel_shape, mask_partial=mask_partial)
    src_ra = float_ra.copy()
    ref_ra = high_res_float_ra

    # create test parameters
    param_ra = kernel_model.fit(ref_ra, src_ra)

    if not mask_partial:
        assert (src_ra.mask == param_ra.mask).all()
    else:
        se = np.ones(np.array(kernel_shape) + 2).astype(int)
        test_mask = cv2.erode(src_ra.mask.astype('uint8'), se)
        assert (src_ra.mask.sum() > param_ra.mask.sum())
        assert src_ra.mask[param_ra.mask].all()
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