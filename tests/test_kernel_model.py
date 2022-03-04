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

@pytest.mark.parametrize('method, kernel_shape, r2_inpaint_thresh, mask_partial', [
    (Method.gain, (1, 1), None, False),
    (Method.gain, (3, 3), None, True),
    (Method.gain_blk_offset, (1, 1), None, False),
    (Method.gain_blk_offset, (5, 5), None, True),
    (Method.gain_offset, (5, 5), 0.25, True),
    (Method.gain_offset, (5, 5), None, False),
])
def test_ref_kernel_model_fit(float_ra: RasterArray, method: Method, kernel_shape: tuple[int, int],
                              r2_inpaint_thresh: float, mask_partial: bool):
    kernel_model = RefSpaceModel(method, kernel_shape, mask_partial=mask_partial)
    src_ra = float_ra
    ref_scale = 2
    ref_shape = tuple((np.array(src_ra.shape)/ref_scale).astype('int') + 2)
    ref_transform = src_ra.transform * Affine.scale(ref_scale) * Affine.translation(-1, -1)
    ref_ra = src_ra.reproject(transform=ref_transform, shape=ref_shape, resampling=Resampling.average)

    param_ra = kernel_model.fit(ref_ra, src_ra)
    assert (param_ra.shape == ref_ra.shape)
    assert (param_ra.transform == ref_ra.transform)
    assert ref_ra.mask[param_ra.mask].all()
    assert param_ra.array[0, param_ra.mask] == pytest.approx(1, abs=1.e-3)
    assert param_ra.array[1, param_ra.mask] == pytest.approx(0, abs=1.e-3)

@pytest.mark.parametrize('method, kernel_shape, r2_inpaint_thresh, mask_partial', [
    (Method.gain, (1, 1), None, False),
    (Method.gain, (3, 3), None, True),
    (Method.gain_blk_offset, (1, 1), None, False),
    (Method.gain_blk_offset, (5, 5), None, True),
    (Method.gain_offset, (5, 5), 0.25, False),
    (Method.gain_offset, (5, 5), None, True),
])
def test_src_kernel_model_fit(float_ra: RasterArray, method: Method, kernel_shape: tuple[int, int],
                              r2_inpaint_thresh: float, mask_partial: bool):
    kernel_model = SrcSpaceModel(method, kernel_shape, mask_partial=mask_partial)
    src_ra = float_ra
    ref_scale = .5
    ref_shape = tuple((np.array(src_ra.shape)/ref_scale  + (2 / ref_scale)).astype('int'))
    ref_transform = src_ra.transform  * Affine.translation(-1, -1) * Affine.scale(ref_scale)
    ref_ra = src_ra.reproject(transform=ref_transform, shape=ref_shape, resampling=Resampling.nearest)

    param_ra = kernel_model.fit(ref_ra, src_ra)
    assert (param_ra.shape == src_ra.shape)
    assert (param_ra.transform == src_ra.transform)
    assert src_ra.mask[param_ra.mask].all()
    assert param_ra.array[0, param_ra.mask] == pytest.approx(1, abs=1e-2)
    assert param_ra.array[1, param_ra.mask] == pytest.approx(0, abs=1e-2)

# TODO:
# - different src and ref masks (check param mask is as expected)
# - src and ref not aligned on same grid (check we don't lose data, and that params are close to expected similar to above)
# - r2 inpainting (above e.g. has r2~1 so does not do inpainting, not sure really how to test this... make artifical bad r2 area)
# - mask-partial works as expected for src and ref space (if possible check explicity with synthetic data)
# - separate fit and apply tests if possible.  apply does mask_partial in for ref-space model.  fit does it for src space model.