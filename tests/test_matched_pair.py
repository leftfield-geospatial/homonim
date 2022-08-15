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
from typing import Tuple, List, Dict

import numpy as np
import pytest
import rasterio as rio
from pytest import FixtureRequest
from rasterio import MemoryFile
from rasterio.enums import Resampling
from rasterio.windows import Window, union

from homonim import utils
from homonim.enums import ProcCrs
from homonim.errors import ImageContentError, BlockSizeError, IoError
from homonim.matched_pair import MatchedPairReader


@pytest.fixture
def multispec_src_file(tmp_path: Path, float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> Path:
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


@pytest.mark.parametrize(['file', 'bands', 'exp_bands', 'exp_band_names', 'exp_wavelengths'], [
    ('rgba_file', None, (1, 2, 3), ['1', '2', '3'], [.650, .560, .480]),
    ('rgba_file', (1, 2, 3), [1, 2, 3], ['1', '2', '3'], [.650, .560, .480]),
    ('masked_file', None, [1], ['1'], [float('nan')]),
    ('float_100cm_src_file', [1], [1], ['1'], [float('nan')]),
])  # yapf: disable
def test_get_band_info(
    file: str, bands: Tuple, exp_bands: List, exp_band_names: List, exp_wavelengths: List,
    request: pytest.FixtureRequest
):
    file: Path = request.getfixturevalue(file)
    with rio.open(file, 'r') as im:
        bands, band_names, wavelengths = MatchedPairReader._get_band_info(im, bands=bands)
        assert all(bands == np.array(exp_bands))
        assert all(band_names == np.array(exp_band_names))
        assert all(utils.nan_equals(wavelengths, np.array(exp_wavelengths)))


def test_alpha_band_error(rgba_file):
    with rio.open(rgba_file, 'r') as im:
        with pytest.raises(ValueError) as ex:
            _, _, _ = MatchedPairReader._get_band_info(im, bands=(4, 1, 2))
        assert 'bands contain alpha band(s)' in str(ex)

