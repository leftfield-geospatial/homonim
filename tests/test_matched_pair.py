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
from rasterio import Affine
from rasterio.windows import Window

from homonim import utils
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
def multispec_ref_file(tmp_path: Path, float_100cm_array: np.ndarray, float_100cm_profile: Dict) -> Path:
    """
    Single band float32 geotiff with 100cm pixel resolution, the same as float_100cm_src_file, but padded with an
    extra pixel.
    """
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


@pytest.mark.parametrize(['file', 'bands', 'exp_bands', 'exp_band_names', 'exp_wavelengths'], [
    ('rgba_file', None, [1, 2, 3], ['1', '2', '3'], [.650, .560, .480]),
    ('rgba_file', (1, 2, 3), [1, 2, 3], ['1', '2', '3'], [.650, .560, .480]),
    ('masked_file', None, [1], ['1'], [float('nan')]),
    ('float_100cm_src_file', [1], [1], ['1'], [float('nan')]),
    ('s2_ref_file', None, [1, 2, 3], ['B4', 'B3', 'B2'], [0.6645, 0.56, 0.4966]),
    ('s2_ref_file', (3, 2, 1), [3, 2, 1], ['B2', 'B3', 'B4'], [0.4966, 0.56, 0.6645]),
    (
        'landsat_ref_file', None, list(range(1, 8)) + [9], [f'SR_B{i}' for i in range(1, 8)] + ['ST_B10'],
        [0.443, 0.482, 0.562, 0.655, 0.865, 1.609, 2.201, 10.895]
    ),
    ('landsat_ref_file', [7, 8], [7, 8], ['SR_B7', 'SR_QA_AEROSOL'], [2.201, float('nan')]),
])  # yapf: disable
def test_get_band_info(
    file: str, bands: Tuple, exp_bands: List, exp_band_names: List, exp_wavelengths: List,
    request: pytest.FixtureRequest
):
    """ Test _get_band_info returns expected values with different image files. """
    file: Path = request.getfixturevalue(file)
    with rio.open(file, 'r') as im:
        bands, band_names, wavelengths = MatchedPairReader._get_band_info(im, bands=bands)
        assert all(bands == np.array(exp_bands))
        assert all(band_names == np.array(exp_band_names))
        assert all(utils.nan_equals(wavelengths, np.array(exp_wavelengths)))


def test_alpha_band_error(rgba_file):
    """ Test an error is raised in _get_band_info when user bands contain alpha bands. """
    with rio.open(rgba_file, 'r') as im:
        with pytest.raises(ValueError) as ex:
            _, _, _ = MatchedPairReader._get_band_info(im, bands=(4, 1, 2))
        assert 'bands contain alpha band(s)' in str(ex)


def test_invalid_band_error(rgba_file):
    """ Test an error is raised in _get_band_info when user bands contain invalid bands. """
    with rio.open(rgba_file, 'r') as im:
        with pytest.raises(ValueError) as ex:
            _, _, _ = MatchedPairReader._get_band_info(im, bands=(4, 1, 0, 2, 5))
        assert 'bands contain invalid band(s)' in str(ex)


def test_invalid_band_error(rgba_file):
    """ Test an error is raised in _get_band_info when user bands contain invalid bands. """
    with rio.open(rgba_file, 'r') as im:
        with pytest.raises(ValueError) as ex:
            _, _, _ = MatchedPairReader._get_band_info(im, bands=(4, 1, 0, 2, 5))
        assert 'bands contain invalid band(s)' in str(ex)


@pytest.mark.parametrize(
    ['src_file', 'ref_file', 'src_bands', 'ref_bands', 'exp_src_bands', 'exp_ref_bands', 'force'],
    [
        ('float_100cm_src_file', 'float_100cm_ref_file', None, None, [1], [1], False),
        ('float_100cm_src_file', 'float_100cm_ref_file', (1,), (1,), [1], [1], False),
        ('float_100cm_src_file', 'rgba_file', None, [1], [1], [1], False),
        ('rgba_file', 'rgba_file', None, None, [1, 2, 3], [1, 2, 3], False),
        ('rgba_file', 'rgba_file', [3, 2], None, [3, 2], [3, 2], False),
        ('rgba_file', 'rgba_file', [3, 2], [3, 1, 2], [3, 2], [3, 2], False),
        ('s2_ref_file', 's2_ref_file', None, None, [1, 2, 3], [1, 2, 3], False),
        ('s2_ref_file', 's2_ref_file', [3, 2], None, [3, 2], [3, 2], False),
        ('s2_ref_file', 'landsat_ref_file', None, None, [1, 2, 3], [4, 3, 2], False),
        ('s2_ref_file', 'landsat_ref_file', [1, 2], list(range(1, 9)), [1, 2], [4, 3], False),
        ('s2_ref_file', 'landsat_ref_file', [1, 2], [3, 8], [1, 2], [8, 3], False),
        ('landsat_src_file', 'landsat_ref_file', [7, 8, 9], [7, 9, 10], [7, 8, 9], [7, 10, 9], False),
        ('landsat_src_file', 'landsat_ref_file', [7, 8, 9, 10], list(range(1, 12)), [7, 8, 9, 10], [7, 1, 9, 2], True),
        ('landsat_src_file', 's2_ref_file', None, None, [2, 3, 4], [3, 2, 1], True),
    ]
)  # yapf: disable
def test_matching(
    src_file: str, ref_file: str, src_bands: Tuple, ref_bands: Tuple, exp_src_bands: List, exp_ref_bands: List,
    force: bool, request: pytest.FixtureRequest
):
    """ Test matching of different source and reference files. """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    with MatchedPairReader(src_file, ref_file, src_bands=src_bands, ref_bands=ref_bands, force=force) as matched_pair:
        assert all(np.array(matched_pair.src_bands) == exp_src_bands)
        assert all(np.array(matched_pair.ref_bands) == exp_ref_bands)
