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
from typing import Tuple, List
import warnings

import numpy as np
import pytest
import rasterio as rio

from homonim import utils
from homonim.matched_pair import MatchedPairReader
from homonim.errors import HomonimWarning


@pytest.mark.parametrize(['file', 'bands', 'exp_bands', 'exp_band_names', 'exp_wavelengths'], [
    ('file_rgba', None, [1, 2, 3], ['1', '2', '3'], [.650, .560, .480]),
    ('file_rgba', (1, 2, 3), [1, 2, 3], ['1', '2', '3'], [.650, .560, .480]),
    ('file_rgba', (1, 2, 3), [1, 2, 3], ['1', '2', '3'], [.650, .560, .480]),
    ('file_masked', None, [1], ['1'], [float('nan')]),
    ('src_file_100cm_float', [1], [1], ['1'], [float('nan')]),
    ('s2_ref_file', None, [1, 2, 3], ['B4', 'B3', 'B2'], [0.6645, 0.56, 0.4966]),
    ('s2_ref_file', (3, 2, 1), [3, 2, 1], ['B2', 'B3', 'B4'], [0.4966, 0.56, 0.6645]),
    (
        'landsat_ref_file', None, list(range(1, 8)) + [9], [f'SR_B{i}' for i in range(1, 8)] + ['ST_B10'],
        [0.443, 0.482, 0.562, 0.655, 0.865, 1.609, 2.201, 10.895]
    ),
    ('landsat_ref_file', [7, 8], [7, 8], ['SR_B7', 'SR_QA_AEROSOL'], [2.201, float('nan')]),
    ('file_bgr', None, [1, 2, 3], ['1', '2', '3'], [ .480, .560, .650]),
    ('file_bgr', (3, 2, 1), [3, 2, 1], ['3', '2', '1'], [.650, .560, .480]),
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


def test_get_band_info_alpha_band_error(file_rgba):
    """ Test an error is raised in _get_band_info when user bands contain alpha bands. """
    with rio.open(file_rgba, 'r') as im:
        with pytest.raises(ValueError) as ex:
            _, _, _ = MatchedPairReader._get_band_info(im, bands=(4, 1, 2))
        assert 'bands contain alpha band(s)' in str(ex)


def test_get_band_info_invalid_band_error(file_rgba):
    """ Test an error is raised in _get_band_info when user bands contain invalid bands. """
    with rio.open(file_rgba, 'r') as im:
        with pytest.raises(ValueError) as ex:
            _, _, _ = MatchedPairReader._get_band_info(im, bands=(4, 1, 0, 2, 5))
        assert 'bands contain invalid band(s)' in str(ex)


@pytest.mark.parametrize(
    ['src_file', 'ref_file', 'src_bands', 'ref_bands', 'exp_src_bands', 'exp_ref_bands', 'force'], [
        ('src_file_100cm_float', 'ref_file_100cm_float', None, None, [1], [1], False),
        ('src_file_100cm_float', 'ref_file_100cm_float', (1,), (1,), [1], [1], False),
        ('src_file_100cm_float', 'file_rgba', None, [1], [1], [1], False),
        ('file_rgba', 'file_rgba', None, None, [1, 2, 3], [1, 2, 3], False),
        ('file_rgba', 'file_rgba', [3, 2], None, [3, 2], [3, 2], False),
        ('file_rgba', 'file_rgba', [3, 2], [3, 1, 2], [3, 2], [3, 2], False),
        ('s2_ref_file', 's2_ref_file', None, None, [1, 2, 3], [1, 2, 3], False),
        ('s2_ref_file', 's2_ref_file', [3, 2], None, [3, 2], [3, 2], False),
        ('s2_ref_file', 'landsat_ref_file', None, None, [1, 2, 3], [4, 3, 2], False),
        ('s2_ref_file', 'landsat_ref_file', [1, 2], list(range(1, 9)), [1, 2], [4, 3], False),
        # ref with mix of spectral and non-alpha bands, len(src) == len(ref)
        ('s2_ref_file', 'landsat_ref_file', [1, 2], [3, 8], [1, 2], [8, 3], False),
        # src & ref with mix of spectral and non-alpha bands, len(src) == len(ref)
        ('landsat_src_file', 'landsat_ref_file', [7, 8, 9, 10], [7, 9, 10, 11], [7, 8, 9, 10], [7, 10, 9, 11], False),
        # src & ref with mix of spectral and non-alpha bands, len(src) < len(ref) & force
        ('landsat_src_file', 'landsat_ref_file', [7, 8, 9, 10], list(range(1, 12)), [7, 8, 9, 10], [1, 2, 3, 4], True),
        # len(src) > len(ref) & force
        ('landsat_src_file', 's2_ref_file', None, None, [1, 2, 3], [1, 2, 3], True),
    ]
)  # yapf: disable
def test_match(
    src_file: str, ref_file: str, src_bands: Tuple, ref_bands: Tuple, exp_src_bands: List, exp_ref_bands: List,
    force: bool, request: pytest.FixtureRequest
):
    """ Test matching of different source and reference files. """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)

    with warnings.catch_warnings():
        # test there are no all-nan warnings by turning them RuntimeWarning into an error, while allowing
        # HomonimWarning which sub-classes RuntimeWarning
        warnings.simplefilter("error", category=RuntimeWarning)
        warnings.simplefilter("default", category=HomonimWarning)
        with MatchedPairReader(src_file, ref_file, src_bands=src_bands, ref_bands=ref_bands, force=force) as matched_pair:
            assert all(np.array(matched_pair.src_bands) == exp_src_bands)
            assert all(np.array(matched_pair.ref_bands) == exp_ref_bands)


def test_match_fewer_ref_bands_error(s2_ref_file, landsat_ref_file):
    """  Test an error is raised if num src bands > num ref bands. """
    with pytest.raises(ValueError) as ex:
        with MatchedPairReader(landsat_ref_file, s2_ref_file) as matched_pair:
            pass
    assert 'has fewer bands than' in str(ex)


def test_match_wavelength_dist_error(s2_ref_file, landsat_ref_file):
    """  Test an error is raised if src/ref spectral band wavelengths are too far apart. """
    with pytest.raises(ValueError) as ex:
        with MatchedPairReader(s2_ref_file, landsat_ref_file, ref_bands=[1, 3, 5]) as matched_pair:
            pass
    assert 'could not be auto-matched' in str(ex)

@pytest.mark.parametrize(
    ['src_file', 'ref_file', 'src_bands', 'ref_bands','force'], [
        ('src_file_100cm_float', 'file_rgba', None, None, False),
        ('landsat_src_file', 'landsat_ref_file', [7, 8, 9], None, False),
    ]
)  # yapf: disable
def test_match_error(
    src_file: str, ref_file: str, src_bands: Tuple, ref_bands: Tuple, force: bool, request: pytest.FixtureRequest
):
    """  Test an error is raised if src/ref spectral band wavelengths are too far apart. """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    with pytest.raises(ValueError) as ex:
        with MatchedPairReader(
            src_file, ref_file, src_bands=src_bands, ref_bands=ref_bands, force=force
        ) as matched_pair:
            pass
    assert 'Could not match' in str(ex)
