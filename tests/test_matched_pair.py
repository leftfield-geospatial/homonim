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

import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio

from homonim import utils
from homonim.errors import HomonimError, HomonimWarning
from homonim.matched_pair import MatchedPairReader


@pytest.mark.parametrize(
    ['file', 'bands', 'exp_bands', 'exp_names', 'exp_wavelengths'],
    [
        ('file_rgba', None, [1, 2, 3], ['1', '2', '3'], [0.650, 0.560, 0.480]),
        ('file_rgba', (1, 2, 3), [1, 2, 3], ['1', '2', '3'], [0.650, 0.560, 0.480]),
        ('file_rgba', (1, 2, 3), [1, 2, 3], ['1', '2', '3'], [0.650, 0.560, 0.480]),
        ('file_masked', None, [1], ['1'], [float('nan')]),
        ('src_file_100cm_float', [1], [1], ['1'], [float('nan')]),
        ('s2_ref_file', None, [1, 2, 3], ['B4', 'B3', 'B2'], [0.6645, 0.56, 0.4966]),
        ('s2_ref_file', (3, 2, 1), [3, 2, 1], ['B2', 'B3', 'B4'], [0.4966, 0.56, 0.6645]),
        ('landsat_ref_file', None, [*range(1, 8), 9], [f'SR_B{i}' for i in range(1, 8)] + ['ST_B10'], [0.443, 0.482, 0.562, 0.655, 0.865, 1.609, 2.201, 10.895]),
        ('landsat_ref_file', [7, 8], [7, 8], ['SR_B7', 'SR_QA_AEROSOL'], [2.201, float('nan')]),
        ('file_bgr', None, [1, 2, 3], ['1', '2', '3'], [0.480, 0.560, 0.650]),
        ('file_bgr', (3, 2, 1), [3, 2, 1], ['3', '2', '1'], [0.650, 0.560, 0.480]),
    ],
)  # fmt: skip
def test_get_band_info(
    file: str,
    bands: tuple,
    exp_bands: list,
    exp_names: list,
    exp_wavelengths: list,
    request: pytest.FixtureRequest,
):
    """Test _get_band_info() returns expected values with different image files."""
    file: Path = request.getfixturevalue(file)
    with rio.open(file, 'r') as im:
        bands, names, wavelengths = MatchedPairReader._get_band_info(im, bands=bands)
        assert all(bands == np.array(exp_bands))
        assert all(names == np.array(exp_names))
        assert all(utils.nan_equals(wavelengths, np.array(exp_wavelengths)))


def test_get_band_info_alpha_band_error(file_rgba):
    """Test _get_band_info() raises an error when provided bands contain alpha bands."""
    with rio.open(file_rgba, 'r') as im:
        with pytest.raises(HomonimError, match='contains alpha band indexes'):
            MatchedPairReader._get_band_info(im, bands=(4, 1, 2))


def test_get_band_info_invalid_band_error(file_rgba):
    """Test _get_band_info() raises an error when provided bands contain out of range
    indexes.
    """
    with rio.open(file_rgba, 'r') as im:
        with pytest.raises(HomonimError, match='contains out of range indexes'):
            MatchedPairReader._get_band_info(im, bands=(4, 1, 0, 2, 5))


@pytest.mark.parametrize(
    ['src_file', 'ref_file', 'src_bands', 'ref_bands', 'exp_src_bands', 'exp_ref_bands', 'force'],
    [
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
        # ref with mix of reflectance and non-alpha bands, len(src) == len(ref)
        ('s2_ref_file', 'landsat_ref_file', [1, 2], [3, 8], [1, 2], [8, 3], False),
        # src & ref with mix of reflectance and non-alpha bands, len(src) == len(ref)
        ('landsat_src_file', 'landsat_ref_file', [7, 8, 9, 10], [7, 9, 10, 11], [7, 8, 9, 10], [7, 10, 9, 11], False),
        # src & ref with mix of reflectance and non-alpha bands, len(src) < len(ref) & force
        ('landsat_src_file', 'landsat_ref_file', [7, 8, 9, 10], list(range(1, 12)), [7, 8, 9, 10], [1, 2, 3, 4], True),
        # len(src) > len(ref) & force
        ('landsat_src_file', 's2_ref_file', None, None, [1, 2, 3], [1, 2, 3], True),
    ],
)  # fmt: skip
def test_match(
    src_file: str,
    ref_file: str,
    src_bands: tuple,
    ref_bands: tuple,
    exp_src_bands: list,
    exp_ref_bands: list,
    force: bool,
    request: pytest.FixtureRequest,
):
    """Test matching of different source / reference images and bands."""
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)

    with warnings.catch_warnings():
        # test there are no all-nan warnings by turning RuntimeWarning into an
        # error
        warnings.simplefilter('error', category=RuntimeWarning)
        with MatchedPairReader(
            src_file, ref_file, src_bands=src_bands, ref_bands=ref_bands, force=force
        ) as mp:
            assert all(np.array(mp.src_bands) == exp_src_bands)
            assert all(np.array(mp.ref_bands) == exp_ref_bands)


def test_match_fewer_ref_bands_error(s2_ref_file, landsat_ref_file):
    """Test matching raises an error if there fewer reference than source bands."""
    with pytest.raises(HomonimError, match='fewer reference than source'):
        with MatchedPairReader(landsat_ref_file, s2_ref_file):
            pass
    with pytest.raises(HomonimError, match='fewer reference than source'):
        with MatchedPairReader(
            landsat_ref_file, s2_ref_file, src_bands=(1, 2), ref_bands=(1,)
        ):
            pass


def test_match_wavelength_dist_error(s2_ref_file, landsat_ref_file):
    """Test matching raises an error if source / reference reflectance band wavelengths
    are too far apart.
    """
    with pytest.raises(HomonimError, match='could not be wavelength matched'):
        with MatchedPairReader(s2_ref_file, landsat_ref_file, ref_bands=[1, 3, 5]):
            pass


@pytest.mark.parametrize(
    ['src_file', 'ref_file', 'src_bands', 'ref_bands'],
    [
        ('src_file_100cm_float', 'file_rgba', None, None),
        ('landsat_src_file', 'landsat_ref_file', [7, 8, 9], None),
    ],
)
def test_match_error(
    src_file: str,
    ref_file: str,
    src_bands: tuple,
    ref_bands: tuple,
    request: pytest.FixtureRequest,
):
    """Test matching raises an error when one or more bands don't have center
    wavelengths and source / reference band counts are unequal.
    """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    with pytest.raises(HomonimError, match='Could not match'):
        with MatchedPairReader(
            src_file, ref_file, src_bands=src_bands, ref_bands=ref_bands
        ):
            pass


@pytest.mark.parametrize(
    'src_file, ref_file',
    [
        ('s2_ref_file', 'landsat_ref_file'),
        ('ngi_src_file', 's2_ref_file'),
    ],
)
def test_matched_band_table(
    src_file: str, ref_file: str, request: pytest.FixtureRequest
):
    """Test matched_band_table()."""
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    with MatchedPairReader(src_file, ref_file) as mp:
        table = mp.matched_band_table()
        for im, bands in zip(
            [mp._src_im, mp._ref_im], [mp.src_bands, mp.ref_bands], strict=True
        ):
            names = [
                im.tags(bi).get('name', None) or im.descriptions[bi - 1] or str(bi)
                for bi in bands
            ]
            assert all(n in table for n in names)
