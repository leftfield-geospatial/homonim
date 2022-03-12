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

from homonim.compare import RasterCompare
from homonim.enums import ProcCrs


@pytest.mark.parametrize('src_file, ref_file', [
    ('float_50cm_rgb_file', 'float_100cm_rgb_file'),
    ('float_100cm_rgb_file', 'float_50cm_rgb_file'),
])
def test_compare(src_file, ref_file, request):
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    compare = RasterCompare(src_file, ref_file)
    res_dict = compare.compare()
    assert (len(res_dict) == 4)
    assert ('Mean' in res_dict)
    band_dict = res_dict.copy()
    band_dict.pop('Mean')
    r2 = np.array([stats_dict['r2'] for stats_dict in band_dict.values()])
    RMSE = np.array([stats_dict['RMSE'] for stats_dict in band_dict.values()])
    rRMSE = np.array([stats_dict['rRMSE'] for stats_dict in band_dict.values()])
    N = np.array([stats_dict['N'] for stats_dict in band_dict.values()])
    assert (r2 == pytest.approx(1))
    assert (RMSE == pytest.approx(0))
    assert (rRMSE == pytest.approx(0))
    assert (N == N[0]).all()
    assert (res_dict['Mean']['r2'] == pytest.approx(1))
    assert (res_dict['Mean']['RMSE'] == pytest.approx(0))
    assert (res_dict['Mean']['rRMSE'] == pytest.approx(0))


@pytest.mark.parametrize('src_file, ref_file, exp_proc_crs', [
    ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.ref),
    ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.src),
])
def test_proc_crs(src_file, ref_file, exp_proc_crs, request):
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    compare = RasterCompare(src_file, ref_file)
    assert (compare.proc_crs == exp_proc_crs)
