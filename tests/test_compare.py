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
import json

import numpy as np
import pytest

from homonim.cli import cli
from homonim.compare import RasterCompare
from homonim.enums import ProcCrs
from tests.conftest import str_contain_nos


def _test_identical_compare_dict(res_dict):
    """ Helper function to run tests on a compare results dictionary, where the compare was between identical images. """
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


@pytest.mark.parametrize(
    'src_file, ref_file', [
        ('float_50cm_rgb_file', 'float_100cm_rgb_file'),
        ('float_100cm_rgb_file', 'float_50cm_rgb_file'),
    ]
)  # yapf:disable
def test_api(src_file, ref_file, request):
    """ Basic test of RasterCompare for proc_crs=ref&src image combinations. """
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    compare = RasterCompare(src_file, ref_file)
    res_dict = compare.compare()
    _test_identical_compare_dict(res_dict)


@pytest.mark.parametrize(
    'src_file, ref_file, proc_crs, exp_proc_crs', [
        ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, ProcCrs.ref),
        ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.src, ProcCrs.src),
        ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.auto, ProcCrs.src),
        ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.ref, ProcCrs.ref),
    ]
)  # yapf:disable
def test_api__proc_crs(src_file, ref_file, proc_crs, exp_proc_crs, request):
    """ Test resolution and forcing of the proc_crs parameter with different combinations of src/ref images. """
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    compare = RasterCompare(src_file, ref_file, proc_crs=proc_crs)
    assert (compare.proc_crs == exp_proc_crs)
    res_dict = compare.compare()
    assert (len(res_dict) == 2)


def test_api__single_thread(float_100cm_src_file, float_100cm_ref_file):
    """ Test single threaded compare. """
    compare = RasterCompare(float_100cm_src_file, float_100cm_ref_file, threads=1)
    res_dict = compare.compare()
    assert (len(res_dict) == 2)


def test_cli(runner, float_50cm_rgb_file, float_100cm_rgb_file):
    """ Test compare CLI with known outputs. """
    ref_file = float_100cm_rgb_file
    src_file = float_50cm_rgb_file

    cli_str = f'compare {src_file} {ref_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    res_str = """Band 1 1.00  0.00  0.00   144
Band 2 1.00  0.00  0.00   144
Band 3 1.00  0.00  0.00   144
Mean   1.00  0.00  0.00   144"""
    assert (str_contain_nos(res_str, result.output))


def test_cli__output_file(tmp_path, runner, float_50cm_rgb_file, float_100cm_rgb_file):
    """ Test compare CLI generated json file. """
    ref_file = float_100cm_rgb_file
    src_file = float_50cm_rgb_file

    output_file = tmp_path.joinpath('compare.json')
    cli_str = f'compare {src_file} {ref_file} --output {output_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (output_file.exists())

    with open(output_file) as f:
        stats_dict = json.load(f)

    src_file = str(src_file)
    assert (src_file in stats_dict)
    _test_identical_compare_dict(stats_dict[src_file])


def test_cli__mult_inputs(tmp_path, runner, float_50cm_rgb_file, float_100cm_rgb_file):
    """ Test compare CLI with multiple src files. """
    ref_file = float_100cm_rgb_file
    src_file = float_50cm_rgb_file

    output_file = tmp_path.joinpath('compare.json')
    cli_str = f'compare {src_file} {src_file} {ref_file} --output {output_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (output_file.exists())

    with open(output_file) as f:
        stats_dict = json.load(f)

    src_file = str(src_file)
    assert (src_file in stats_dict)


@pytest.mark.parametrize('proc_crs', ['auto', 'src', 'ref'])
def test_cli__proc_crs(tmp_path, runner, float_50cm_rgb_file, float_100cm_rgb_file, proc_crs):
    """ Test compare CLI proc_crs behaviour ok. """
    ref_file = float_100cm_rgb_file
    src_file = float_50cm_rgb_file

    output_file = tmp_path.joinpath('compare.json')
    cli_str = f'compare {src_file} {ref_file} --proc-crs {proc_crs} --output {output_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (output_file.exists())

    with open(output_file) as f:
        stats_dict = json.load(f)

    src_file = str(src_file)
    assert (src_file in stats_dict)
    _test_identical_compare_dict(stats_dict[src_file])
