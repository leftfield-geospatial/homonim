"""
    Homonim: Correction of aerial and satellite imagery to surface relfectance
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

import pytest

from homonim.cli import cli
from homonim.errors import ImageFormatError
from homonim.stats import ParamStats
from tests.conftest import str_contain_nos


def _test_vals(param_stats):
    """ Helper function to test statistics against known values for param_file (i.e. gain=1, offset=0, r2=1). """
    assert len(param_stats) == 9
    for band_stats in param_stats:
        assert ({'band', 'mean', 'std', 'min', 'max'} <= set(band_stats.keys()))

    exp_param_stats_list = (
        3 * [{'mean': 1, 'std': 0, 'min': 1, 'max': 1}] +  # gains
        3 * [{'mean': 0, 'std': 0, 'min': 0, 'max': 0}] +  # offsets
        3 * [{'mean': 1, 'std': 0, 'min': 1, 'max': 1, 'inpaint_p': 0}]   # r2
    )  # yapf: disable

    for param_band_stats, exp_param_band_stats in zip(param_stats, exp_param_stats_list):
        for k, v in exp_param_band_stats.items():
            assert (param_band_stats[k] == pytest.approx(exp_param_band_stats[k], abs=1e-2))


def test_api(param_file):
    """ Test ParamStats creation and execution. """
    with ParamStats(param_file) as stats:
        assert (len(stats.metadata) > 0)
        param_stats = stats.stats()
    _test_vals(param_stats)


def test_api__file_format_error(float_100cm_rgb_file):
    """ Test incorrect parameter file format raises an error. """
    with pytest.raises(ImageFormatError):
        _ = ParamStats(float_100cm_rgb_file)


def test_cli(runner, param_file):
    """ Test stats cli generates the correct output. """
    cli_str = f'stats {param_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (param_file.name in result.output)

    res_str = """B1_GAIN    1.000 0.000  1.000 1.000          
    B2_GAIN    1.000 0.000  1.000 1.000          
    B3_GAIN    1.000 0.000  1.000 1.000          
    B1_OFFSET -0.000 0.000 -0.001 0.000          
    B2_OFFSET -0.000 0.000 -0.001 0.000          
    B3_OFFSET -0.000 0.000 -0.001 0.000          
    B1_R2      1.000 0.000  1.000 1.000     0.000    
    B2_R2      1.000 0.000  1.000 1.000     0.000    
    B3_R2      1.000 0.000  1.000 1.000     0.000"""
    assert (str_contain_nos(res_str, result.output))


def test_cli__out_file(tmp_path, runner, param_file):
    """ Test stats cli --output option generates the correct values. """
    output_file = tmp_path.joinpath('stats.json')
    cli_str = f'stats {param_file} --output {output_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (output_file.exists())

    with open(output_file) as f:
        stats_dict = json.load(f)

    param_file = str(param_file)
    assert (param_file in stats_dict)
    param_stats = stats_dict[param_file]
    _test_vals(param_stats)


def test_cli__mult_inputs(tmp_path, runner, param_file):
    """ Test stats cli with multiple input files. """
    output_file = tmp_path.joinpath('stats.json')
    cli_str = f'stats {param_file} {param_file} --output {output_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (output_file.exists())

    with open(output_file) as f:
        stats_dict = json.load(f)

    param_file = str(param_file)
    assert (param_file in stats_dict)


def test_cli__file_format_error(runner, float_100cm_rgb_file):
    """ Test stats cli fails with error message when the parameter file format is incorrect. """
    cli_str = f'stats {float_100cm_rgb_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ('Invalid value' in result.output)
