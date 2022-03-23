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

import pytest

from homonim.cli import cli
from homonim.errors import ImageFormatError
from homonim.stats import ParamStats


def _test_vals(param_stats):
    assert len(param_stats) == 9
    for band_name, band_stats in param_stats.items():
        assert ({'Mean', 'Std', 'Min', 'Max'} <= set(band_stats.keys()))

    exp_param_stats_list = (3 * [{'Mean': 1, 'Std': 0, 'Min': 1, 'Max': 1}] +  # gains
                            3 * [{'Mean': 0, 'Std': 0, 'Min': 0, 'Max': 0}] +  # offsets
                            3 * [{'Mean': 1, 'Std': 0, 'Min': 1, 'Max': 1, 'Inpaint (%)': 0}])  # r2 vals

    for param_band_stats, exp_param_band_stats in zip(param_stats.values(), exp_param_stats_list):
        for k, v in exp_param_band_stats.items():
            assert (param_band_stats[k] == pytest.approx(exp_param_band_stats[k], abs=1e-2))


def test_api(param_file):
    stats = ParamStats(param_file)
    assert (len(stats.metadata) > 0)
    param_stats = stats.stats()
    _test_vals(param_stats)


def test_api__file_format_error(float_100cm_rgb_file):
    with pytest.raises(ImageFormatError):
        _ = ParamStats(float_100cm_rgb_file)


def test_cli(runner, param_file):
    cli_str = f'stats {param_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (param_file.name in result.output)

    res_str = """Mean  Std   Min  Max  Inpaint (%)
B1_GAIN    1.00 0.00  1.00 1.00      NaN    
B2_GAIN    1.00 0.00  1.00 1.00      NaN    
B3_GAIN    1.00 0.00  1.00 1.00      NaN    
B1_OFFSET -0.00 0.00 -0.00 0.00      NaN    
B2_OFFSET -0.00 0.00 -0.00 0.00      NaN    
B3_OFFSET -0.00 0.00 -0.00 0.00      NaN    
B1_R2      1.00 0.00  1.00 1.00     0.00    
B2_R2      1.00 0.00  1.00 1.00     0.00    
B3_R2      1.00 0.00  1.00 1.00     0.00 """
    assert (res_str in result.output)


def test_cli__out_file(tmp_path, runner, param_file):
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
    output_file = tmp_path.joinpath('stats.json')
    cli_str = f'stats {param_file} {param_file} --output {output_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (output_file.exists())

    with open(output_file) as f:
        stats_dict = json.load(f)

    param_file = str(param_file)
    assert (param_file in stats_dict)
