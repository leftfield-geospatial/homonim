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

import json
from pathlib import Path
from typing import Dict, List

import pytest
import rasterio as rio
from click.testing import CliRunner
from pytest import FixtureRequest
from rasterio.windows import get_data_window, Window

from homonim.cli import cli
from homonim.errors import ImageFormatError, IoError
from homonim.stats import ParamStats
from tests.conftest import str_contain_no_space


def _test_vals(param_stats: List[Dict]):
    """ Helper function to test statistics against known values for param_file (i.e. gain=1, offset=0, r2=1). """
    assert len(param_stats) == 9
    for band_stats in param_stats:
        assert ({'band', 'mean', 'std', 'min', 'max'} <= set(band_stats.keys()))

    exp_param_stats_list = (
        3 * [{'mean': 1, 'std': 0, 'min': 1, 'max': 1}] +  # gains
        3 * [{'mean': 0, 'std': 0, 'min': 0, 'max': 0}] +  # offsets
        3 * [{'mean': 1, 'std': 0, 'min': 1, 'max': 1, 'inpaint_p': 0}]  # r2
    )  # yapf: disable

    for param_band_stats, exp_param_band_stats in zip(param_stats, exp_param_stats_list):
        for k, v in exp_param_band_stats.items():
            assert (param_band_stats[k] == pytest.approx(exp_param_band_stats[k], abs=1e-2))


@pytest.mark.parametrize('param_file_str', ['param_file', 'param_file_tile_10x20'])
def test_api__stats(param_file_str: str, request: FixtureRequest):
    """ Test ParamStats creation and execution. """
    param_file: Path = request.getfixturevalue(param_file_str)
    with ParamStats(param_file) as stats:
        assert (len(stats.metadata) > 0)
        param_stats = stats.stats()
    _test_vals(param_stats)


def test_api__context(param_file: Path):
    """ Test ParamStats context management. """
    stats = ParamStats(param_file)
    # test ParamStats usage without context entry raises an IoError
    with pytest.raises(IoError):
        _ = stats.stats()
    # test ParamStats.stats usage with context entry is ok, and file is closed on context exit
    with stats:
        assert (len(stats.metadata) > 0)
        _ = stats.stats()
    assert stats.closed


def test_api__tables(param_file: Path):
    """ Test ParamStats table generation. """
    with ParamStats(param_file) as stats:
        assert (len(stats.metadata) > 0)
        param_stats = stats.stats()

    schema_table = stats.schema_table
    assert len(schema_table) > 0

    param_table = stats.stats_table(param_stats)
    assert len(param_table) > 0

    # test stats data are in table
    for band_stats in param_stats:
        for k, v in band_stats.items():
            assert (f'{v:.3f}' in param_table) if isinstance(v, float) else (v in param_table)

    # test stats abbreviations are in table
    for schema_dict in stats.schema.values():
        assert schema_dict['abbrev'] in param_table


@pytest.mark.parametrize('threads', [1, 0])
def test_api__threads(param_file: Path, threads: int):
    """ Test ParamStats.stats works multi- and single-threaded. """
    with ParamStats(param_file) as stats:
        assert (len(stats.metadata) > 0)
        param_stats = stats.stats(threads=threads)
    _test_vals(param_stats)


@pytest.mark.parametrize('param_file_str', ['param_file', 'param_file_tile_10x20'])
def test_api__data_window(param_file_str: str, request: FixtureRequest):
    """ Test ParamStats._get_data_window() accumulates block windows correctly. """
    param_file: Path = request.getfixturevalue(param_file_str)
    with rio.open(param_file, 'r') as param_im:
        mask = param_im.read_masks(indexes=1)
        data_win = get_data_window(mask, nodata=0)
        image_win = Window(0, 0, param_im.width, param_im.height)
        assert data_win != image_win

    with ParamStats(param_file) as stats:
        assert data_win == stats._get_data_window()


def test_api__file_format_error(float_100cm_rgb_file: Path):
    """ Test incorrect parameter file format raises an error. """
    with pytest.raises(ImageFormatError):
        _ = ParamStats(float_100cm_rgb_file)


@pytest.mark.parametrize('param_file_str', ['param_file', 'param_file_tile_10x20'])
def test_cli(runner: CliRunner, param_file_str: str, request: FixtureRequest):
    """ Test stats cli generates the correct output. """
    param_file: Path = request.getfixturevalue(param_file_str)
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
    assert (str_contain_no_space(res_str, result.output))


def test_cli__out_file(tmp_path: Path, runner: CliRunner, param_file: Path):
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


def test_cli__mult_inputs(tmp_path: Path, runner: CliRunner, param_file: Path):
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


def test_cli__file_format_error(runner: CliRunner, float_100cm_rgb_file: Path):
    """ Test stats cli fails with error message when the parameter file format is incorrect. """
    cli_str = f'stats {float_100cm_rgb_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ('Invalid value' in result.output)

##
