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

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from pytest import FixtureRequest

from homonim.cli import cli
from homonim.errors import HomonimError
from homonim.stats import ParamStats
from tests.conftest import str_contain_no_space

# expected ParamStats.stats_table() contents for the param_file fixture
_exp_table = """B1_GAIN    1.000 0.000  1.000 1.000
    B2_GAIN    1.000 0.000  1.000 1.000
    B3_GAIN    1.000 0.000  1.000 1.000
    B1_OFFSET -0.000 0.000 -0.001 0.000
    B2_OFFSET -0.000 0.000 -0.001 0.000
    B3_OFFSET -0.000 0.000 -0.001 0.000
    B1_R2      1.000 0.000  1.000 1.000     0.000
    B2_R2      1.000 0.000  1.000 1.000     0.000
    B3_R2      1.000 0.000  1.000 1.000     0.000"""


def _test_vals(param_stats: list[dict]):
    """Test statistics against expected values."""
    assert len(param_stats) == 9
    for band_stats in param_stats:
        assert {'band', 'mean', 'std', 'min', 'max'} <= set(band_stats.keys())

    exp_param_stats = (
        3 * [{'mean': 1, 'std': 0, 'min': 1, 'max': 1}]  # gains
        + 3 * [{'mean': 0, 'std': 0, 'min': 0, 'max': 0}]  # offsets
        + 3 * [{'mean': 1, 'std': 0, 'min': 1, 'max': 1, 'inpaint_p': 0}]  # r2
    )

    for band_stats, exp_band_stats in zip(param_stats, exp_param_stats, strict=True):
        for k in exp_band_stats.keys():
            assert band_stats[k] == pytest.approx(exp_band_stats[k], abs=1e-2)


def test_api__init(param_file: Path):
    """Test ParamStats initialisation and context management."""
    with ParamStats(param_file) as stats:
        assert len(stats.metadata) > 0
    assert stats.closed


@pytest.mark.parametrize('param_file_str', ['param_file', 'param_file_tile_10x20'])
def test_api__stats(param_file_str: str, request: FixtureRequest):
    """Test ParamStats.stats() results against expected values for differently tiled
    parameter images.
    """
    param_file: Path = request.getfixturevalue(param_file_str)
    with ParamStats(param_file) as stats:
        param_stats = stats.stats()
    _test_vals(param_stats)


def test_api__ctx_error(param_file: Path):
    """Test that using a closed ParamStats context raises an error."""
    stats = ParamStats(param_file)
    stats.close()

    with pytest.raises(RuntimeError, match='closed'):
        stats.stats()
    with pytest.raises(RuntimeError, match='closed'):
        with stats:
            pass


def test_api__tables(param_file: Path):
    """Test ParamStats.stats_table()."""
    with ParamStats(param_file) as stats:
        param_stats = stats.stats()

    param_table = stats.stats_table(param_stats)
    assert str_contain_no_space(_exp_table, param_table)


def test_api__invalid_file_error(file_rgb_100cm_float: Path):
    """Test ParamStats() raises an error when the parameter file is invalid."""
    with pytest.raises(HomonimError, match='valid'):
        _ = ParamStats(file_rgb_100cm_float)


@pytest.mark.parametrize('param_file_str', ['param_file', 'param_file_tile_10x20'])
def test_cli(runner: CliRunner, param_file_str: str, request: FixtureRequest):
    """Test the CLI stats table output against its expected value."""
    param_file: Path = request.getfixturevalue(param_file_str)
    cli_str = f'stats {param_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert param_file.name in result.output

    assert str_contain_no_space(_exp_table, result.output)


def test_cli__out_file(tmp_path: Path, runner: CliRunner, param_file: Path):
    """Test the CLI --output option generates the correct values."""
    output_file = tmp_path.joinpath('stats.json')
    cli_str = f'stats {param_file} --output {output_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file) as f:
        stats_dict = json.load(f)

    param_file = str(param_file)
    assert param_file in stats_dict
    _test_vals(stats_dict[param_file])


def test_cli__mult_inputs(
    tmp_path: Path, runner: CliRunner, param_file: Path, param_file_tile_10x20: Path
):
    """Test the CLI with multiple input files."""
    output_file = tmp_path.joinpath('stats.json')
    cli_str = f'stats {param_file} {param_file_tile_10x20} --output {output_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file) as f:
        stats_dict = json.load(f)

    assert str(param_file) in stats_dict
    assert str(param_file_tile_10x20) in stats_dict
    for param_stats in stats_dict.values():
        _test_vals(param_stats)


def test_cli__invalid_file_error(runner: CliRunner, file_rgb_100cm_float):
    """Test the CLI prints an error when the parameter file is invalid."""
    cli_str = f'stats {file_rgb_100cm_float}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0
    assert 'valid' in result.output
