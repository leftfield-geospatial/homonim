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

import numpy as np
import pytest
from click.testing import CliRunner
from pytest import FixtureRequest

from homonim.cli import cli
from homonim.compare import RasterCompare
from homonim.enums import ProcCrs
from tests.conftest import str_contain_no_space


def _test_identical_compare_dict(stats_dict: dict, exp_len: int = 4):
    """Test a compare results dictionary, where the compare was between identical
    images.
    """
    assert len(stats_dict) == exp_len
    bands = list(stats_dict.keys())
    assert bands[-1] == 'Mean'
    band_list = list(stats_dict.values())

    r2 = np.array([res_item['r2'] for res_item in band_list])
    rmse = np.array([res_item['rmse'] for res_item in band_list])
    rrmse = np.array([res_item['rrmse'] for res_item in band_list])
    n = np.array([res_item['n'] for res_item in band_list])
    assert r2 == pytest.approx(1)
    assert (n == n[0]).all()
    assert rmse == pytest.approx(0)
    assert rrmse == pytest.approx(0)


def test_api(file_rgb_50cm_float: Path, file_rgb_100cm_float: Path):
    """Test comparison results with default parameter values."""
    with RasterCompare(file_rgb_50cm_float, file_rgb_100cm_float) as raster_compare:
        stats_dict = raster_compare.process()
    _test_identical_compare_dict(stats_dict)


@pytest.mark.parametrize(
    'proc_crs, config',
    [
        (ProcCrs.ref, dict(downsampling='lanczos')),
        (ProcCrs.src, dict(upsampling='lanczos')),
    ],
)
def test_api__resampling(
    src_file_50cm_float: Path,
    ref_file_100cm_float: Path,
    proc_crs: ProcCrs,
    config: dict,
    request: FixtureRequest,
):
    """Test different resampling types give similar but not identical comparison
    results.
    """
    with RasterCompare(
        src_file_50cm_float, ref_file_100cm_float, proc_crs=proc_crs
    ) as raster_compare:
        # default resampling results
        stats_dict_def = raster_compare.process()
        # non-default resampling results
        stats_dict_res = raster_compare.process(**config)

    band = 'Mean'
    assert stats_dict_def[band]['r2'] != stats_dict_res[band]['r2']
    assert stats_dict_res[band]['r2'] == pytest.approx(
        stats_dict_def[band]['r2'], rel=0.1
    )


@pytest.mark.parametrize(
    'src_file, ref_file',
    [
        ('src_file_45cm_float', 'ref_file_100cm_float'),
        ('src_file_100cm_float', 'ref_file_45cm_float'),
    ],
)
def test_api__max_block_mem(src_file: str, ref_file: str, request: FixtureRequest):
    """Test different max_block_mem values give similar but not identical comparison
    results.
    """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    with RasterCompare(src_file, ref_file) as compare:
        # compare by band
        stats_dict_band = compare.process(threads=1, max_block_mem=0)
        # compare by small block
        stats_dict_block = compare.process(threads=1, max_block_mem=2e-4)

    assert stats_dict_block != stats_dict_band
    band = 'Mean'
    for k in stats_dict_band[band].keys():
        assert stats_dict_block[band][k] == pytest.approx(
            stats_dict_band[band][k], rel=1e-5
        )


def test_api__proc_crs(src_file_50cm_float: Path, ref_file_100cm_float: Path):
    """Test the proc_crs parameter affects comparison results."""
    with RasterCompare(
        src_file_50cm_float, ref_file_100cm_float, proc_crs=ProcCrs.ref
    ) as compare:
        stats_dict_ref = compare.process()

    with RasterCompare(
        src_file_50cm_float, ref_file_100cm_float, proc_crs=ProcCrs.src
    ) as compare:
        stats_dict_src = compare.process()

    _test_identical_compare_dict(stats_dict_ref, exp_len=2)
    assert stats_dict_src != stats_dict_ref


# ruff: ignore[E501]
@pytest.mark.parametrize(
    'src_file, ref_file, src_bands, ref_bands, force, exp_bands',
    [
        ('file_rgb_50cm_float', 'file_rgb_100cm_float', None, None, False, (1, 2, 3)),
        ('file_rgb_100cm_float', 'file_rgb_50cm_float', (3, 2, 1), None, False, (3, 2, 1)),
        ('file_rgb_50cm_float', 'file_rgb_100cm_float', (2, 1), (3, 1, 2), False, (2, 1)),
        ('file_rgb_100cm_float', 'file_rgb_50cm_float', (2, 1), (3, 2, 1), True, (3, 2)),
    ]
)  # fmt: skip
def test_api_src_ref_bands(
    request: FixtureRequest,
    src_file: str,
    ref_file: str,
    src_bands: tuple[int],
    ref_bands: tuple[int],
    force: bool,
    exp_bands: tuple[int],
):
    """Test comparison results and band matching with different src_bands, ref_bands
    and force values.
    """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    with RasterCompare(
        src_file, ref_file, src_bands=src_bands, ref_bands=ref_bands, force=force
    ) as raster_compare:
        stats_dict = raster_compare.process()
        assert raster_compare.ref_bands == exp_bands
    if not force:
        _test_identical_compare_dict(stats_dict, len(exp_bands) + 1)


def test_cli(runner: CliRunner, file_rgb_50cm_float, file_rgb_100cm_float):
    """Test the compare CLI report against known values."""
    ref_file = file_rgb_100cm_float
    src_file = file_rgb_50cm_float

    cli_str = f'compare {src_file} {ref_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0

    res_str = """Ref. band 1 1.000  0.000  0.000   144
Ref. band 2 1.000  0.000  0.000   144
Ref. band 3 1.000  0.000  0.000   144
Mean   1.000  0.000  0.000   144"""
    assert str_contain_no_space(res_str, result.output)


def test_cli__output_file(
    tmp_path: Path, runner: CliRunner, file_rgb_50cm_float, file_rgb_100cm_float
):
    """Test the compare CLI generated JSON file."""
    ref_file = file_rgb_100cm_float
    src_file = file_rgb_50cm_float

    output_file = tmp_path.joinpath('compare.json')
    cli_str = f'compare {src_file} {ref_file} --output {output_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file) as f:
        stats_dict = json.load(f)
    src_file = str(src_file)
    assert src_file in stats_dict
    _test_identical_compare_dict(stats_dict[src_file])


def test_cli__mult_inputs(
    tmp_path: Path, runner: CliRunner, file_rgb_50cm_float, file_rgb_100cm_float
):
    """Test the compare CLI with multiple source files."""
    ref_file = file_rgb_100cm_float
    src_files = (file_rgb_50cm_float, file_rgb_100cm_float)

    output_file = tmp_path.joinpath('compare.json')
    cli_str = f'compare {src_files[0]} {src_files[1]} {ref_file} --output {output_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file) as f:
        stats_dict = json.load(f)
    assert all(str(sf) in stats_dict for sf in src_files)


@pytest.mark.parametrize(
    'proc_crs, config',
    [
        (ProcCrs.ref, dict(downsampling='lanczos')),
        (ProcCrs.src, dict(upsampling='lanczos')),
    ],
)
def test_cli__resampling(
    tmp_path: Path,
    runner: CliRunner,
    src_file_50cm_float: Path,
    ref_file_100cm_float: Path,
    proc_crs: ProcCrs,
    config: dict,
):
    """Test different resampling types give similar but not identical comparison
    results.
    """
    ref_file = ref_file_100cm_float
    src_file = src_file_50cm_float
    out_file = tmp_path.joinpath('compare.json')
    # default resampling
    cli_str_def = (
        f'compare {src_file} {ref_file} --output {out_file} --proc-crs {proc_crs}'
    )
    # non-default resampling
    res_str = ''.join([f' --{k} {v}' for k, v in config.items()])
    cli_str_res = cli_str_def + res_str

    stats_list = []
    for cli_str in [cli_str_def, cli_str_res]:
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0
        assert out_file.exists()
        with open(out_file) as f:
            stats_dict = json.load(f)
        assert str(src_file) in stats_dict
        stats_list.append(stats_dict[str(src_file)]['Mean'])
        out_file.unlink()

    assert stats_list[1]['r2'] != stats_list[0]['r2']
    assert stats_list[1]['r2'] == pytest.approx(stats_list[0]['r2'], rel=0.1)


def test_cli__max_block_mem(
    tmp_path: Path,
    runner: CliRunner,
    src_file_45cm_float: Path,
    ref_file_100cm_float: Path,
):
    """Test different --max-block-mem values give similar but not identical comparison
    results.
    """
    ref_file = ref_file_100cm_float
    src_file = src_file_45cm_float
    out_file = tmp_path.joinpath('compare.json')

    stats_list = []
    for max_block_mem in [0, 2e-4]:
        cli_str = (
            f'compare {src_file} {ref_file} --output {out_file} --threads 1 '
            f'--max-block-mem {max_block_mem}'
        )
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0
        assert out_file.exists()
        with open(out_file) as f:
            stats_dict = json.load(f)
        stats_list.append(stats_dict[str(src_file)]['Mean'])
        out_file.unlink()

    # test band-based and block-based results are similar but not the same
    assert stats_list[1] != stats_list[0]
    for k in stats_list[0].keys():
        assert stats_list[1][k] == pytest.approx(stats_list[0][k], rel=1e-5)


def test_cli__proc_crs(
    tmp_path: Path,
    runner: CliRunner,
    src_file_50cm_float: Path,
    ref_file_100cm_float: Path,
):
    """Test the --proc-crs option affects comparison results."""
    ref_file = ref_file_100cm_float
    src_file = src_file_50cm_float
    out_file = tmp_path.joinpath('compare.json')

    stats_list = []
    for proc_crs in [ProcCrs.ref, ProcCrs.src]:
        cli_str = (
            f'compare {src_file} {ref_file} --output {out_file} --proc-crs {proc_crs}'
        )
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0
        assert out_file.exists()
        with open(out_file) as f:
            stats_dict = json.load(f)
        stats_list.append(stats_dict[str(src_file)])
        out_file.unlink()

    _test_identical_compare_dict(stats_list[0], exp_len=2)
    assert len(stats_list[1]) == 2
    assert stats_list[1] != stats_list[0]


@pytest.mark.parametrize(
    'src_bands, ref_bands, force, exp_bands',
    [
        ((3, 2, 1), None, False, (3, 2, 1)),
        ((2, 1), (3, 1, 2), False, (2, 1)),
        ((2, 1), (3, 2, 1), True, (3, 2)),
    ],
)
def test_cli__src_ref_bands(
    src_bands: tuple[int],
    ref_bands: tuple[int],
    force: bool,
    exp_bands: tuple[int],
    file_rgb_50cm_float,
    file_rgb_100cm_float,
    tmp_path: Path,
    runner: CliRunner,
):
    """Test the compare CLI band matching and results with --src-band, --ref-band and
    --force-match parameters.
    """
    src_file = file_rgb_50cm_float
    ref_file = file_rgb_100cm_float
    out_file = tmp_path.joinpath('results.json')
    cli_str = f'compare {src_file} {ref_file} -op {out_file}'
    if src_bands:
        cli_str += ''.join([' -sb ' + str(bi) for bi in src_bands])
    if ref_bands:
        cli_str += ''.join([' -rb ' + str(bi) for bi in ref_bands])
    if force:
        cli_str += ' -f'

    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert out_file.exists()

    with open(out_file) as f:
        stats_dict = json.load(f)
    assert str(src_file) in stats_dict
    stats_dict = stats_dict[str(src_file)]

    exp_band_names = [f'Ref. band {bi}' for bi in exp_bands] + ['Mean']
    assert list(stats_dict.keys()) == exp_band_names
    if not force:
        _test_identical_compare_dict(stats_dict, len(exp_bands) + 1)
