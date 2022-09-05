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
import multiprocessing
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pytest
from click.testing import CliRunner
from pytest import FixtureRequest

from homonim.cli import cli
from homonim.compare import RasterCompare
from homonim.enums import ProcCrs
from tests.conftest import str_contain_no_space


def _test_identical_compare_dict(res_dict: Dict, exp_len: int = 4):
    """ Helper function to run tests on a compare results list, where the compare was between identical images. """
    assert (len(res_dict) == exp_len)
    bands = list(res_dict.keys())
    assert (bands[-1] == 'Mean')
    band_list = list(res_dict.values())[:-1]

    r2 = np.array([res_item['r2'] for res_item in band_list])
    rmse = np.array([res_item['rmse'] for res_item in band_list])
    rrmse = np.array([res_item['rrmse'] for res_item in band_list])
    n = np.array([res_item['n'] for res_item in band_list])
    assert (r2 == pytest.approx(1))
    assert (n == n[0]).all()
    assert (rmse == pytest.approx(0))
    assert (rrmse == pytest.approx(0))
    assert (res_dict['Mean']['r2'] == pytest.approx(1))
    assert (res_dict['Mean']['rmse'] == pytest.approx(0))
    assert (res_dict['Mean']['rrmse'] == pytest.approx(0))


@pytest.mark.parametrize(
    'src_file, ref_file, src_bands, ref_bands, force, exp_bands', [
        ('file_rgb_50cm_float', 'file_rgb_100cm_float', None, None, False, (1, 2, 3)),
        ('file_rgb_100cm_float', 'file_rgb_50cm_float', (3, 2, 1), None, False, (3, 2, 1)),
        ('file_rgb_50cm_float', 'file_rgb_100cm_float', (2, 1), (3, 1, 2), False, (2, 1)),
        ('file_rgb_100cm_float', 'file_rgb_50cm_float', (2, 1), (3, 2, 1), True, (3, 2)),
    ]
)  # yapf: disable
def test_api(
    src_file: str, ref_file: str, src_bands: Tuple[int], ref_bands: Tuple[int], force: bool, exp_bands: Tuple[int],
    tmp_path: Path, request: FixtureRequest
):
    """ Test fusion with the src_bands and ref_bands parameters. """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    with RasterCompare(src_file, ref_file, src_bands=src_bands, ref_bands=ref_bands, force=force) as raster_compare:
        res_dict = raster_compare.compare()
        assert raster_compare.ref_bands == exp_bands
    if not force:
        _test_identical_compare_dict(res_dict, len(exp_bands) + 1)


def test_api__thread(src_file_45cm_float, ref_file_100cm_float):
    """ Test compasison results remain the same with different `threads` configurations. """
    with RasterCompare(src_file_45cm_float, ref_file_100cm_float) as raster_compare:
        res_dict_single = raster_compare.compare(threads=1)
        res_dict_mult = raster_compare.compare(threads=multiprocessing.cpu_count())
    assert (len(res_dict_single) == 2)
    assert (len(res_dict_mult) == 2)
    assert (res_dict_mult == res_dict_single)


@pytest.mark.parametrize(
    'src_file, ref_file, proc_crs, config', [
        ('src_file_50cm_float', 'ref_file_100cm_float', ProcCrs.ref, dict(downsampling='lanczos')),
        ('src_file_50cm_float', 'ref_file_100cm_float', ProcCrs.src, dict(upsampling='lanczos')),
    ]
)  # yapf:disable
def test_api__resampling(src_file: str, ref_file: str, proc_crs: ProcCrs, config: Dict, request: FixtureRequest):
    """ Test non-default resampling parameters give similar but different results to the defaults. """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    with RasterCompare(src_file, ref_file, proc_crs=proc_crs) as raster_compare:
        res_dict_def = raster_compare.compare()  # default configuration results
        res_dict_lz = raster_compare.compare(**config)  # non-default configuration
    assert (len(res_dict_def) == 2)
    assert (len(res_dict_lz) == 2)
    # test non-default r2 is similar but different to default r2
    for band in res_dict_def.keys():
        assert res_dict_def[band]['r2'] != pytest.approx(res_dict_lz[band]['r2'], rel=1e-5)
        assert res_dict_def[band]['r2'] == pytest.approx(res_dict_lz[band]['r2'], rel=1e-1)


@pytest.mark.parametrize(
    'src_file, ref_file', [
        ('src_file_100cm_float', 'ref_file_100cm_float'),
        ('src_file_45cm_float', 'ref_file_100cm_float'),
        ('src_file_100cm_float', 'ref_file_45cm_float'),
    ]
)  # yapf:disable
def test_api__max_block_mem(src_file: str, ref_file: str, request: FixtureRequest):
    """ Test changing the number and shape of blocks (i.e. max_block_mem) gives the same comparison results. """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    with RasterCompare(src_file, ref_file) as compare:
        stats_dict_band = compare.compare(max_block_mem=100)  # compare by band
        stats_dict_block = compare.compare(max_block_mem=2e-4)  # compare by small block
    assert (len(stats_dict_band) == 2)
    assert (len(stats_dict_block) == 2)
    # test band-based and block-based results are approx the same
    for band in stats_dict_band.keys():
        for k in stats_dict_band[band].keys():
            assert stats_dict_band[band][k] == pytest.approx(stats_dict_block[band][k], rel=1e-5)


def test_api__proc_crs(
    src_file_45cm_float, ref_file_100cm_float, src_file_100cm_float, ref_file_45cm_float
):
    """
    Test comparison of high res source with low res reference (proc_crs=ref) gives approx same results as comparison of
    low res source with high res reference (proc_crs=src).
    """
    with RasterCompare(src_file_45cm_float, ref_file_100cm_float, proc_crs=ProcCrs.ref) as raster_compare:
        stats_dict_ref = raster_compare.compare()  # compare by band
        assert (raster_compare.proc_crs == ProcCrs.ref)
    assert (len(stats_dict_ref) == 2)
    with RasterCompare(src_file_100cm_float, ref_file_45cm_float, proc_crs=ProcCrs.src) as raster_compare:
        stats_dict_src = raster_compare.compare()  # compare by band
        assert (raster_compare.proc_crs == ProcCrs.src)
    assert (len(stats_dict_src) == 2)
    # test ProcCrs.ref and ProcCrs.src results are approx the same
    for band in stats_dict_ref.keys():
        for k in stats_dict_ref[band].keys():
            assert stats_dict_ref[band][k] == pytest.approx(stats_dict_src[band][k], rel=1e-3)


def test_cli(runner: CliRunner, file_rgb_50cm_float, file_rgb_100cm_float):
    """ Test compare CLI with known outputs. """
    ref_file = file_rgb_100cm_float
    src_file = file_rgb_50cm_float

    cli_str = f'compare {src_file} {ref_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    res_str = """Ref. band 1 1.000  0.000  0.000   144
Ref. band 2 1.000  0.000  0.000   144
Ref. band 3 1.000  0.000  0.000   144
Mean   1.000  0.000  0.000   144"""
    assert (str_contain_no_space(res_str, result.output))


def test_cli__output_file(tmp_path: Path, runner: CliRunner, file_rgb_50cm_float, file_rgb_100cm_float):
    """ Test compare CLI generated json file. """
    ref_file = file_rgb_100cm_float
    src_file = file_rgb_50cm_float

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


def test_cli__mult_inputs(tmp_path: Path, runner: CliRunner, file_rgb_50cm_float, file_rgb_100cm_float):
    """ Test compare CLI with multiple src files. """
    ref_file = file_rgb_100cm_float
    src_file = file_rgb_50cm_float

    output_file = tmp_path.joinpath('compare.json')
    cli_str = f'compare {src_file} {src_file} {ref_file} --output {output_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (output_file.exists())

    with open(output_file) as f:
        stats_dict = json.load(f)

    src_file = str(src_file)
    assert (src_file in stats_dict)


def test_cli__adv_options(tmp_path: Path, runner: CliRunner, src_file_50cm_float, ref_file_100cm_float):
    """ Test that the combined advanced CLI options affect comparison results as expected. """
    ref_file = ref_file_100cm_float
    src_file = src_file_50cm_float

    # run a comparison with default advanced options, and with specified advanced options, then compare results
    out_file_def = tmp_path.joinpath('compare_defaults.json')
    out_file_adv = tmp_path.joinpath('compare_adv.json')
    cli_str_def = f'compare {src_file} {ref_file} --output {out_file_def}'
    cli_str_adv = f"""compare --threads 1 --max-block-mem 1e-3 --downsampling bilinear -pc ref {src_file} {ref_file}  
    --output {out_file_adv}"""
    stats_list = []
    for cli_str, out_file in zip([cli_str_def, cli_str_adv], [out_file_def, out_file_adv]):
        result = runner.invoke(cli, cli_str.split())
        assert (result.exit_code == 0)
        assert (out_file.exists())
        with open(out_file) as f:
            stats_dict = json.load(f)
            assert (str(src_file) in stats_dict)
            stats_list.append(stats_dict)
    b1_dict_def = stats_list[0][str(src_file)]['Mean']
    b1_dict_adv = stats_list[1][str(src_file)]['Mean']
    # test that r2 with default options, and r2 with advanced options, are different, but not too different
    assert b1_dict_def['r2'] != pytest.approx(b1_dict_adv['r2'], 1e-5)
    assert b1_dict_def['r2'] == pytest.approx(b1_dict_adv['r2'], 1e-1)


@pytest.mark.parametrize(
    'src_bands, ref_bands, force, exp_bands', [
        ((3, 2, 1), None, False, (3, 2, 1)),
        ((2, 1), (3, 1, 2), False, (2, 1)),
        ((2, 1), (3, 2, 1), True, (3, 2)),
    ]
)  # yapf: disable
def test_cli_src_ref_bands(
    src_bands: Tuple[int], ref_bands: Tuple[int], force: bool, exp_bands: Tuple[int], file_rgb_50cm_float,
    file_rgb_100cm_float, tmp_path: Path, runner: CliRunner,
):
    """ Test compare with --src_band, --ref_band and --force-match parameters. """
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
    assert (result.exit_code == 0)
    assert (out_file.exists())
    with open(out_file) as f:
        stats_dict = json.load(f)
    assert str(src_file) in stats_dict
    stats_dict = stats_dict[str(src_file)]
    exp_band_names = [f'Ref. band {bi}' for bi in exp_bands] + ['Mean']
    assert list(stats_dict.keys()) == exp_band_names
    if not force:
        _test_identical_compare_dict(stats_dict, len(exp_bands) + 1)

