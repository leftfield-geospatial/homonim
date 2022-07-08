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
import multiprocessing

import numpy as np
import pytest
from typing import List, Dict, Union

from homonim.cli import cli
from homonim.compare import RasterCompare
from homonim.enums import ProcCrs
from tests.conftest import str_contain_nos


def _test_identical_compare_dict(res_list: List):
    """ Helper function to run tests on a compare results list, where the compare was between identical images. """
    assert (len(res_list) == 4)
    bands = [res_item['band'] for res_item in res_list]
    assert (bands[-1] == 'Mean')
    band_list = res_list[:-1]
    r2 = np.array([res_item['r2'] for res_item in band_list])
    RMSE = np.array([res_item['RMSE'] for res_item in band_list])
    rRMSE = np.array([res_item['rRMSE'] for res_item in band_list])
    N = np.array([res_item['N'] for res_item in band_list])
    assert (r2 == pytest.approx(1))
    assert (RMSE == pytest.approx(0))
    assert (rRMSE == pytest.approx(0))
    assert (N == N[0]).all()
    assert (res_list[-1]['r2'] == pytest.approx(1))
    assert (res_list[-1]['RMSE'] == pytest.approx(0))
    assert (res_list[-1]['rRMSE'] == pytest.approx(0))


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
    with RasterCompare(src_file, ref_file) as compare:
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
    with RasterCompare(src_file, ref_file, proc_crs=proc_crs) as compare:
        assert (compare.proc_crs == exp_proc_crs)
        res_dict = compare.compare()
    assert (len(res_dict) == 2)


def test_api__thread(float_45cm_src_file, float_100cm_ref_file):
    """ Test changing threads parameter gives same results. """
    with RasterCompare(float_45cm_src_file, float_100cm_ref_file) as raster_compare:
        res_list_single = raster_compare.compare(threads=1)
        res_list_mult = raster_compare.compare(threads=multiprocessing.cpu_count())
    assert (len(res_list_single) == 2)
    assert (len(res_list_mult) == 2)
    assert (res_list_mult == res_list_single)


def test_api__resampling(float_45cm_src_file, float_100cm_ref_file):
    """ Test changing up/downsampling parmaeter gives same similar but different results. """
    with RasterCompare(float_45cm_src_file, float_100cm_ref_file) as raster_compare:
        res_list_def = raster_compare.compare()
        res_list_lz = raster_compare.compare(downsampling='lanczos')
    assert (len(res_list_def) == 2)
    assert (len(res_list_lz) == 2)
    # test downsampling has changed r2, but not by too much
    for stats_dict_def, stats_dict_lz in zip(res_list_def, res_list_lz):
        assert stats_dict_def['r2'] != pytest.approx(stats_dict_lz['r2'], rel=1e-5)
        assert stats_dict_def['r2'] == pytest.approx(stats_dict_lz['r2'], rel=1e-1)


@pytest.mark.parametrize(
    'src_file, ref_file', [
        ('float_100cm_src_file', 'float_100cm_ref_file'),
        ('float_45cm_src_file', 'float_100cm_ref_file'),
        ('float_100cm_src_file', 'float_45cm_ref_file'),
    ]
)  # yapf:disable
def test_api__max_block_mem(src_file:str, ref_file:str, request):
    """ Test changing the number and shape of blocks (i.e. max_block_mem) gives same results. """
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    with RasterCompare(src_file, ref_file) as compare:
        stats_list_band = compare.compare(max_block_mem=100)    # compare by band
        stats_list_block = compare.compare(max_block_mem=2e-4)  # compare by small block
    assert (len(stats_list_band) == 2)
    assert (len(stats_list_block) == 2)
    # test band-based and block-based results are approx the same
    for stats_dict_band, stats_dict_block in zip(stats_list_band, stats_list_block):
        for k in stats_dict_band.keys():
            assert stats_dict_band[k] == pytest.approx(stats_dict_block[k], rel=1e-5)


def test_cli(runner, float_50cm_rgb_file, float_100cm_rgb_file):
    """ Test compare CLI with known outputs. """
    ref_file = float_100cm_rgb_file
    src_file = float_50cm_rgb_file

    cli_str = f'compare {src_file} {ref_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    res_str = """Band 1 1.000  0.000  0.000   144
Band 2 1.000  0.000  0.000   144
Band 3 1.000  0.000  0.000   144
Mean   1.000  0.000  0.000   144"""
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


def test_cli__adv_options(tmp_path, runner, float_50cm_src_file, float_100cm_ref_file):
    """ Test that the combined advanced CLI options affect comparison results as expected. """
    ref_file = float_100cm_ref_file
    src_file = float_50cm_src_file

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
    b1_dict_def = stats_list[0][str(src_file)][0]
    b1_dict_adv = stats_list[1][str(src_file)][1]
    # test that r2 with default options, and r2 with advanced options, are different, but not too different
    assert b1_dict_def['r2'] != pytest.approx(b1_dict_adv['r2'], 1e-5)
    assert b1_dict_def['r2'] == pytest.approx(b1_dict_adv['r2'], 1e-1)
