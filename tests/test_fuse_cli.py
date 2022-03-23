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

import pytest
import rasterio as rio
import os

from homonim import utils
from homonim.cli import cli
from homonim.enums import ProcCrs, Method
from homonim.fuse import RasterFuse


@pytest.mark.parametrize('method, kernel_shape', [
    (Method.gain, (1, 1)),
    (Method.gain_blk_offset, (1, 1)),
    (Method.gain_offset, (5, 5)),
])
def test_fuse(tmp_path, runner, float_100cm_rgb_file, float_50cm_rgb_file, method, kernel_shape):
    ref_file = float_100cm_rgb_file
    src_file = float_50cm_rgb_file
    post_fix = utils.create_homo_postfix(ProcCrs.ref, method, kernel_shape,
                                         RasterFuse.default_out_profile['driver'])
    homo_file = tmp_path.joinpath(src_file.stem + post_fix)
    cli_str = f'fuse -m {method.value} -k {kernel_shape[0]} {kernel_shape[1]} -od {tmp_path} {src_file} {ref_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (homo_file.exists())

    with rio.open(src_file, 'r') as src_ds, rio.open(homo_file, 'r') as out_ds:
        src_array = src_ds.read(indexes=src_ds.indexes)
        src_mask = src_ds.dataset_mask().astype('bool', copy=False)
        out_array = out_ds.read(indexes=out_ds.indexes)
        out_mask = out_ds.dataset_mask().astype('bool', copy=False)
        assert (out_mask == src_mask).all()
        assert (out_array[:, out_mask] == pytest.approx(src_array[:, src_mask], abs=.1))

def test_fuse_defaults(runner, float_100cm_rgb_file, float_50cm_rgb_file):
    ref_file = float_100cm_rgb_file
    src_file = float_50cm_rgb_file
    post_fix = utils.create_homo_postfix(ProcCrs.ref, Method.gain_blk_offset, (5, 5), RasterFuse.default_out_profile['driver'])
    homo_file = src_file.parent.joinpath(src_file.stem + post_fix)
    cli_str = f'fuse {src_file} {ref_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (homo_file.exists())


def test_file_exists(tmp_path, runner, float_100cm_ref_file, float_100cm_src_file):
    ref_file = float_100cm_ref_file
    src_file = float_100cm_src_file
    method = Method.gain_blk_offset
    kernel_shape = (3, 3)
    post_fix = utils.create_homo_postfix(ProcCrs.ref, method, kernel_shape, RasterFuse.default_out_profile['driver'])
    homo_file = tmp_path.joinpath(src_file.stem + post_fix)
    homo_file.touch()
    cli_str = (f'fuse -m {method.value} -k {kernel_shape[0]} {kernel_shape[1]} -od {tmp_path} {src_file} {ref_file}')
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ('FileExistsError' in result.output)

    param_file = utils.create_param_filename(homo_file)
    param_file.touch()
    os.remove(homo_file)
    cli_str = (f'fuse -m {method.value} -k {kernel_shape[0]} {kernel_shape[1]} -od {tmp_path} --param-image {src_file} '
               f'{ref_file}')
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ('FileExistsError' in result.output)

def test_overwrite(tmp_path, runner, float_100cm_ref_file, float_100cm_src_file):
    ref_file = float_100cm_ref_file
    src_file = float_100cm_src_file
    method = Method.gain_blk_offset
    kernel_shape = (3, 3)
    post_fix = utils.create_homo_postfix(ProcCrs.ref, method, kernel_shape, RasterFuse.default_out_profile['driver'])
    homo_file = tmp_path.joinpath(src_file.stem + post_fix)
    homo_file.touch()
    param_file = utils.create_param_filename(homo_file)
    param_file.touch()
    cli_str = (f'fuse -m {method.value} -k {kernel_shape[0]} {kernel_shape[1]} -od {tmp_path} --param-image '
               f'-ovw {src_file} {ref_file}')
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (homo_file.exists())
    assert (param_file.exists())


def test_compare(runner, float_100cm_ref_file, float_100cm_src_file):
    ref_file = float_100cm_ref_file
    src_file = float_100cm_src_file
    cli_str = (f'fuse --compare {src_file} {ref_file}')
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    src_cmp_str = """float_100cm_src.tif:

        r2   RMSE  rRMSE   N 
Band 1 1.00  0.00  0.00   144
Mean   1.00  0.00  0.00   144"""
    assert (src_cmp_str in result.output)

    homo_cmp_str = """float_100cm_src_HOMO_cREF_mGAIN-BLK-OFFSET_k5_5.tif:

        r2   RMSE  rRMSE   N 
Band 1 1.00  0.00  0.00   144
Mean   1.00  0.00  0.00   144"""
    assert (homo_cmp_str in result.output)

    sum_cmp_str = """File                         Mean r2  Mean RMSE  Mean rRMSE  Mean N
                                float_100cm_src.tif   1.00      0.00        0.00      144  
float_100cm_src_HOMO_cREF_mGAIN-BLK-OFFSET_k5_5.tif   1.00      0.00        0.00      144"""
    assert (sum_cmp_str in result.output)


@pytest.mark.parametrize('proc_crs', [ProcCrs.auto, ProcCrs.ref, ProcCrs.src])
def test_proc_crs(tmp_path, runner, float_100cm_ref_file, float_100cm_src_file, proc_crs):
    ref_file = float_100cm_ref_file
    src_file = float_100cm_src_file
    method = Method.gain_blk_offset
    kernel_shape = (3, 3)
    _res_proc_crs = ProcCrs.ref if proc_crs == ProcCrs.auto else proc_crs
    post_fix = utils.create_homo_postfix(_res_proc_crs, method, kernel_shape, RasterFuse.default_out_profile['driver'])
    homo_file = tmp_path.joinpath(src_file.stem + post_fix)
    cli_str = (f'fuse -m {method.value} -k {kernel_shape[0]} {kernel_shape[1]} -od {tmp_path} -pc {proc_crs.value} '
               f'{src_file} {ref_file}')
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (homo_file.exists())

    with rio.open(src_file, 'r') as src_ds, rio.open(homo_file, 'r') as out_ds:
        assert (out_ds.tags()['HOMO_PROC_CRS'] == _res_proc_crs.name)
        src_array = src_ds.read(indexes=src_ds.indexes)
        src_mask = src_ds.dataset_mask().astype('bool', copy=False)
        out_array = out_ds.read(indexes=out_ds.indexes)
        out_mask = out_ds.dataset_mask().astype('bool', copy=False)
        assert (out_mask == src_mask).all()
        assert (out_array[:, out_mask] == pytest.approx(src_array[:, src_mask], abs=1e-3))
