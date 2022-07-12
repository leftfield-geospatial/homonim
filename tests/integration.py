"""
    Homonim: Correction of aerial and satellite imagery to surface reflectance.
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

import pathlib
from typing import Tuple

import pytest
import rasterio as rio
from click.testing import CliRunner
from rasterio.features import shapes

from homonim import root_path, utils, RasterFuse, RasterCompare, ProcCrs, Model
from homonim.cli import cli


@pytest.fixture()
def modis_ref_file() -> pathlib.Path:
    return root_path.joinpath(r'tests/data/reference/MODIS-006-MCD43A4-2015_09_15_B143.tif')


@pytest.fixture()
def landsat_ref_file() -> pathlib.Path:
    return root_path.joinpath(r'tests/data/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.tif')


@pytest.fixture()
def s2_ref_file() -> pathlib.Path:
    return root_path.joinpath(
        r'tests/data/reference/COPERNICUS-S2-20151003T075826_20151003T082014_T35HKC_B432_Byte.tif'
    )


@pytest.fixture()
def landsat_src_file() -> pathlib.Path:
    return root_path.joinpath(r'tests/data/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.vrt')


@pytest.fixture()
def ngi_src_files() -> Tuple[pathlib.Path, ...]:
    source_root = root_path.joinpath('tests/data/source/')
    return tuple([fn for fn in source_root.glob('3324c_2015_*_RGB.tif')])


@pytest.fixture()
def ngi_src_file() -> pathlib.Path:
    return root_path.joinpath(r'tests/data/source/3324c_2015_1004_05_0182_RGB.tif')


@pytest.mark.parametrize(
    'src_files, ref_file, model, kernel_shape, proc_crs, mask_partial, exp_proc_crs', [
        ('ngi_src_files', 'modis_ref_file', Model.gain, (1, 1), ProcCrs.auto, False, ProcCrs.ref),
        ('ngi_src_files', 'landsat_ref_file', Model.gain_blk_offset, (5, 5), ProcCrs.auto, False, ProcCrs.ref),
        ('ngi_src_files', 's2_ref_file', Model.gain_offset, (15, 15), ProcCrs.auto, False, ProcCrs.ref),
        ('landsat_src_file', 's2_ref_file', Model.gain_blk_offset, (5, 5), ProcCrs.auto, False, ProcCrs.src),
        ('landsat_src_file', 's2_ref_file', Model.gain_blk_offset, (31, 31), ProcCrs.ref, False, ProcCrs.ref),
        ('ngi_src_files', 's2_ref_file', Model.gain_offset, (31, 31), ProcCrs.src, False, ProcCrs.src),
        ('ngi_src_files', 'modis_ref_file', Model.gain, (1, 1), ProcCrs.auto, True, ProcCrs.ref),
        ('ngi_src_files', 'landsat_ref_file', Model.gain_blk_offset, (5, 5), ProcCrs.auto, True, ProcCrs.ref),
        ('ngi_src_files', 's2_ref_file', Model.gain_offset, (15, 15), ProcCrs.auto, True, ProcCrs.ref),
        ('landsat_src_file', 's2_ref_file', Model.gain_blk_offset, (5, 5), ProcCrs.auto, True, ProcCrs.src),
        ('landsat_src_file', 's2_ref_file', Model.gain_blk_offset, (31, 31), ProcCrs.ref, True, ProcCrs.ref),
        ('ngi_src_files', 's2_ref_file', Model.gain_offset, (31, 31), ProcCrs.src, True, ProcCrs.src),
    ]
)
def test_fuse_compare(
    tmp_path: pathlib.Path, runner: CliRunner, src_files: str, ref_file: str, model: Model,
    kernel_shape: Tuple[int, int], proc_crs: ProcCrs, mask_partial: bool, exp_proc_crs: ProcCrs,
    request: pytest.FixtureRequest
):
    """ Additional integration tests using 'real' aerial and satellite imagery. """

    src_files = request.getfixturevalue(src_files)
    src_files: Tuple[pathlib.Path, ...] = src_files if isinstance(src_files, tuple) else (src_files, )
    ref_file: pathlib.Path = request.getfixturevalue(ref_file)
    src_file_str = ' '.join([str(fn) for fn in src_files])
    post_fix = utils.create_out_postfix(exp_proc_crs, model, kernel_shape, RasterFuse.create_out_profile()['driver'])
    corr_files = [tmp_path.joinpath(src_file.stem + post_fix) for src_file in src_files]

    cli_str = (
        f'fuse -m {model.value} -k {kernel_shape[0]} {kernel_shape[1]} -od {tmp_path} -pc {proc_crs.value}'
        f' -mbm 1 {src_file_str} {ref_file}'
    )
    if mask_partial:
        cli_str += ' --mask-partial'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert all([corr_file.exists() for corr_file in corr_files])

    for (src_file, corr_file) in zip(src_files, corr_files):
        # test corr_file improves on src_file (by comparing both to ref_file)
        with RasterCompare(src_file, ref_file, proc_crs=proc_crs) as src_compare:
            src_res = src_compare.compare()
        with RasterCompare(corr_file, ref_file, proc_crs=proc_crs) as corr_compare:
            corr_res = corr_compare.compare()
        for src_dict, corr_dict in zip(src_res, corr_res):
            assert (corr_dict['r2'] > src_dict['r2'])
            assert (corr_dict['rmse'] < src_dict['rmse'])
            assert (corr_dict['rrmse'] < src_dict['rrmse'])

        # test corr_file mask
        with rio.open(src_file, 'r') as src_ds, rio.open(corr_file, 'r') as corr_ds:
            src_mask = src_ds.dataset_mask().astype('bool', copy=False)
            corr_mask = corr_ds.dataset_mask().astype('bool', copy=False)
            if not mask_partial:
                # test src and homo masks are identical
                assert (corr_res[-1]['n'] == src_res[-1]['n'])
                assert (corr_mask == src_mask).all()
            else:
                # test homo mask is smaller than src mask
                assert (corr_res[-1]['n'] < src_res[-1]['n'])
                assert (corr_mask.sum() > 0)
                assert (corr_mask.sum() < src_mask.sum())
                assert (src_mask[corr_mask].all())
                # test homo mask consists of one blob
                corr_mask_shapes = list(shapes(corr_mask.astype('uint8', copy=False), mask=corr_mask, connectivity=8))
                assert (len(corr_mask_shapes) == 1)


##
