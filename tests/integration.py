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
from rasterio.features import shapes

from homonim import root_path
from homonim import utils
from homonim.cli import cli
from homonim.compare import RasterCompare
from homonim.enums import ProcCrs, Method
from homonim.fuse import RasterFuse


@pytest.fixture()
def modis_ref_file():
    return root_path.joinpath(r'data/test_example/reference/MODIS-006-MCD43A4-2015_09_15_B143.tif')


@pytest.fixture()
def landsat_ref_file():
    return root_path.joinpath(r'data/test_example/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.tif')


@pytest.fixture()
def s2_ref_file():
    return root_path.joinpath(
        r'data/test_example/reference/COPERNICUS-S2-20151003T075826_20151003T082014_T35HKC_B432_Byte.tif')


@pytest.fixture()
def landsat_src_file():
    return root_path.joinpath(r'data/test_example/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.vrt')


@pytest.fixture()
def ngi_src_files():
    source_root = root_path.joinpath('data/test_example/source/')
    return [fn for fn in source_root.glob('3324c_2015_*_RGB.tif')]


@pytest.fixture()
def ngi_src_file():
    return root_path.joinpath(r'data/test_example/source/3324c_2015_1004_05_0182_RGB.tif')



@pytest.mark.parametrize('src_files, ref_file, method, kernel_shape, proc_crs, mask_partial, exp_proc_crs', [
    ('ngi_src_files', 'modis_ref_file', Method.gain, (1, 1), ProcCrs.auto, False, ProcCrs.ref),
    ('ngi_src_files', 'landsat_ref_file', Method.gain_blk_offset, (5, 5), ProcCrs.auto, False, ProcCrs.ref),
    ('ngi_src_files', 's2_ref_file', Method.gain_offset, (15, 15), ProcCrs.auto, False, ProcCrs.ref),
    ('landsat_src_file', 's2_ref_file', Method.gain_blk_offset, (5, 5), ProcCrs.auto, False, ProcCrs.src),
    ('landsat_src_file', 's2_ref_file', Method.gain_blk_offset, (31, 31), ProcCrs.ref, False, ProcCrs.ref),
    ('ngi_src_files', 's2_ref_file', Method.gain_offset, (31, 31), ProcCrs.src, False, ProcCrs.src),

    ('ngi_src_files', 'modis_ref_file', Method.gain, (1, 1), ProcCrs.auto, True, ProcCrs.ref),
    ('ngi_src_files', 'landsat_ref_file', Method.gain_blk_offset, (5, 5), ProcCrs.auto, True, ProcCrs.ref),
    ('ngi_src_files', 's2_ref_file', Method.gain_offset, (15, 15), ProcCrs.auto, True, ProcCrs.ref),
    ('landsat_src_file', 's2_ref_file', Method.gain_blk_offset, (5, 5), ProcCrs.auto, True, ProcCrs.src),
    ('landsat_src_file', 's2_ref_file', Method.gain_blk_offset, (31, 31), ProcCrs.ref, True, ProcCrs.ref),
    ('ngi_src_files', 's2_ref_file', Method.gain_offset, (31, 31), ProcCrs.src, True, ProcCrs.src),
])
def test_fuse(tmp_path, runner, src_files, ref_file, method, kernel_shape, proc_crs, mask_partial, exp_proc_crs,
              request):
    """Additional integration tests using 'real' aerial and satellite imagery"""

    src_files = request.getfixturevalue(src_files)
    src_files = src_files if isinstance(src_files, list) else [src_files]
    ref_file = request.getfixturevalue(ref_file)
    src_file_str = ' '.join([str(fn) for fn in src_files])
    post_fix = utils.create_homo_postfix(exp_proc_crs, method, kernel_shape, RasterFuse.default_out_profile['driver'])
    homo_files = [tmp_path.joinpath(src_file.stem + post_fix) for src_file in src_files]

    cli_str = (f'fuse -m {method.value} -k {kernel_shape[0]} {kernel_shape[1]} -od {tmp_path} -pc {proc_crs.value}'
               f' -mbm 1 {src_file_str} {ref_file}')
    if mask_partial:
        cli_str += ' --mask-partial'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert all([homo_file.exists() for homo_file in homo_files])

    for (src_file, homo_file) in zip(src_files, homo_files):
        # test homo_file improves on src_file (by comparing both to ref_file)
        src_compare = RasterCompare(src_file, ref_file, proc_crs=proc_crs)
        src_res = src_compare.compare()
        homo_compare = RasterCompare(homo_file, ref_file, proc_crs=proc_crs)
        homo_res = homo_compare.compare()
        for band_key in src_res.keys():
            assert (homo_res[band_key]['r2'] > src_res[band_key]['r2'])
            assert (homo_res[band_key]['RMSE'] < src_res[band_key]['RMSE'])

       # test homo_file mask
        with rio.open(src_file, 'r') as src_ds, rio.open(homo_file, 'r') as homo_ds:
            src_mask = src_ds.dataset_mask().astype('bool', copy=False)
            homo_mask = homo_ds.dataset_mask().astype('bool', copy=False)
            if not mask_partial:
                # test src and homo masks are identical
                assert (homo_res['Mean']['N'] == src_res['Mean']['N'])
                assert (homo_mask == src_mask).all()
            else:
                # test homo mask is smaller than src mask
                assert (homo_res['Mean']['N'] < src_res['Mean']['N'])
                assert (homo_mask.sum() > 0)
                assert (homo_mask.sum() < src_mask.sum())
                assert (src_mask[homo_mask].all())
                # test homo mask consists of one blob
                homo_mask_shapes = list(shapes(homo_mask.astype('uint8', copy=False), mask=homo_mask, connectivity=8))
                assert (len(homo_mask_shapes) == 1)

##
