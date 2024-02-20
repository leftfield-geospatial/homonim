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

import os
from pathlib import Path
from typing import Tuple

import pytest
import rasterio as rio
import yaml
from click.testing import CliRunner
from rasterio.warp import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.enums import MaskFlags

from homonim import utils
from homonim.cli import cli
from homonim.enums import ProcCrs, Model
from homonim.fuse import RasterFuse
from tests.conftest import str_contain_no_space, FuseCliParams


@pytest.mark.parametrize(
    'model, kernel_shape', [
        (Model.gain, (1, 1)),
        (Model.gain_blk_offset, (1, 1)),
        (Model.gain_offset, (5, 5)),
    ]
)  # yapf: disable
def test_fuse(
    tmp_path: Path, runner: CliRunner, file_rgb_100cm_float, file_rgb_50cm_float, model: Model,
    kernel_shape: Tuple[int, int],
):
    """ Test fuse cli output with different methods and kernel shapes. """
    ref_file = file_rgb_100cm_float
    src_file = file_rgb_50cm_float
    post_fix = utils.create_out_postfix(ProcCrs.ref, model, kernel_shape, RasterFuse.create_out_profile()['driver'])
    corr_file = tmp_path.joinpath(src_file.stem + post_fix)
    cli_str = f'fuse -m {model.value} -k {kernel_shape[0]} {kernel_shape[1]} -od {tmp_path} {src_file} {ref_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (corr_file.exists())

    with rio.open(src_file, 'r') as src_ds, rio.open(corr_file, 'r') as out_ds:
        assert (out_ds.tags()['FUSE_MODEL'] == model.name)
        assert (out_ds.tags()['FUSE_KERNEL_SHAPE'] == str(kernel_shape))

        src_array = src_ds.read(indexes=src_ds.indexes)
        src_mask = src_ds.dataset_mask().astype('bool', copy=False)
        out_array = out_ds.read(indexes=out_ds.indexes)
        out_mask = out_ds.dataset_mask().astype('bool', copy=False)
        assert (out_mask == src_mask).all()
        assert (out_array[:, out_mask] == pytest.approx(src_array[:, src_mask], abs=.1))


def test_fuse_defaults(runner: CliRunner, default_fuse_cli_params: FuseCliParams):
    """ Test fuse cli works without model or kernel shape arguments. """
    result = runner.invoke(cli, default_fuse_cli_params.cli_str.split())
    assert (result.exit_code == 0)
    assert (default_fuse_cli_params.corr_file.exists())


def test_method_error(runner: CliRunner, default_fuse_cli_params: FuseCliParams):
    """ Test unknown model generates an error. """
    cli_str = default_fuse_cli_params.cli_str + ' -m unk'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ("Invalid value for '-m' / '--model'" in result.output)


@pytest.mark.parametrize('bad_kernel_shape', [(0, 0), (2, 3), (3, 2)])
def test_kernel_shape_error(runner: CliRunner, default_fuse_cli_params: FuseCliParams, bad_kernel_shape):
    """ Test bad kernel shape generates an error. """
    cli_str = default_fuse_cli_params.cli_str + f' -k {bad_kernel_shape[0]} {bad_kernel_shape[1]}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ("kernel_shape" in result.output)


def test_file_exists_error(runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test that attempting to overwrite an existing output file generates an error. """
    basic_fuse_cli_params.corr_file.touch()
    cli_str = basic_fuse_cli_params.cli_str
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ('FileExistsError' in result.output)

    os.remove(basic_fuse_cli_params.corr_file)
    basic_fuse_cli_params.param_file.touch()
    cli_str = basic_fuse_cli_params.cli_str + ' --param-image'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ('FileExistsError' in result.output)


def test_overwrite(runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test overwriting existing output file(s) with -o. """
    basic_fuse_cli_params.corr_file.touch()
    basic_fuse_cli_params.param_file.touch()
    cli_str = basic_fuse_cli_params.cli_str + ' --param-image -o'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (basic_fuse_cli_params.corr_file.exists())
    assert (basic_fuse_cli_params.param_file.exists())


def test_compare(runner: CliRunner, ref_file_100cm_float, src_file_100cm_float):
    """ Test --compare, in flag and value configurations, against expected output. """
    ref_file = ref_file_100cm_float
    src_file = src_file_100cm_float
    # test --compare in flag (no value), and value configuration
    cli_strs = [
        f'fuse  {src_file} {ref_file} --compare', f'fuse {src_file} {ref_file} --compare {ref_file_100cm_float} -o'
    ]
    for cli_str in cli_strs:
        result = runner.invoke(cli, cli_str.split())
        assert (result.exit_code == 0)
        src_cmp_str = """float_100cm_src.tif:
           Band    r²   RMSE   rRMSE   N
    ----------- ----- ------ ------- ---
    Ref. band 1 1.000  0.000   0.000 144
           Mean 1.000  0.000   0.000 144"""
        assert (str_contain_no_space(src_cmp_str, result.output))

        corr_cmp_str = """float_100cm_src_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif:
           Band      r²   RMSE   rRMSE   N
    ----------- ----- ------ ------- ---
    Ref. band 1 1.000  0.000   0.000 144
           Mean 1.000  0.000   0.000 144"""
        assert (str_contain_no_space(corr_cmp_str, result.output))

        sum_cmp_str = """File    r²   RMSE   rRMSE   N
    --------------------------------------------------- ----- ------ ------- ---
                                    float_100cm_src.tif 1.000  0.000   0.000 144
    float_100cm_src_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif 1.000  0.000   0.000 144"""
        assert (str_contain_no_space(sum_cmp_str, result.output))


def test_compare_file_exists_error(runner: CliRunner, ref_file_100cm_float, src_file_100cm_float):
    """ Test --compare raises an exception when the specified file does not exist. """
    ref_file = ref_file_100cm_float
    src_file = src_file_100cm_float
    # test --compare in flag (no value), and value configurayion
    cli_str = f'fuse  {src_file} {ref_file} --compare unknown.tif'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ('No such file or directory' in result.output)


@pytest.mark.parametrize('proc_crs', [ProcCrs.auto, ProcCrs.ref, ProcCrs.src])
def test_proc_crs(
    tmp_path: Path, runner: CliRunner, ref_file_100cm_float, src_file_100cm_float, proc_crs: ProcCrs,
):
    """ Test valid --proc-crs settings generate an output with correct metadata. """
    ref_file = ref_file_100cm_float
    src_file = src_file_100cm_float
    model = Model.gain_blk_offset
    kernel_shape = (3, 3)
    res_proc_crs = ProcCrs.ref if proc_crs == ProcCrs.auto else proc_crs
    post_fix = utils.create_out_postfix(res_proc_crs, model, kernel_shape, RasterFuse.create_out_profile()['driver'])
    corr_file = tmp_path.joinpath(src_file.stem + post_fix)
    cli_str = (
        f'fuse -m {model.value} -k {kernel_shape[0]} {kernel_shape[1]} -od {tmp_path} -pc {proc_crs.value} '
        f'{src_file} {ref_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (corr_file.exists())

    with rio.open(corr_file, 'r') as out_ds:
        assert (out_ds.tags()['FUSE_PROC_CRS'] == res_proc_crs.name)


def test_conf_file(tmp_path: Path, runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test passing a configuration file results in a correctly configured output. """
    # create test configuration file
    conf_dict = dict(
        mask_partial=True, param_image=True, dtype='uint8', nodata=0, creation_options=dict(compress='lzw')
    )
    conf_file = tmp_path.joinpath('conf.yaml')
    with open(conf_file, 'w') as f:
        yaml.dump(conf_dict, f)

    cli_str = basic_fuse_cli_params.cli_str + f' -c {conf_file}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (basic_fuse_cli_params.corr_file.exists())
    assert (basic_fuse_cli_params.param_file.exists())  # test param_image==True

    with rio.open(basic_fuse_cli_params.src_file, 'r') as src_ds:
        with rio.open(basic_fuse_cli_params.corr_file, 'r') as out_ds:
            # test nodata, dtype and creation_options
            assert (out_ds.nodata == conf_dict['nodata'])
            assert (out_ds.dtypes[0] == conf_dict['dtype'])
            assert (out_ds.profile['compress'] == conf_dict['creation_options']['compress'])
            # test mask_partial==True
            src_mask = src_ds.dataset_mask().astype('bool', copy=False)
            out_mask = out_ds.dataset_mask().astype('bool', copy=False)
            assert (src_mask[out_mask].all())
            assert (src_mask.sum() > out_mask.sum())
            # test proc_crs
            assert (out_ds.tags()['FUSE_PROC_CRS'] == basic_fuse_cli_params.proc_crs.name)


def test_param_image(runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test --param-image. """
    # test that cli without --param-image generates no parameter image
    cli_str = basic_fuse_cli_params.cli_str
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (basic_fuse_cli_params.corr_file.exists())
    assert (not basic_fuse_cli_params.param_file.exists())

    # test --param-image generates a valid parameter image
    cli_str = basic_fuse_cli_params.cli_str + ' --param-image -o'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (basic_fuse_cli_params.corr_file.exists())
    assert (basic_fuse_cli_params.param_file.exists())
    utils.validate_param_image(basic_fuse_cli_params.param_file)


def test_mask_partial(runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test --mask-partial. """
    cli_str = basic_fuse_cli_params.cli_str + ' --mask-partial'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (basic_fuse_cli_params.corr_file.exists())

    with rio.open(basic_fuse_cli_params.src_file, 'r') as src_ds:
        with rio.open(basic_fuse_cli_params.corr_file, 'r') as out_ds:
            # test that the output mask is contained by and smaller than the src mask
            src_mask = src_ds.dataset_mask().astype('bool', copy=False)
            out_mask = out_ds.dataset_mask().astype('bool', copy=False)
            assert (src_mask[out_mask].all())
            assert (src_mask.sum() > out_mask.sum())


def test_threads(runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test --threads. """
    cli_str = basic_fuse_cli_params.cli_str + ' --threads 1'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (basic_fuse_cli_params.corr_file.exists())


def test_max_block_mem(runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test --max-block-mem. """
    cli_str = basic_fuse_cli_params.cli_str + ' -mbm 1e-4'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (basic_fuse_cli_params.corr_file.exists())

    # test that max_block_mem too small raises a BlockSizeError
    cli_str = basic_fuse_cli_params.cli_str + ' -o -mbm 1e-6'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ('BlockSizeError' in result.output)


@pytest.mark.parametrize('upsampling', [r.name for r in rio.warp.SUPPORTED_RESAMPLING])
def test_upsampling(runner: CliRunner, basic_fuse_cli_params: FuseCliParams, upsampling: Resampling):
    """ Test --upsampling with valid values generates output with correct metadata. """
    cli_str = basic_fuse_cli_params.cli_str + f' --upsampling {upsampling}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (basic_fuse_cli_params.corr_file.exists())
    with rio.open(basic_fuse_cli_params.corr_file, 'r') as out_ds:
        tags_dict = out_ds.tags()
        assert ('FUSE_UPSAMPLING' in tags_dict)
        assert (yaml.safe_load(tags_dict['FUSE_UPSAMPLING']) == upsampling)


@pytest.mark.parametrize('downsampling', [r.name for r in rio.warp.SUPPORTED_RESAMPLING])
def test_downsampling(runner: CliRunner, basic_fuse_cli_params: FuseCliParams, downsampling: Resampling):
    """ Test --downsampling with valid values generates output with correct metadata. """
    cli_str = basic_fuse_cli_params.cli_str + f' --downsampling {downsampling}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (basic_fuse_cli_params.corr_file.exists())
    with rio.open(basic_fuse_cli_params.corr_file, 'r') as out_ds:
        tags_dict = out_ds.tags()
        assert ('FUSE_DOWNSAMPLING' in tags_dict)
        assert (yaml.safe_load(tags_dict['FUSE_DOWNSAMPLING']) == downsampling)


def test_upsampling_error(runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test --upsampling with bad value raises an error. """
    cli_str = basic_fuse_cli_params.cli_str + f' --upsampling unknown'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ("Invalid value for '-us' / '--upsampling'" in result.output)


def test_downsampling_error(runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test --downsampling with bad value raises an error. """
    cli_str = basic_fuse_cli_params.cli_str + f' --downsampling unknown'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ("Invalid value for '-ds' / '--downsampling'" in result.output)


@pytest.mark.parametrize('r2_inpaint_thresh', [0, 0.5, 1])
def test_r2_inpaint_thresh(runner: CliRunner, basic_fuse_cli_params: FuseCliParams, r2_inpaint_thresh: float):
    """ Test --r2-inpaint-thresh generates an output with correct metadata. """
    cli_str = basic_fuse_cli_params.cli_str + f' --r2-inpaint-thresh {r2_inpaint_thresh}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (basic_fuse_cli_params.corr_file.exists())
    with rio.open(basic_fuse_cli_params.corr_file, 'r') as out_ds:
        tags_dict = out_ds.tags()
        assert ('FUSE_R2_INPAINT_THRESH' in tags_dict)
        assert (yaml.safe_load(tags_dict['FUSE_R2_INPAINT_THRESH']) == r2_inpaint_thresh)


@pytest.mark.parametrize('bad_r2_inpaint_thresh', [-1, 2])
def test_r2_inpaint_thresh_error(runner: CliRunner, basic_fuse_cli_params: FuseCliParams, bad_r2_inpaint_thresh: float):
    """ Test --r2-inpaint-thresh with bad value raises an error. """
    cli_str = basic_fuse_cli_params.cli_str + f' --r2-inpaint-thresh {bad_r2_inpaint_thresh}'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ('Invalid value' in result.output)


@pytest.mark.parametrize(
    'driver, dtype, nodata', [
        ('GTiff', 'float64', float('nan')),
        ('GTiff', 'uint16', 65535),
        ('PNG', 'uint8', 0),
        ('GTiff', 'uint8', None),
    ]
)  # yapf: disable
def test_out_profile(runner: CliRunner, basic_fuse_cli_params: FuseCliParams, driver: str, dtype: str, nodata: float):
    """ Test --out-* options generate a correctly configured output. """
    cli_str = basic_fuse_cli_params.cli_str + f' --driver {driver} --dtype {dtype} --nodata {nodata}'
    ext_dict = rio.drivers.raster_driver_extensions()
    ext_idx = list(ext_dict.values()).index(driver)
    ext = list(ext_dict.keys())[ext_idx]
    corr_file = basic_fuse_cli_params.corr_file.parent.joinpath(f'{basic_fuse_cli_params.corr_file.stem}.{ext}')
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (corr_file.exists())
    with rio.open(corr_file, 'r') as out_ds:
        assert (out_ds.driver == driver)
        assert (out_ds.dtypes[0] == dtype)
        assert out_ds.nodata is None if nodata is None else (utils.nan_equals(out_ds.nodata, nodata))
        assert out_ds.mask_flag_enums[0] == [MaskFlags.per_dataset] if nodata is None else [MaskFlags.nodata]


def test_out_driver_error(runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test --driver with invalid value raises an error. """
    cli_str = basic_fuse_cli_params.cli_str + f' --driver unk'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ('Invalid value' in result.output)


def test_out_dtype_error(runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test --dtype with invalid value raises an error. """
    cli_str = basic_fuse_cli_params.cli_str + f' --dtype unk'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ('Invalid value' in result.output)


def test_out_nodata_error(runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test --nodata with invalid value (cannot be cast to --dtype) raises an error. """
    cli_str = basic_fuse_cli_params.cli_str + f' --dtype uint8 --nodata nan'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code != 0)
    assert ('Invalid value' in result.output)


def test_creation_options(runner: CliRunner, basic_fuse_cli_params: FuseCliParams):
    """ Test -co creation options generate correctly configured output. """
    cli_str = basic_fuse_cli_params.cli_str + f' -co COMPRESS=LZW -co PREDICTOR=2 -co TILED=NO'
    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (basic_fuse_cli_params.corr_file.exists())
    with rio.open(basic_fuse_cli_params.corr_file, 'r') as out_ds:
        assert (out_ds.profile['compress'] == 'lzw')
        assert (not out_ds.profile['tiled'])


@pytest.mark.parametrize(
    'src_bands, ref_bands, force, exp_bands', [
        ((3, 2, 1), None, False, (3, 2, 1)),
        ((2, 1), (3, 1, 2), False, (2, 1)),
        ((2, 1), (3, 2, 1), True, (3, 2)),
    ]
)  # yapf: disable
def test_src_ref_bands(
    src_bands: Tuple[int], ref_bands: Tuple[int], force: bool, exp_bands: Tuple[int],
    default_fuse_rgb_cli_params: FuseCliParams, tmp_path: Path, runner: CliRunner,
):
    """ Test fuse with --src_band, --ref_band and --force-match parameters. """
    cli_str = default_fuse_rgb_cli_params.cli_str
    if src_bands:
        cli_str += ''.join([' -sb ' + str(bi) for bi in src_bands])
    if ref_bands:
        cli_str += ''.join([' -rb ' + str(bi) for bi in ref_bands])
    if force:
        cli_str += ' -f'

    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (default_fuse_rgb_cli_params.corr_file.exists())
    with WarpedVRT(rio.open(default_fuse_rgb_cli_params.src_file, 'r')) as src_ds:
        with rio.open(default_fuse_rgb_cli_params.corr_file, 'r') as out_ds:
            src_array = src_ds.read(indexes=exp_bands)
            src_mask = src_ds.dataset_mask().astype('bool', copy=False)
            out_array = out_ds.read()
            out_mask = out_ds.dataset_mask().astype('bool', copy=False)

            assert out_ds.count == len(exp_bands)
            assert (out_mask == src_mask).all()
            assert (out_array[:, out_mask] == pytest.approx(src_array[:, src_mask], abs=2))

@pytest.mark.parametrize(
    'src_bands, ref_bands, cmp_bands, force, cmp_ref, exp_bands', [
        (None, None, None, False, False, (1, 2, 3)),
        ((1, 2), (1, 2), (1, 2), False, False, (1, 2)),
        ((1, 2), (1, 2), None, True, False, (1, 2)),
        ((3, 1), (3, 1, 2), None, True, True, (3, 1)),
    ]
)  # yapf: disable
def test_src_ref_cmp_bands(
    src_bands: Tuple[int], ref_bands: Tuple[int], cmp_bands: Tuple[int], force: bool, cmp_ref: bool,
    exp_bands: Tuple[int], default_fuse_rgb_cli_params: FuseCliParams, tmp_path: Path, runner: CliRunner,
):
    """ Test fuse --compare with --src_band, --ref_band, --force-match and --cmp-band parameters. """
    # When bands are matched based on assumed RGB center wavelengths, the corrected file is (intentionally) not written
    # with center wavelengths.  Depending on how --src-band is spec'd, This can result in corrected files with < 3
    # bands, or corrected files with bands not in RGB order.  This in turn can lead to problems with --compare,
    # where these kinds of corrected files cannot be matched with the compare reference, or are matched incorrectly.
    # The above parameter cases avoid any of these situations.  It doesn't seem possible to work around this without
    # writing assumed RGB center wavelengths to corrected files, which seems like a bad idea.  Rather I just
    # generate a warning when RGB wavelengths are assumed.  Perhaps I should also always print out how
    # bands are matched?  Practically, I think most of the time people will correct RGB->RGB, so we wouldn't see this
    # issue often at all.

    cli_str = default_fuse_rgb_cli_params.cli_str
    if src_bands:
        cli_str += ''.join([' -sb ' + str(bi) for bi in src_bands])
    if ref_bands:
        cli_str += ''.join([' -rb ' + str(bi) for bi in ref_bands])
    if force:
        cli_str += ' -f'
    cli_str += ' -cmp' if cmp_ref else f' -cmp {str(default_fuse_rgb_cli_params.ref_file)}'
    if cmp_bands:
        cli_str += ''.join([' -cb ' + str(bi) for bi in cmp_bands])

    result = runner.invoke(cli, cli_str.split())
    assert (result.exit_code == 0)
    assert (default_fuse_rgb_cli_params.corr_file.exists())

    with WarpedVRT(rio.open(default_fuse_rgb_cli_params.src_file, 'r')) as src_ds:
        with rio.open(default_fuse_rgb_cli_params.corr_file, 'r') as out_ds:
            src_array = src_ds.read(indexes=exp_bands)
            src_mask = src_ds.dataset_mask().astype('bool', copy=False)
            out_array = out_ds.read()
            out_mask = out_ds.dataset_mask().astype('bool', copy=False)

            assert out_ds.count == len(exp_bands)
            assert (out_mask == src_mask).all()
            assert (out_array[:, out_mask] == pytest.approx(src_array[:, src_mask], abs=2))

    test_str = """                                              File    r²   RMSE   rRMSE   N
-------------------------------------------------- ----- ------ ------- ---
                                float_50cm_rgb.tif 1.000  0.000   0.000 144
float_50cm_rgb_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif 1.000  0.000   0.000 144"""
    assert test_str in result.output
