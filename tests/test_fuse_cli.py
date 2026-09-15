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

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pytest
import rasterio as rio
import yaml
from click.testing import CliRunner
from rasterio.enums import MaskFlags
from rasterio.warp import Resampling

from homonim import utils
from homonim.cli import cli
from homonim.enums import Driver, Model, ProcCrs
from homonim.kernel_model import KernelModel
from tests.conftest import (
    create_corr_filename,
    create_param_filename,
    str_contain_no_space,
)


@dataclass
class FuseDefaults:
    """Class to provide CLI string and output files for default option values."""

    src_file: Path
    ref_file: Path
    out_dir: Path
    # the resolved proc_crs, not ProcCrs.auto
    proc_crs: ProcCrs

    @cached_property
    def cli_str(self) -> str:
        """CLI string."""
        return f'fuse -od {self.out_dir} {self.src_file} {self.ref_file}'

    @cached_property
    def corr_file(self) -> Path:
        """Path of the corrected image."""
        model = KernelModel._default_config['model']
        kernel_shape = KernelModel._default_config['kernel_shape']
        corr_file = create_corr_filename(
            self.src_file, self.proc_crs, model, kernel_shape
        )
        return self.out_dir.joinpath(corr_file)

    @cached_property
    def param_file(self) -> Path:
        """Path of the parameter image."""
        param_file = create_param_filename(self.corr_file)
        return self.out_dir.joinpath(param_file)


@pytest.fixture
def fuse_defaults(
    tmp_path: Path, src_file_100cm_float: Path, ref_file_100cm_float: Path
) -> FuseDefaults:
    """FuseDefaults using single band source and reference files."""
    return FuseDefaults(
        src_file_100cm_float, ref_file_100cm_float, tmp_path, ProcCrs.ref
    )


@pytest.fixture
def fuse_rgb_defaults(tmp_path: Path, file_rgb_100cm_float: Path) -> FuseDefaults:
    """FuseDefaults using RGB source and reference files."""
    return FuseDefaults(
        file_rgb_100cm_float, file_rgb_100cm_float, tmp_path, ProcCrs.ref
    )


@pytest.mark.parametrize(
    'model, kernel_shape',
    [
        (Model.gain, (1, 1)),
        (Model.gain_blk_offset, (1, 1)),
        (Model.gain_offset, (5, 5)),
    ],
)
def test_fuse(
    tmp_path: Path,
    runner: CliRunner,
    file_rgb_100cm_float,
    model: Model,
    kernel_shape: tuple[int, int],
):
    """Test fuse CLI output with different models and kernel shapes."""
    ref_file = file_rgb_100cm_float
    src_file = file_rgb_100cm_float
    corr_file = create_corr_filename(src_file, ProcCrs.ref, model, kernel_shape)
    corr_file = tmp_path.joinpath(corr_file)
    cli_str = (
        f'fuse -m {model.value} -k {kernel_shape[0]} {kernel_shape[1]} -od {tmp_path} '
        f'{src_file} {ref_file}'
    )
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert corr_file.exists()

    with rio.open(src_file, 'r') as src_ds, rio.open(corr_file, 'r') as out_ds:
        assert out_ds.tags()['FUSE_MODEL'] == model.name
        assert out_ds.tags()['FUSE_KERNEL_SHAPE'] == str(kernel_shape)

        src_array = src_ds.read(indexes=src_ds.indexes)
        src_mask = src_ds.dataset_mask().astype('bool', copy=False)
        out_array = out_ds.read(indexes=out_ds.indexes)
        out_mask = out_ds.dataset_mask().astype('bool', copy=False)
        assert (out_mask == src_mask).all()
        assert out_array[:, out_mask] == pytest.approx(src_array[:, src_mask], abs=0.1)


def test_fuse_defaults(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test fuse cli works without model or kernel shape arguments."""
    result = runner.invoke(cli, fuse_defaults.cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()


def test_kernel_shape_error(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test bad kernel shape generates an error."""
    cli_str = fuse_defaults.cli_str + ' -k 2 3'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0
    assert 'kernel_shape' in result.output


def test_file_exists_error(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test that attempting to overwrite an existing output file generates an error."""
    fuse_defaults.corr_file.touch()
    cli_str = fuse_defaults.cli_str
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0
    assert 'exists' in result.output

    fuse_defaults.corr_file.unlink()
    fuse_defaults.param_file.touch()
    cli_str = fuse_defaults.cli_str + ' --param-image'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0
    assert 'exists' in result.output


def test_overwrite(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test overwriting existing output file(s) with -o."""
    fuse_defaults.corr_file.touch()
    fuse_defaults.param_file.touch()
    cli_str = fuse_defaults.cli_str + ' --param-image -o'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()
    assert fuse_defaults.param_file.exists()


def test_compare(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test --compare against expected output."""
    cli_strs = [
        fuse_defaults.cli_str + ' --compare ref',
        fuse_defaults.cli_str + f' -o --compare {fuse_defaults.ref_file}',
    ]
    for cli_str in cli_strs:
        result = runner.invoke(cli, cli_str.split())
        assert result.exit_code == 0
        src_cmp_str = """float_100cm_src.tif:
           Band    r²   RMSE   rRMSE   N
    ----------- ----- ------ ------- ---
    Ref. band 1 1.000  0.000   0.000 144
           Mean 1.000  0.000   0.000 144"""
        assert str_contain_no_space(src_cmp_str, result.output)

        corr_cmp_str = """float_100cm_src_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif:
           Band      r²   RMSE   rRMSE   N
    ----------- ----- ------ ------- ---
    Ref. band 1 1.000  0.000   0.000 144
           Mean 1.000  0.000   0.000 144"""
        assert str_contain_no_space(corr_cmp_str, result.output)

        sum_cmp_str = """File    r²   RMSE   rRMSE   N
    --------------------------------------------------- ----- ------ ------- ---
                                    float_100cm_src.tif 1.000  0.000   0.000 144
    float_100cm_src_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif 1.000  0.000   0.000 144"""
        assert str_contain_no_space(sum_cmp_str, result.output)


def test_compare_file_exists_error(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test --compare raises an exception when the specified file does not exist."""
    cli_str = fuse_defaults.cli_str + ' --compare unknown.tif'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0
    assert 'No such file or directory' in result.output


@pytest.mark.parametrize('proc_crs', ProcCrs)
def test_proc_crs(
    runner: CliRunner,
    tmp_path: Path,
    src_file_100cm_float: Path,
    ref_file_100cm_float: Path,
    proc_crs: ProcCrs,
):
    """Test --proc-crs generates a corrected image with the correct metadata."""
    res_proc_crs = ProcCrs.ref if proc_crs is ProcCrs.auto else proc_crs
    fuse_defaults = FuseDefaults(
        src_file_100cm_float, ref_file_100cm_float, tmp_path, res_proc_crs
    )
    cli_str = fuse_defaults.cli_str + f' -pc {proc_crs}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()

    with rio.open(fuse_defaults.corr_file, 'r') as out_ds:
        assert out_ds.tags()['FUSE_PROC_CRS'] == res_proc_crs


def test_conf_file(tmp_path: Path, runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test passing a configuration file results in a correctly configured output."""
    # create test configuration file
    conf_dict = dict(
        mask_partial=True,
        param_image=True,
        dtype='uint8',
        nodata=0,
        creation_options=dict(compress='lzw'),
    )
    conf_file = tmp_path.joinpath('conf.yaml')
    with open(conf_file, 'w') as f:
        yaml.dump(conf_dict, f)

    cli_str = fuse_defaults.cli_str + f' -c {conf_file}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()
    # test param_image==True
    assert fuse_defaults.param_file.exists()

    with rio.open(fuse_defaults.src_file, 'r') as src_ds:
        with rio.open(fuse_defaults.corr_file, 'r') as out_ds:
            # test nodata, dtype and creation_options
            assert out_ds.nodata == conf_dict['nodata']
            assert out_ds.dtypes[0] == conf_dict['dtype']
            assert (
                out_ds.profile['compress'] == conf_dict['creation_options']['compress']
            )
            # test mask_partial==True
            src_mask = src_ds.dataset_mask().astype('bool', copy=False)
            out_mask = out_ds.dataset_mask().astype('bool', copy=False)
            assert src_mask[out_mask].all()
            assert src_mask.sum() > out_mask.sum()
            # test proc_crs
            assert out_ds.tags()['FUSE_PROC_CRS'] == fuse_defaults.proc_crs


def test_param_image(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test --param-image."""
    # test that cli without --param-image generates no parameter image
    cli_str = fuse_defaults.cli_str
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()
    assert not fuse_defaults.param_file.exists()

    # test --param-image generates a valid parameter image
    cli_str = fuse_defaults.cli_str + ' --param-image -o'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()
    assert fuse_defaults.param_file.exists()
    utils.validate_param_image(fuse_defaults.param_file)


def test_mask_partial(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test --mask-partial."""
    cli_str = fuse_defaults.cli_str + ' --mask-partial'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()

    with rio.open(fuse_defaults.src_file, 'r') as src_ds:
        with rio.open(fuse_defaults.corr_file, 'r') as out_ds:
            # test that the output mask is contained by and smaller than the src mask
            src_mask = src_ds.dataset_mask().astype('bool', copy=False)
            out_mask = out_ds.dataset_mask().astype('bool', copy=False)
            assert src_mask[out_mask].all()
            assert src_mask.sum() > out_mask.sum()


def test_threads(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test --threads."""
    cli_str = fuse_defaults.cli_str + ' --threads 1'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()

    # test that threads > os.cpu_count() raises an error
    cli_str = fuse_defaults.cli_str + f' -o -threads {os.cpu_count() + 1}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0
    assert 'threads' in result.output


def test_max_block_mem(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test --max-block-mem."""
    max_block_mem = 123
    cli_str = fuse_defaults.cli_str + f' -mbm {max_block_mem}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()

    with rio.open(fuse_defaults.corr_file, 'r') as out_ds:
        assert yaml.safe_load(out_ds.tags()['FUSE_MAX_BLOCK_MEM']) == max_block_mem

    # test that max_block_mem too small raises an error
    cli_str = fuse_defaults.cli_str + ' -o -mbm 1e-6'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0
    assert 'max_block_mem' in result.output


@pytest.mark.parametrize('upsampling', ['cubic', 'bilinear'])
def test_upsampling(
    runner: CliRunner, fuse_defaults: FuseDefaults, upsampling: Resampling
):
    """Test --upsampling with valid values generates output with correct metadata."""
    cli_str = fuse_defaults.cli_str + f' --upsampling {upsampling}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()
    with rio.open(fuse_defaults.corr_file, 'r') as out_ds:
        tags_dict = out_ds.tags()
        assert 'FUSE_UPSAMPLING' in tags_dict
        assert tags_dict['FUSE_UPSAMPLING'] == upsampling


@pytest.mark.parametrize('downsampling', ['bilinear', 'nearest'])
def test_downsampling(
    runner: CliRunner, fuse_defaults: FuseDefaults, downsampling: Resampling
):
    """Test --downsampling with valid values generates output with correct metadata."""
    cli_str = fuse_defaults.cli_str + f' --downsampling {downsampling}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()
    with rio.open(fuse_defaults.corr_file, 'r') as out_ds:
        tags_dict = out_ds.tags()
        assert 'FUSE_DOWNSAMPLING' in tags_dict
        assert tags_dict['FUSE_DOWNSAMPLING'] == downsampling


def test_upsampling_error(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test --upsampling with bad value raises an error."""
    cli_str = fuse_defaults.cli_str + ' --upsampling unknown'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0
    assert "Invalid value for '-us' / '--upsampling'" in result.output


def test_downsampling_error(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test --downsampling with bad value raises an error."""
    cli_str = fuse_defaults.cli_str + ' --downsampling unknown'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0
    assert "Invalid value for '-ds' / '--downsampling'" in result.output


@pytest.mark.parametrize('r2_inpaint_thresh', [0, 0.5])
def test_r2_inpaint_thresh(
    runner: CliRunner, fuse_defaults: FuseDefaults, r2_inpaint_thresh: float
):
    """Test --r2-inpaint-thresh generates an output with correct metadata."""
    cli_str = fuse_defaults.cli_str + f' --r2-inpaint-thresh {r2_inpaint_thresh}'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()
    with rio.open(fuse_defaults.corr_file, 'r') as out_ds:
        tags_dict = out_ds.tags()
        assert 'FUSE_R2_INPAINT_THRESH' in tags_dict
        r2_inpaint_thresh_tag = yaml.safe_load(tags_dict['FUSE_R2_INPAINT_THRESH'])
        assert r2_inpaint_thresh_tag == (
            'None' if r2_inpaint_thresh == 0 else r2_inpaint_thresh
        )


@pytest.mark.parametrize(
    'driver, dtype, nodata',
    [
        (Driver.gtiff, 'float64', float('nan')),
        (Driver.gtiff, 'uint16', 65535),
        (Driver.cog, 'uint8', 0),
        (Driver.gtiff, 'uint8', None),
    ],
)
def test_corr_profile(
    runner: CliRunner,
    fuse_defaults: FuseDefaults,
    driver: Driver,
    dtype: str,
    nodata: float,
):
    """Test the --driver, --dtype and --nodata options generate a correctly configured
    image.
    """
    cli_str = (
        fuse_defaults.cli_str + f' --driver {driver} --dtype {dtype} --nodata {nodata}'
    )

    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()
    with rio.open(fuse_defaults.corr_file, 'r') as out_ds:
        assert out_ds.driver.lower() == 'gtiff'
        if driver is Driver.cog:
            # TODO: GDAL 3.13.3 sets LAYOUT=COG for any GeoTIFF with 1 tile so this
            #  this test will pass for --driver gtiff too
            im_struct = out_ds.tags(ns='IMAGE_STRUCTURE')
            assert im_struct['LAYOUT'].lower() == 'cog'

        assert out_ds.dtypes[0] == dtype
        assert (
            out_ds.nodata is None
            if nodata is None
            else (utils.nan_equals(out_ds.nodata, nodata))
        )
        assert (
            out_ds.mask_flag_enums[0] == [MaskFlags.per_dataset]
            if nodata is None
            else [MaskFlags.nodata]
        )


def test_dtype_error(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test --dtype with invalid value raises an error."""
    cli_str = fuse_defaults.cli_str + ' --dtype unk'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0
    assert 'Invalid value' in result.output


def test_nodata_error(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test --nodata with a value that cannot be cast to --dtype raises an error."""
    cli_str = fuse_defaults.cli_str + ' --dtype uint8 --nodata nan'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code != 0
    assert 'nodata' in result.output and 'dtype' in result.output


def test_creation_options(runner: CliRunner, fuse_defaults: FuseDefaults):
    """Test -co creation options generate correctly configured output."""
    cli_str = fuse_defaults.cli_str + ' -co COMPRESS=LZW -co PREDICTOR=2 -co TILED=NO'
    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_defaults.corr_file.exists()
    with rio.open(fuse_defaults.corr_file, 'r') as out_ds:
        assert out_ds.profile['compress'] == 'lzw'
        assert not out_ds.profile['tiled']


@pytest.mark.parametrize(
    'src_bands, ref_bands, force, exp_bands',
    [
        ((3, 2, 1), None, False, (3, 2, 1)),
        ((2, 1), (3, 1, 2), False, (2, 1)),
        ((2, 1), (3, 2, 1), True, (3, 2)),
    ],
)
def test_src_ref_bands(
    runner: CliRunner,
    tmp_path: Path,
    src_bands: tuple[int],
    ref_bands: tuple[int],
    force: bool,
    exp_bands: tuple[int],
    fuse_rgb_defaults: FuseDefaults,
):
    """Test fuse with --src_band, --ref_band and --force-match parameters."""
    cli_str = fuse_rgb_defaults.cli_str
    if src_bands:
        cli_str += ''.join([' -sb ' + str(bi) for bi in src_bands])
    if ref_bands:
        cli_str += ''.join([' -rb ' + str(bi) for bi in ref_bands])
    if force:
        cli_str += ' -f'

    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_rgb_defaults.corr_file.exists()
    with rio.open(fuse_rgb_defaults.src_file, 'r') as src_ds:
        with rio.open(fuse_rgb_defaults.corr_file, 'r') as out_ds:
            src_array = src_ds.read(indexes=exp_bands)
            src_mask = src_ds.dataset_mask().astype('bool', copy=False)
            out_array = out_ds.read()
            out_mask = out_ds.dataset_mask().astype('bool', copy=False)

            assert out_ds.count == len(exp_bands)
            assert (out_mask == src_mask).all()
            assert out_array[:, out_mask] == pytest.approx(
                src_array[:, src_mask], abs=2
            )


# TODO: test --src-band / --ref-band errors
@pytest.mark.parametrize(
    'src_bands, ref_bands, cmp_bands, force, cmp_ref, exp_bands',
    [
        (None, None, None, False, False, (1, 2, 3)),
        ((1, 2), (1, 2), (1, 2), False, False, (1, 2)),
        ((1, 2), (1, 2), None, True, False, (1, 2)),
        ((3, 1), (3, 1, 2), None, True, True, (3, 1)),
    ],
)
def test_src_ref_cmp_bands(
    runner: CliRunner,
    tmp_path: Path,
    src_bands: tuple[int],
    ref_bands: tuple[int],
    cmp_bands: tuple[int],
    force: bool,
    cmp_ref: bool,
    exp_bands: tuple[int],
    fuse_rgb_defaults: FuseDefaults,
):
    """Test fuse --compare with --src_band, --ref_band, --force-match and --cmp-band
    parameters.
    """
    # When bands are matched based on assumed RGB center wavelengths, the corrected
    # file is (intentionally) not written with center wavelengths.  Depending on how
    # --src-band is spec'd, This can result in corrected files with < 3 bands,
    # or corrected files with bands not in RGB order.  This in turn can lead to
    # problems with --compare, where these kinds of corrected files cannot be matched
    # with the compare reference, or are matched incorrectly. The above parameter
    # cases avoid any of these situations.

    # TODO: It doesn't seem possible to work around this without writing assumed RGB
    #  center wavelengths to corrected files, which seems like a bad idea.  Rather I
    #  just generate a warning when RGB wavelengths are assumed.  Perhaps I should
    #  also always print out how bands are matched?  Practically, I think most of the
    #  time people will correct RGB->RGB, so we wouldn't see this issue often at all.

    cli_str = fuse_rgb_defaults.cli_str
    if src_bands:
        cli_str += ''.join([' -sb ' + str(bi) for bi in src_bands])
    if ref_bands:
        cli_str += ''.join([' -rb ' + str(bi) for bi in ref_bands])
    if force:
        cli_str += ' -f'
    cli_str += ' -cmp ref' if cmp_ref else f' -cmp {fuse_rgb_defaults.ref_file!s}'
    if cmp_bands:
        cli_str += ''.join([' -cb ' + str(bi) for bi in cmp_bands])

    result = runner.invoke(cli, cli_str.split())
    assert result.exit_code == 0
    assert fuse_rgb_defaults.corr_file.exists()

    with rio.open(fuse_rgb_defaults.src_file, 'r') as src_ds:
        with rio.open(fuse_rgb_defaults.corr_file, 'r') as out_ds:
            src_array = src_ds.read(indexes=exp_bands)
            src_mask = src_ds.dataset_mask().astype('bool', copy=False)
            out_array = out_ds.read()
            out_mask = out_ds.dataset_mask().astype('bool', copy=False)

            assert out_ds.count == len(exp_bands)
            assert (out_mask == src_mask).all()
            assert out_array[:, out_mask] == pytest.approx(
                src_array[:, src_mask], abs=2
            )

    test_str = """File    r²   RMSE   rRMSE   N
--------------------------------------------------- ----- ------ ------- ---
                                float_100cm_rgb.tif 1.000  0.000   0.000 144
float_100cm_rgb_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif 1.000  0.000   0.000 144"""
    assert str_contain_no_space(test_str, result.output)
