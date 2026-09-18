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

import itertools
from pathlib import Path

import pytest
import rasterio as rio
import yaml
from pytest import FixtureRequest
from rasterio.enums import MaskFlags
from rasterio.features import shapes
from rasterio.vrt import WarpedVRT

from homonim import utils
from homonim.enums import Driver, Model, ProcCrs
from homonim.errors import IoError
from homonim.fuse import RasterFuse, _default_creation_options
from homonim.kernel_model import KernelModel
from homonim.raster_array import RasterArray


def test_init(src_file_50cm_float: Path, ref_file_100cm_float: Path):
    """Test RasterFuse initialisation and context entry / exit."""
    raster_fuse = RasterFuse(src_file_50cm_float, ref_file_100cm_float)
    with raster_fuse:
        assert raster_fuse.proc_crs is ProcCrs.ref
        assert not raster_fuse.closed
    assert raster_fuse.closed


@pytest.mark.parametrize('overwrite', [False, True])
def test_overwrite(
    tmp_path: Path, src_file_50cm_float, ref_file_100cm_float, overwrite: bool
):
    """Test the overwrite parameter."""
    corr_file = tmp_path.joinpath('corrected.tif')
    param_file = tmp_path.joinpath('parameter.tif')
    params = dict(
        corr_filename=corr_file, param_filename=param_file, overwrite=overwrite
    )
    raster_fuse = RasterFuse(
        src_filename=src_file_50cm_float, ref_filename=ref_file_100cm_float
    )

    # test overwriting the corrected image
    corr_file.touch()
    with raster_fuse:
        if not overwrite:
            with pytest.raises(FileExistsError):
                raster_fuse.process(**params)
        else:
            raster_fuse.process(**params)

    # test overwriting the parameter image
    corr_file.unlink()
    param_file.touch()
    with raster_fuse:
        if not overwrite:
            with pytest.raises(FileExistsError):
                raster_fuse.process(**params)
        else:
            raster_fuse.process(**params)


# ruff: ignore[E501]
@pytest.mark.parametrize(
    'src_file, ref_file, model, kernel_shape, max_block_mem',
    [
        ('src_file_45cm_float', 'ref_file_100cm_float', Model.gain, (1, 1), 2.e-4),
        ('src_file_45cm_float', 'ref_file_100cm_float', Model.gain_blk_offset, (1, 1), 1.e-3),
        ('src_file_45cm_float', 'ref_file_100cm_float', Model.gain_offset, (5, 5), 1.e-3),
        ('src_file_100cm_float', 'ref_file_45cm_float', Model.gain, (1, 1), 2.e-4),
        ('src_file_100cm_float', 'ref_file_45cm_float', Model.gain_blk_offset, (1, 1), 1.e-3),
        ('src_file_100cm_float', 'ref_file_45cm_float', Model.gain_offset, (5, 5), 1.e-3),
        ('src_file_45cm_float', 'ref_file_wgs84_sup_float', Model.gain_blk_offset, (1, 1), 1.e-3),
        ('src_file_wgs84_sup_100cm_float', 'ref_file_45cm_float', Model.gain_blk_offset, (1, 1), 1.e-3),
    ]
)  # fmt: skip
def test_corr_content(
    src_file: str,
    ref_file: str,
    model: Model,
    kernel_shape: tuple[int, int],
    max_block_mem: float,
    tmp_path: Path,
    request: FixtureRequest,
):
    """Test the corrected image content with different src / ref images, and model
    etc combinations.
    """
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    corr_file = tmp_path.joinpath('corrected.tif')
    raster_fuse = RasterFuse(src_file, ref_file)
    with raster_fuse:
        raster_fuse.process(corr_file, model, kernel_shape, max_block_mem=max_block_mem)
    assert corr_file.exists()

    # open src_file in a WarpedVRT to reproject it North-up (if necessary)
    with (
        WarpedVRT(rio.open(src_file, 'r')) as src_ds,
        rio.open(corr_file, 'r') as corr_ds,
    ):
        src_array = src_ds.read(indexes=1)
        src_mask = src_ds.dataset_mask().astype('bool', copy=False)
        out_array = corr_ds.read(indexes=1)
        out_mask = corr_ds.dataset_mask().astype('bool', copy=False)
        assert (out_mask == src_mask).all()
        assert out_array[out_mask] == pytest.approx(src_array[src_mask], abs=2)


@pytest.mark.parametrize('driver', Driver)
def test_corr_profile_defaults(
    tmp_path: Path, src_file_100cm_float: Path, driver: Driver
):
    """Test the corrected image format with default parameter values for different
    drivers.
    """
    raster_fuse = RasterFuse(src_file_100cm_float, src_file_100cm_float)
    corr_file = tmp_path.joinpath('corrected.tif')
    with raster_fuse:
        raster_fuse.process(corr_file, driver=driver)
    assert corr_file.exists()

    src_keys = ['crs', 'transform', 'width', 'height', 'count']
    creation_option_keys = [
        'tiled',
        'blockxsize',
        'blockysize',
        'compress',
        'interleave',
    ]
    with (
        rio.open(src_file_100cm_float, 'r') as src_ds,
        rio.open(corr_file, 'r') as corr_ds,
    ):
        assert corr_ds.driver.lower() == 'gtiff'
        if driver is Driver.cog:
            # TODO: GDAL 3.13.3 sets LAYOUT=COG for any GeoTIFF with 1 tile so this
            #  this test will pass for driver=gtiff too
            im_struct = corr_ds.tags(ns='IMAGE_STRUCTURE')
            assert im_struct['LAYOUT'].lower() == 'cog'

        assert corr_ds.dtypes[0] == RasterArray.default_dtype
        assert utils.nan_equals(corr_ds.nodata, RasterArray.default_nodata)

        for k in src_keys:
            assert corr_ds.profile[k] == src_ds.profile[k]

        # test cog creation options against the gtiff defaults (cogs are read as
        # gtiffs and default gtiff and cog creation options amount to the same thing)
        for k in creation_option_keys:
            assert corr_ds.profile[k] == _default_creation_options['gtiff'][k]


@pytest.mark.parametrize(
    'driver, dtype, nodata',
    [
        (Driver.gtiff, 'float64', float('inf')),
        (Driver.gtiff, 'uint16', 65535),
        (Driver.cog, 'uint8', None),
        (Driver.gtiff, 'uint8', None),
    ],
)
def test_corr_profile(
    tmp_path: Path,
    src_file_100cm_float: Path,
    driver: Driver,
    dtype: str,
    nodata: float | None,
):
    """Test the corrected image format with different driver, dtype and nodata
    parameters.
    """
    raster_fuse = RasterFuse(src_file_100cm_float, src_file_100cm_float)
    corr_file = tmp_path.joinpath('corrected.tif')
    with raster_fuse:
        raster_fuse.process(corr_file, driver=driver, dtype=dtype, nodata=nodata)
    assert corr_file.exists()

    with rio.open(corr_file, 'r') as corr_ds:
        assert corr_ds.driver.lower() == 'gtiff'
        if driver is Driver.cog:
            im_struct = corr_ds.tags(ns='IMAGE_STRUCTURE')
            assert im_struct['LAYOUT'].lower() == 'cog'
        assert corr_ds.dtypes[0] == dtype
        assert (
            corr_ds.nodata is None
            if nodata is None
            else (utils.nan_equals(corr_ds.nodata, nodata))
        )
        assert (
            corr_ds.mask_flag_enums[0] == [MaskFlags.per_dataset]
            if nodata is None
            else [MaskFlags.nodata]
        )


def test_corr_creation_options(tmp_path: Path, file_rgb_100cm_float: Path):
    """Test the corrected image is formatted according to the creation_options
    parameter.
    """
    raster_fuse = RasterFuse(file_rgb_100cm_float, file_rgb_100cm_float)
    corr_file = tmp_path.joinpath('corrected.tif')
    creation_options = dict(
        tiled=True,
        blockxsize=64,
        blockysize=64,
        compress='jpeg',
        interleave='pixel',
        photometric='ycbcr',
    )
    with raster_fuse:
        raster_fuse.process(corr_file, creation_options=creation_options)
    assert corr_file.exists()

    with rio.open(corr_file, 'r') as corr_ds:
        for k, v in creation_options.items():
            assert corr_ds.profile[k] == v


def test_param_profile(tmp_path: Path, src_file_100cm_float: Path):
    """Test the parameter image format."""
    corr_file = tmp_path.joinpath('corrected.tif')
    param_file = tmp_path.joinpath('parameter.tif')
    raster_fuse = RasterFuse(src_file_100cm_float, src_file_100cm_float)
    with raster_fuse:
        raster_fuse.process(corr_file, param_filename=param_file)
    assert param_file.exists()

    src_keys = ['crs', 'transform', 'width', 'height']
    creation_option_keys = [
        'tiled',
        'blockxsize',
        'blockysize',
        'compress',
        'interleave',
    ]
    with (
        rio.open(src_file_100cm_float, 'r') as src_ds,
        rio.open(param_file, 'r') as param_ds,
    ):
        assert param_ds.driver.lower() == 'gtiff'
        assert param_ds.dtypes[0] == RasterArray.default_dtype
        assert utils.nan_equals(param_ds.nodata, RasterArray.default_nodata)

        for k in src_keys:
            assert param_ds.profile[k] == src_ds.profile[k]
        assert param_ds.count == src_ds.count * 3

        for k in creation_option_keys:
            assert param_ds.profile[k] == _default_creation_options['gtiff'][k]


@pytest.mark.parametrize(
    'model, proc_crs', itertools.product(Model, [ProcCrs.ref, ProcCrs.src])
)
def test_param_content(
    tmp_path: Path,
    file_rgb_50cm_float: Path,
    file_rgb_100cm_float: Path,
    model: Model,
    proc_crs: ProcCrs,
):
    """Test the parameter image content for different model and proc_crs
    combinations.
    """
    corr_file = tmp_path.joinpath('corrected.tif')
    param_file = tmp_path.joinpath('parameter.tif')
    src_file, ref_file = (
        (file_rgb_50cm_float, file_rgb_100cm_float)
        if proc_crs is ProcCrs.ref
        else (file_rgb_100cm_float, file_rgb_50cm_float)
    )
    raster_fuse = RasterFuse(src_file, ref_file, proc_crs=proc_crs)
    with raster_fuse:
        raster_fuse.process(corr_file, model, (5, 5), param_filename=param_file)
    assert param_file.exists()

    proc_file = ref_file if proc_crs is ProcCrs.ref else src_file
    with (
        rio.open(proc_file, 'r') as proc_ds,
        rio.open(param_file, 'r') as param_ds,
    ):
        param_mask = param_ds.dataset_mask().astype('bool', copy=False)
        proc_mask = proc_ds.dataset_mask().astype('bool', copy=False)
        assert (param_mask == proc_mask).all()

        param_array = param_ds.read()
        assert param_array[0:3, param_mask] == pytest.approx(1, abs=0.1)
        assert param_array[3:6, param_mask] == pytest.approx(0, abs=0.1)
        assert param_array[6:9, param_mask] == pytest.approx(1, abs=0.1)


@pytest.mark.parametrize(
    'src_file, ref_file, kernel_shape, mask_partial',
    [
        ('src_file_45cm_float', 'ref_file_100cm_float', (1, 1), False),
        ('src_file_45cm_float', 'ref_file_100cm_float', (1, 1), True),
        ('src_file_45cm_float', 'ref_file_100cm_float', (3, 3), True),
        ('src_file_100cm_float', 'ref_file_45cm_float', (1, 1), False),
        ('src_file_100cm_float', 'ref_file_45cm_float', (1, 1), True),
        ('src_file_100cm_float', 'ref_file_45cm_float', (3, 3), True),
    ],
)
def test_mask_partial(
    tmp_path: Path,
    request: FixtureRequest,
    src_file: str,
    ref_file: str,
    kernel_shape: tuple[int, int],
    mask_partial: bool,
):
    """Test partial masking with multiple image blocks."""
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    corr_file = tmp_path.joinpath('corrected.tif')

    raster_fuse = RasterFuse(src_file, ref_file)
    with raster_fuse:
        raster_fuse.process(
            corr_file,
            Model.gain_blk_offset,
            kernel_shape,
            mask_partial=mask_partial,
            max_block_mem=0.1,
        )
    assert corr_file.exists()

    with rio.open(src_file, 'r') as src_ds, rio.open(corr_file, 'r') as corr_ds:
        src_mask = src_ds.dataset_mask().astype('bool', copy=False)
        corr_mask = corr_ds.dataset_mask().astype('bool', copy=False)
        if not mask_partial:
            assert (corr_mask == src_mask).all()
        else:
            assert corr_mask.sum() < src_mask.sum()
            assert corr_mask.sum() > 0
            assert src_mask[corr_mask].all()
            # check that the output mask consists of 1 blob
            out_mask_shapes = [*shapes(corr_mask.view('uint8'), mask=corr_mask)]
            assert len(out_mask_shapes) == 1


def test_build_overviews(
    tmp_path: Path, ref_file_100cm_float, monkeypatch: pytest.MonkeyPatch
):
    """Test that overviews are built for corrected and parameter files."""
    corr_file = tmp_path.joinpath('corrected.tif')
    param_file = tmp_path.joinpath('parameter.tif')
    raster_fuse = RasterFuse(ref_file_100cm_float, ref_file_100cm_float)

    # patch RasterFuse._build_overviews() to force min_level_pixels==1, otherwise
    # overviews won't be built for the small test raster
    def build_overviews(im):
        _build_overviews(im, min_level_pixels=1)

    _build_overviews = RasterFuse._build_overviews
    monkeypatch.setattr(RasterFuse, '_build_overviews', staticmethod(build_overviews))

    with raster_fuse:
        raster_fuse.process(
            corr_file,
            Model.gain_blk_offset,
            (3, 3),
            param_filename=param_file,
            build_ovw=True,
        )
    assert corr_file.exists()
    assert param_file.exists()

    with rio.open(corr_file, 'r') as fuse_ds:
        assert len(fuse_ds.overviews(1)) > 0
    with rio.open(param_file, 'r') as param_ds:
        for band_i in param_ds.indexes:
            assert len(param_ds.overviews(band_i)) > 0


def test_io_error(tmp_path: Path, ref_file_50cm_float):
    """Test we get an IoError if processing without opening/entering the context."""
    raster_fuse = RasterFuse(ref_file_50cm_float, ref_file_50cm_float)
    with pytest.raises(IoError):
        raster_fuse.process(tmp_path, Model.gain_blk_offset, (3, 3))


@pytest.mark.parametrize(
    'src_file, ref_file, proc_crs, exp_proc_crs',
    [
        ('src_file_50cm_float', 'ref_file_100cm_float', ProcCrs.auto, ProcCrs.ref),
        ('src_file_50cm_float', 'ref_file_100cm_float', ProcCrs.src, ProcCrs.src),
        ('src_file_100cm_float', 'ref_file_50cm_float', ProcCrs.auto, ProcCrs.src),
        ('src_file_100cm_float', 'ref_file_50cm_float', ProcCrs.ref, ProcCrs.ref),
    ],
)
def test_proc_crs(
    tmp_path: Path,
    src_file: str,
    ref_file: str,
    proc_crs: ProcCrs,
    exp_proc_crs: ProcCrs,
    request: FixtureRequest,
):
    """Test corrected image creation for forced and auto proc_crs with different
    src / ref combinations.
    """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    corr_filename = tmp_path.joinpath('corrected.tif')
    raster_fuse = RasterFuse(src_file, ref_file, proc_crs=proc_crs)
    assert raster_fuse.proc_crs == exp_proc_crs
    with raster_fuse:
        raster_fuse.process(corr_filename, Model.gain_blk_offset, (5, 5))
    assert corr_filename.exists()


def test_corr_tags(tmp_path: Path, ref_file_100cm_float):
    """Test the corrected image tags."""
    model = Model.gain_blk_offset
    kernel_shape = (3, 3)
    proc_crs = ProcCrs.ref
    raster_fuse = RasterFuse(
        ref_file_100cm_float, ref_file_100cm_float, proc_crs=proc_crs
    )
    corr_file = tmp_path.joinpath('corrected.tif')
    param_file = tmp_path.joinpath('parameter.tif')

    with raster_fuse:
        raster_fuse.process(corr_file, model, kernel_shape, param_filename=param_file)

    assert corr_file.exists()
    assert param_file.exists()
    utils.validate_param_image(param_file)

    with rio.open(corr_file, 'r') as out_ds:
        tags = out_ds.tags()
        assert {
            'FUSE_SRC_FILE',
            'FUSE_REF_FILE',
            'FUSE_MODEL',
            'FUSE_KERNEL_SHAPE',
            'FUSE_PROC_CRS',
            'FUSE_MAX_BLOCK_MEM',
            *{f'FUSE_{k.upper()}' for k in KernelModel._default_config.keys()},
        } <= set(tags)
        assert tags['FUSE_SRC_FILE'] == ref_file_100cm_float.name
        assert tags['FUSE_REF_FILE'] == ref_file_100cm_float.name
        assert tags['FUSE_MODEL'] == model
        assert tags['FUSE_PROC_CRS'] == proc_crs
        assert tags['FUSE_KERNEL_SHAPE'] == str(kernel_shape)

        for key, val in KernelModel._default_config.items():
            assert (
                tags[f'FUSE_{key.upper()}'] == val.name
                if hasattr(val, 'name')
                else str(val)
            )
        assert (
            yaml.safe_load(tags['FUSE_MAX_BLOCK_MEM'])
            == RasterFuse._default_config['max_block_mem']
        )


# ruff: ignore[E501]
@pytest.mark.parametrize(
    'src_file, ref_file, src_bands, ref_bands, force, exp_bands',
    [
        ('file_rgb_50cm_float', 'file_rgb_100cm_float', None, None, False, (1, 2, 3)),
        ('file_rgb_50cm_float', 'file_rgb_100cm_float', (3, 2, 1), None, False, (3, 2, 1)),
        ('file_rgb_50cm_float', 'file_rgb_100cm_float', None, (3, 2, 1), False, (1, 2, 3)),
        ('file_rgb_50cm_float', 'file_rgb_100cm_float', (2, 1), (3, 1, 2), False, (2, 1)),
        ('file_rgb_50cm_float', 'file_rgb_100cm_float', (2, 1), (3, 2, 1), True, (3, 2))
    ]
)  # fmt: skip
def test_src_ref_bands(
    src_file: str,
    ref_file: str,
    src_bands: tuple[int],
    ref_bands: tuple[int],
    force: bool,
    exp_bands: tuple[int],
    tmp_path: Path,
    request: FixtureRequest,
):
    """Test the corrected image content is as expected with the src_bands and
    ref_bands parameters.
    """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    corr_file = tmp_path.joinpath('corrected.tif')

    with RasterFuse(
        src_file, ref_file, src_bands=src_bands, ref_bands=ref_bands, force=force
    ) as raster_fuse:
        raster_fuse.process(corr_file, model=Model.gain_blk_offset, kernel_shape=(3, 3))
    assert corr_file.exists()

    with (
        rio.open(src_file, 'r') as src_ds,
        rio.open(corr_file, 'r') as corr_ds,
    ):
        src_array = src_ds.read(indexes=exp_bands)
        src_mask = src_ds.dataset_mask().astype('bool', copy=False)
        out_array = corr_ds.read()
        out_mask = corr_ds.dataset_mask().astype('bool', copy=False)

        assert corr_ds.count == len(exp_bands)
        assert (out_mask == src_mask).all()
        assert out_array[:, out_mask] == pytest.approx(src_array[:, src_mask], abs=2)
