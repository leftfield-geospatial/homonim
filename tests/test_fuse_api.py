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
import os

import pytest
import rasterio as rio
import yaml
from homonim import utils
from homonim.enums import ProcCrs, Model
from homonim.errors import IoError
from homonim.fuse import RasterFuse
from homonim.kernel_model import KernelModel
from rasterio.features import shapes


@pytest.mark.parametrize('src_file, ref_file', [
        ('float_50cm_src_file', 'float_100cm_ref_file'),
        ('float_100cm_src_file', 'float_50cm_ref_file'),
    ]
) # yapf: disable
def test_creation(src_file, ref_file, tmp_path, request):
    """ Test creation and configuration of RasterFuse. """
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    # TODO: tidy comments, and add replacement test for process
    raster_fuse = RasterFuse(src_file, ref_file)
    with raster_fuse:
        assert (raster_fuse.proc_crs != ProcCrs.auto)
        assert (not raster_fuse.closed)
    assert raster_fuse.closed


@pytest.mark.parametrize('overwrite', [False, True])
def test_overwrite(tmp_path, float_50cm_src_file, float_100cm_ref_file, overwrite):
    """ Test overwrite behaviour. """
    corr_filename = tmp_path.joinpath('corrected.tif')
    param_filename = utils.create_param_filename(corr_filename)
    params = dict(
        corr_filename=corr_filename, param_filename=param_filename, model=Model.gain_blk_offset, kernel_shape=(5, 5),
        overwrite=overwrite, 
    )

    raster_fuse = RasterFuse(src_filename=float_50cm_src_file, ref_filename=float_100cm_ref_file)
    corr_filename.touch()
    with raster_fuse:
        if not overwrite:
            with pytest.raises(FileExistsError):
                raster_fuse.process(**params)
        else:
            raster_fuse.process(**params)

    os.remove(corr_filename)
    param_filename.touch()
    with raster_fuse:
        if not overwrite:
            with pytest.raises(FileExistsError):
                raster_fuse.process(**params)
        else:
            raster_fuse.process(**params)


@pytest.mark.parametrize(
    'src_file, ref_file, model, kernel_shape, max_block_mem', [
        ('float_45cm_src_file', 'float_100cm_ref_file', Model.gain, (1, 1), 2.e-4),
        ('float_45cm_src_file', 'float_100cm_ref_file', Model.gain_blk_offset, (1, 1), 1.e-3),
        ('float_45cm_src_file', 'float_100cm_ref_file', Model.gain_offset, (5, 5), 1.e-3),
        ('float_100cm_src_file', 'float_45cm_ref_file', Model.gain, (1, 1), 2.e-4),
        ('float_100cm_src_file', 'float_45cm_ref_file', Model.gain_blk_offset, (1, 1), 1.e-3),
        ('float_100cm_src_file', 'float_45cm_ref_file', Model.gain_offset, (5, 5), 1.e-3),
    ]
) # yapf: disable
def test_basic_fusion(src_file, ref_file, model, kernel_shape, max_block_mem, tmp_path, request):
    """ Test fusion output with different src/ref images, and model etc combinations. """
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    block_config = RasterFuse.create_config(max_block_mem=max_block_mem)
    corr_filename = tmp_path.joinpath('corrected.tif')
    raster_fuse = RasterFuse(src_file, ref_file)
    with raster_fuse:
        raster_fuse.process(corr_filename, model, kernel_shape, block_config=block_config)
    assert (corr_filename.exists())
    with rio.open(src_file, 'r') as src_ds, rio.open(corr_filename, 'r') as out_ds:
        src_array = src_ds.read(indexes=1)
        src_mask = src_ds.dataset_mask().astype('bool', copy=False)
        out_array = out_ds.read(indexes=1)
        out_mask = out_ds.dataset_mask().astype('bool', copy=False)
        assert (out_mask == src_mask).all()
        assert (out_array[out_mask] == pytest.approx(src_array[src_mask], abs=2))

@pytest.mark.parametrize(
    'out_profile', [
        dict(
            driver='GTiff', dtype='float32', nodata=float('nan'),
            creation_options=dict(
                tiled=True, blockxsize=512, blockysize=512, compress='deflate', interleave='band', photometric=None
            )
        ),
        dict(
            driver='GTiff', dtype='uint8', nodata=0,
            creation_options=dict(
                tiled=True, blockxsize=64, blockysize=64, compress='jpeg', interleave='pixel', photometric='ycbcr'
            )
        ),
        dict(driver='PNG', dtype='uint16', nodata=0, creation_options=dict()),
    ]
)  # yapf: disable
def test_out_profile(float_100cm_rgb_file, tmp_path, out_profile):
    """ Test fusion output image format (profile) with different out_profile configurations. """
    raster_fuse = RasterFuse(float_100cm_rgb_file, float_100cm_rgb_file)
    corr_filename = tmp_path.joinpath('corrected.tif')
    with raster_fuse:
        raster_fuse.process(corr_filename, Model.gain_blk_offset, (3, 3), out_profile=out_profile)
    assert (corr_filename.exists())
    out_profile.update(**out_profile['creation_options'])
    out_profile.pop('creation_options')
    with rio.open(float_100cm_rgb_file, 'r') as src_ds, rio.open(corr_filename, 'r') as fuse_ds:
        # test output image has been set with out_profile properties
        for k, v in out_profile.items():
            assert (
                (v is None and k not in fuse_ds.profile) or
                (fuse_ds.profile[k] == v) or
                (str(fuse_ds.profile[k]) == str(v))
            ) # yapf: disable

        # test output image has been set with src image properties not in out_profile
        if src_ds.profile['driver'].lower() == out_profile['driver'].lower():
            # source image keys including driver specific creation options, not present in out_profile
            src_keys = set(src_ds.profile.keys()).difference(out_profile.keys())
        else:
            # source image keys excluding driver specific creation options, not present in out_profile
            src_keys = {'width', 'height', 'count', 'dtype', 'crs', 'transform'}.difference(out_profile.keys())
        for k in src_keys:
            v = src_ds.profile[k]
            assert (
                (v is None and k not in fuse_ds.profile) or
                (fuse_ds.profile[k] == v) or
                (str(fuse_ds.profile[k]) == str(v))
            ) # yapf: disable


@pytest.mark.parametrize(
    'model, proc_crs', [
        (Model.gain, ProcCrs.ref),
        (Model.gain_blk_offset, ProcCrs.ref),
        (Model.gain_offset, ProcCrs.ref),
        (Model.gain, ProcCrs.src),
        (Model.gain_blk_offset, ProcCrs.src),
        (Model.gain_offset, ProcCrs.src),
    ]
) # yapf: disable
def test_param_image(float_100cm_rgb_file, tmp_path, model, proc_crs):
    """ Test creation and masking of parameter image for different model and proc_crs combinations. """
    corr_filename = tmp_path.joinpath('corrected.tif')
    param_filename = utils.create_param_filename(corr_filename)
    raster_fuse = RasterFuse(float_100cm_rgb_file, float_100cm_rgb_file, proc_crs=proc_crs)
    with raster_fuse:
        raster_fuse.process(corr_filename, model, (5, 5), param_filename=param_filename)

    assert (param_filename.exists())

    with rio.open(float_100cm_rgb_file, 'r') as ref_src_ds, rio.open(param_filename, 'r') as param_ds:
        assert (param_ds.count == ref_src_ds.count * 3)
        param_mask = param_ds.dataset_mask().astype('bool', copy=False)
        src_ref_mask = ref_src_ds.dataset_mask().astype('bool', copy=False)
        assert (param_mask == src_ref_mask).all()


@pytest.mark.parametrize(
    'src_file, ref_file, kernel_shape, proc_crs, mask_partial', [
        ('float_45cm_src_file', 'float_100cm_ref_file', (1, 1), ProcCrs.auto, False),
        ('float_45cm_src_file', 'float_100cm_ref_file', (1, 1), ProcCrs.auto, True),
        ('float_45cm_src_file', 'float_100cm_ref_file', (3, 3), ProcCrs.auto, True),
        ('float_100cm_src_file', 'float_45cm_ref_file', (1, 1), ProcCrs.auto, False),
        ('float_100cm_src_file', 'float_45cm_ref_file', (1, 1), ProcCrs.auto, True),
        ('float_100cm_src_file', 'float_45cm_ref_file', (3, 3), ProcCrs.auto, True),
    ]
) # yapf: disable
def test_mask_partial(src_file, ref_file, tmp_path, kernel_shape, proc_crs, mask_partial, request):
    """ Test partial masking with multiple image blocks. """
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    model_config = KernelModel.create_config(mask_partial=mask_partial)
    block_config = RasterFuse.create_config(max_block_mem=1.e-1)
    corr_file = tmp_path.joinpath('corrected.tif')
    raster_fuse = RasterFuse(src_file, ref_file, proc_crs=proc_crs)
    with raster_fuse:
        raster_fuse.process(
            corr_file, Model.gain_blk_offset, kernel_shape, model_config=model_config, block_config=block_config
        )
    assert (corr_file.exists())
    with rio.open(src_file, 'r') as src_ds, rio.open(corr_file, 'r') as fuse_ds:
        fuse_mask = fuse_ds.dataset_mask().astype('bool', copy=False)
        src_mask = src_ds.dataset_mask().astype('bool', copy=False)
        if not mask_partial:
            assert (fuse_mask == src_mask).all()
        else:
            assert (fuse_mask.sum() < src_mask.sum())
            assert (fuse_mask.sum() > 0)
            assert (src_mask[fuse_mask]).all()
            # check that the output mask consists of 1 blob
            out_mask_shapes = [shape for shape in shapes(fuse_mask.astype('uint8', copy=False), mask=fuse_mask)]
            assert (len(out_mask_shapes) == 1)


def test_build_overviews(float_50cm_ref_file, tmp_path):
    """ Test that overviews are built for corrected and parameter files. """
    corr_filename = tmp_path.joinpath('corrected.tif')
    param_filename = utils.create_param_filename(corr_filename)
    raster_fuse = RasterFuse(float_50cm_ref_file, float_50cm_ref_file)

    # replace raster_fuse.build_overviews() with a test_build_overviews() that forces min_level_pixels==1, otherwise
    # overviews won't be built for the small test raster
    orig_build_overviews = raster_fuse._build_overviews
    def test_build_overviews(im):
        orig_build_overviews(im, min_level_pixels=1)
    raster_fuse._build_overviews = test_build_overviews

    with raster_fuse:
        raster_fuse.process(
            corr_filename, Model.gain_blk_offset, (3, 3), param_filename=param_filename, build_ovw=True
        )

    assert (corr_filename.exists())
    assert (param_filename.exists())

    with rio.open(corr_filename, 'r') as fuse_ds:
        assert (len(fuse_ds.overviews(1)) > 0)

    with rio.open(param_filename, 'r') as param_ds:
        for band_i in param_ds.indexes:
            assert (len(param_ds.overviews(band_i)) > 0)


def test_io_error(tmp_path, float_50cm_ref_file):
    """ Test we get an IoError if processing without opening/entering the context. """
    raster_fuse = RasterFuse(float_50cm_ref_file, float_50cm_ref_file)
    with pytest.raises(IoError):
        raster_fuse.process(tmp_path, Model.gain_blk_offset, (3, 3))


def test_corr_filename(tmp_path, float_50cm_ref_file):
    """ Test corrected file is created. """
    corr_filename = tmp_path.joinpath('corrected.tif')
    raster_fuse = RasterFuse(float_50cm_ref_file, float_50cm_ref_file)
    with raster_fuse:
        raster_fuse.process(corr_filename, Model.gain_blk_offset, (3, 3))

    assert (corr_filename.exists())


def test_single_thread(tmp_path, float_50cm_ref_file):
    """ Test single-threaded processing creates a corrected file. """
    block_config = RasterFuse.create_config(threads=1)
    corr_filename = tmp_path.joinpath('corrected.tif')
    raster_fuse = RasterFuse(float_50cm_ref_file, float_50cm_ref_file)
    with raster_fuse:
        raster_fuse.process(corr_filename, Model.gain_blk_offset, (3, 3), block_config=block_config)

    assert (corr_filename.exists())


@pytest.mark.parametrize(
    'src_file, ref_file, proc_crs, exp_proc_crs', [
        ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, ProcCrs.ref),
        ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.src, ProcCrs.src),
        ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.auto, ProcCrs.src),
        ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.ref, ProcCrs.ref),
    ]
) # yapf: disable
def test_proc_crs(tmp_path, src_file, ref_file, proc_crs, exp_proc_crs, request):
    """ Test corrected file creation for forced and auto proc_crs with different src/ref combinations. """
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    corr_filename = tmp_path.joinpath('corrected.tif')
    raster_fuse = RasterFuse(src_file, ref_file, proc_crs=proc_crs)
    assert (raster_fuse.proc_crs == exp_proc_crs)
    with raster_fuse:
        raster_fuse.process(corr_filename, Model.gain_blk_offset, (5, 5))
    assert (corr_filename.exists())


def test_tags(tmp_path, float_50cm_ref_file):
    """ Test corrected file metadata. """
    model = Model.gain_blk_offset
    kernel_shape = (3, 3)
    proc_crs = ProcCrs.ref
    block_config = RasterFuse.create_config()
    raster_fuse = RasterFuse(float_50cm_ref_file, float_50cm_ref_file, proc_crs=proc_crs)
    corr_filename = tmp_path.joinpath('corrected.tif')
    param_filename = utils.create_param_filename(corr_filename)
    with raster_fuse:
        raster_fuse.process(corr_filename, model, kernel_shape, param_filename=param_filename, block_config=block_config)

    assert (corr_filename.exists())
    assert (param_filename.exists())
    utils.validate_param_image(param_filename)

    with rio.open(corr_filename, 'r') as out_ds:
        tags = out_ds.tags()
        assert (
            {
                'FUSE_SRC_FILE', 'FUSE_REF_FILE', 'FUSE_MODEL', 'FUSE_KERNEL_SHAPE', 'FUSE_PROC_CRS',
                'FUSE_MAX_BLOCK_MEM', 'FUSE_THREADS',
                *{f'FUSE_{k.upper()}' for k in KernelModel.create_config().keys()},
            } <= set(tags)
        )
        assert (tags['FUSE_SRC_FILE'] == float_50cm_ref_file.name)
        assert (tags['FUSE_REF_FILE'] == float_50cm_ref_file.name)
        assert (tags['FUSE_MODEL'] == str(model.name))
        assert (tags['FUSE_PROC_CRS'] == str(proc_crs.name))
        assert (tags['FUSE_KERNEL_SHAPE'] == str(kernel_shape))
        for key,val in KernelModel.create_config().items():
            assert (tags[f'FUSE_{key.upper()}'] == val.name if hasattr(val, 'name') else str(val))
        assert (yaml.safe_load(tags['FUSE_MAX_BLOCK_MEM']) == block_config['max_block_mem'])
        assert (yaml.safe_load(tags['FUSE_THREADS']) == block_config['threads'])
