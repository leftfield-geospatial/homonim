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
from rasterio.windows import Window, union
from rasterio.enums import Resampling
from rasterio import MemoryFile

from homonim.enums import ProcCrs
from homonim.errors import ImageContentError, BlockSizeError, IoError
from homonim.raster_pair import RasterPairReader
from homonim.raster_array import RasterArray


@pytest.mark.parametrize('src_file, ref_file, expected_proc_crs', [
    ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.ref),
    ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.src)
])
def test_creation(src_file, ref_file, expected_proc_crs, request):
    """Test RasterPair creation and proc_crs resolution"""
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    raster_pair = RasterPairReader(src_file, ref_file)
    assert (raster_pair.proc_crs == expected_proc_crs)
    assert (raster_pair.src_bands == (1,))
    assert (raster_pair.ref_bands == (1,))
    with raster_pair as rp:
        block_pairs = list(rp.block_pairs())
        assert (len(block_pairs) == 1)
        assert (block_pairs[0].src_out_block == Window(0, 0, rp.src_im.width, rp.src_im.height))
    assert (rp.closed)


@pytest.mark.parametrize('src_file, ref_file', [
    ('float_50cm_ref_file', 'float_100cm_src_file'),
    ('float_50cm_ref_file', 'float_50cm_src_file'),
    ('float_100cm_ref_file', 'float_50cm_src_file'),
    ('float_100cm_ref_file', 'float_100cm_src_file')
])
def test_coverage_exception(src_file, ref_file, request):
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    with pytest.raises(ImageContentError):
        _ = RasterPairReader(src_file, ref_file)


def test_band_count_exception(rgba_file, byte_file):
    with pytest.raises(ImageContentError):
        _ = RasterPairReader(rgba_file, byte_file)


def test_block_shape_exception(float_50cm_src_file, float_100cm_ref_file):
    with pytest.raises(BlockSizeError):
        _ = RasterPairReader(float_50cm_src_file, float_100cm_ref_file, max_block_mem=1.e-5)

    with pytest.raises(BlockSizeError):
        _ = RasterPairReader(float_50cm_src_file, float_100cm_ref_file, overlap=(5, 5), max_block_mem=1.e-4)


def test_not_open_exception(float_50cm_src_file, float_100cm_ref_file):
    raster_pair = RasterPairReader(float_50cm_src_file, float_100cm_ref_file)
    with pytest.raises(IoError):
        _ = raster_pair.ref_im
    with pytest.raises(IoError):
        _ = raster_pair.src_im
    with pytest.raises(IoError):
        _ = list(raster_pair.block_pairs())
    with pytest.raises(IoError):
        _ = raster_pair.read(None)


@pytest.mark.parametrize('src_file, ref_file, proc_crs, overlap, max_block_mem', [
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 2.e-4),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 2.e-4),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 2.e-4),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (2, 2), 2.e-4),
])
def test_block_pair_continuity(src_file, ref_file, proc_crs, overlap, max_block_mem, request):
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    raster_pair = RasterPairReader(src_file, ref_file, proc_crs=proc_crs, overlap=overlap, max_block_mem=max_block_mem)
    with raster_pair:
        block_pairs = list(raster_pair.block_pairs())
        prev_block_pair = block_pairs[0]
        for block_pair in block_pairs[1:]:
            if block_pair.band_i == prev_block_pair.band_i:
                for block_key in ['ref_in_block', 'src_in_block', 'src_out_block', 'ref_out_block']:
                    block = block_pair.__getattribute__(block_key)
                    prev_block = prev_block_pair.__getattribute__(block_key)
                    if block.row_off == prev_block.row_off:
                        assert (block.col_off <= prev_block.col_off + prev_block.width)
                    else:
                        assert (block.row_off <= prev_block.row_off + prev_block.height)
            else:
                assert (block_pair.band_i == prev_block_pair.band_i + 1)
            prev_block_pair = block_pair


@pytest.mark.parametrize('src_file, ref_file, proc_crs, overlap, max_block_mem', [
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 2.e-4),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 2.e-4),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 2.e-4),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (2, 2), 2.e-4),
])
def test_block_pair_coverage(src_file, ref_file, proc_crs, overlap, max_block_mem, request):
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    raster_pair = RasterPairReader(src_file, ref_file, proc_crs=proc_crs, overlap=overlap, max_block_mem=max_block_mem)
    with raster_pair:
        block_pairs = list(raster_pair.block_pairs())
        accum_block_pair = block_pairs[0]._asdict()

        for block_pair in block_pairs[1:]:
            for field in ['src_in_block', 'ref_in_block', 'src_out_block', 'ref_out_block']:
                accum_block_pair[field] = union(block_pair.__getattribute__(field), accum_block_pair[field])

        if raster_pair.proc_crs == ProcCrs.ref:
            assert (accum_block_pair['ref_in_block'] == raster_pair._ref_win)
            assert (accum_block_pair['ref_out_block'] == raster_pair._ref_win)
            src_win = Window(0, 0, raster_pair.src_im.width, raster_pair.src_im.height)
            assert (accum_block_pair['src_in_block'].intersection(src_win) == src_win)
            assert (accum_block_pair['src_out_block'].intersection(src_win) == src_win)
        elif raster_pair.proc_crs == ProcCrs.src:
            assert (accum_block_pair['src_in_block'] == raster_pair._src_win)
            assert (accum_block_pair['src_out_block'] == raster_pair._src_win)
            assert (accum_block_pair['ref_in_block'].intersection(raster_pair._ref_win) == raster_pair._ref_win)
            assert (accum_block_pair['ref_out_block'].intersection(raster_pair._ref_win) == raster_pair._ref_win)


@pytest.mark.parametrize('src_file, ref_file, proc_crs, overlap, max_block_mem', [
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 2.e-4),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 2.e-4),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 2.e-4),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (2, 2), 2.e-4),
])
def test_block_pair_io(src_file, ref_file, proc_crs, overlap, max_block_mem, request):
    """
    An RasterPairReader-RasterArray integration test for checking that blocks provided by RasterPairReader reprojected
    and written with RasterArray without loss of data.
    """
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    raster_pair = RasterPairReader(src_file, ref_file, proc_crs=proc_crs, overlap=overlap, max_block_mem=max_block_mem)
    with MemoryFile() as src_mf, MemoryFile() as ref_mf, raster_pair:
        with src_mf.open(**raster_pair.src_im.profile) as src_ds, ref_mf.open(**raster_pair.ref_im.profile) as ref_ds:
            block_pairs = list(raster_pair.block_pairs())
            for block_pair in block_pairs:
                src_ra, ref_ra = raster_pair.read(block_pair)
                if raster_pair.proc_crs == ProcCrs.ref:
                    _src_ra = src_ra.reproject(**ref_ra.proj_profile, resampling=Resampling.average)
                    __src_ra = _src_ra.reproject(**src_ra.proj_profile, resampling=Resampling.cubic_spline)
                    _src_ra.to_rio_dataset(ref_ds, indexes=1, window=block_pair.ref_out_block)
                    __src_ra.to_rio_dataset(src_ds, indexes=1, window=block_pair.src_out_block)
                else:
                    _ref_ra = ref_ra.reproject(**src_ra.proj_profile, resampling=Resampling.average)
                    __ref_ra = _ref_ra.reproject(**ref_ra.proj_profile, resampling=Resampling.cubic_spline)
                    __ref_ra.to_rio_dataset(ref_ds, indexes=1, window=block_pair.ref_out_block)
                    _ref_ra.to_rio_dataset(src_ds, indexes=1, window=block_pair.src_out_block)

        with rio.open(src_file, 'r') as src_ds, src_mf.open() as test_src_ds:
            src_mask = src_ds.dataset_mask().astype('bool', copy=False)
            test_mask = test_src_ds.dataset_mask().astype('bool', copy=False)
            assert (test_mask[src_mask]).all()

        with rio.open(ref_file, 'r') as ref_ds, ref_mf.open() as test_ref_ds:
            ref_mask = ref_ds.dataset_mask().astype('bool', copy=False)
            test_mask = test_ref_ds.dataset_mask().astype('bool', copy=False)
            assert (test_mask[ref_mask]).all()

