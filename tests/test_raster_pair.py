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

from typing import Tuple

import numpy as np
import pytest
from rasterio.windows import Window, union

from homonim.enums import ProcCrs
from homonim.errors import ImageContentError, BlockSizeError, IoError
from homonim.raster_pair import RasterPairReader


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


def _test_block_pair_continuity(raster_pair: RasterPairReader):
    def _test_block_continuity(block: Window, prev_block: Window, overlap: Tuple[int, int] = (0, 0), outer=False):
        if block.row_off == prev_block.row_off:
            if not outer:
                assert (block.col_off == prev_block.col_off + prev_block.width - 2 * overlap[1])
            else:
                # assume prev_block could also be outer and have been clipped to image extents
                assert (block.col_off <= prev_block.col_off + prev_block.width - overlap[1])
        else:
            assert (block.row_off == prev_block.row_off + prev_block.height - 2 * overlap[0])

    proc_overlap = np.array(raster_pair.overlap)
    block_pairs = list(raster_pair.block_pairs())
    res_ratio = np.divide(raster_pair.ref_im.res, raster_pair.src_im.res)
    other_overlap = proc_overlap * res_ratio if raster_pair.proc_crs == ProcCrs.ref else proc_overlap / res_ratio
    other_overlap = np.round(other_overlap).astype('int')
    src_overlap = other_overlap if raster_pair.proc_crs == ProcCrs.ref else proc_overlap
    ref_overlap = proc_overlap if raster_pair.proc_crs == ProcCrs.ref else other_overlap

    prev_block_pair = block_pairs[0]
    for block_pair in block_pairs[1:]:
        if block_pair.band_i == prev_block_pair.band_i:
            _test_block_continuity(block_pair.src_in_block, prev_block_pair.src_in_block, src_overlap, block_pair.outer)
            _test_block_continuity(block_pair.src_out_block, prev_block_pair.src_out_block, (0, 0), block_pair.outer)
            _test_block_continuity(block_pair.ref_in_block, prev_block_pair.ref_in_block, ref_overlap, block_pair.outer)
            _test_block_continuity(block_pair.ref_out_block, prev_block_pair.ref_out_block, (0, 0), block_pair.outer)
        else:
            assert (block_pair.band_i == prev_block_pair.band_i + 1)
        prev_block_pair = block_pair


@pytest.mark.parametrize('src_file, ref_file, proc_crs, overlap, max_block_mem', [
    ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
    ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (1, 1), 1.e-3),
    ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
    ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 1.e-4),
    ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (1, 1), 1.e-4),
    ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 1.e-4),
    ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
    ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.auto, (1, 1), 1.e-3),
    ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
    ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.auto, (0, 0), 1.e-4),
    ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.auto, (1, 1), 1.e-4),
    ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.auto, (2, 2), 1.e-4),
])
def test_block_pair_continuity(src_file, ref_file, proc_crs, overlap, max_block_mem, request):
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    raster_pair = RasterPairReader(src_file, ref_file, proc_crs=proc_crs, overlap=overlap, max_block_mem=max_block_mem)
    with raster_pair as rp:
        _test_block_pair_continuity(rp)


def _test_block_pair_coverage(raster_pair: RasterPairReader):
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
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (3, 3), 1.e-3),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (5, 5), 1.e-3),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 1.e-4),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (1, 1), 1.e-4),
    ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 1.e-4),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (3, 3), 1.e-3),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (5, 5), 1.e-3),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 1.e-4),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (1, 1), 1.e-4),
    ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (2, 2), 1.e-4),
])
def test_block_pair_coverage(src_file, ref_file, proc_crs, overlap, max_block_mem, request):
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    raster_pair = RasterPairReader(src_file, ref_file, proc_crs=proc_crs, overlap=overlap, max_block_mem=max_block_mem)
    with raster_pair as rp:
        _test_block_pair_coverage(rp)

# TO DO:
# - test exceptions in init for band mismatch, non-coverage
# - test block contigiousness for different overlaps
# - test reading blocks and that ref<->src block reprojections don't lose mask pixels for different overlaps.  use the non-aligned file(s) for this.
# - test resolution of proc-crs, and forcing.  and check the above two tests for all cases of proc-crs??
# - in the case of non-aligned image, check that src gets boundless window ?
# - test closed property working
# - test src and ref in different CRSs

# Exceptions
# - test 12 bit exception with synthetic file... except we won't be able to create one.
# - ref doesn't cover src
# - ref doesn't have enough bands
# - auto block shape < 1
# - IoError if using without opening
