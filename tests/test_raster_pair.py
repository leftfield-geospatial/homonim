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

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import rasterio as rio
from pytest import FixtureRequest
from rasterio import MemoryFile
from rasterio.enums import Resampling
from rasterio.windows import Window, union

from homonim.enums import ProcCrs
from homonim.errors import ImageContentError, BlockSizeError, IoError
from homonim.raster_pair import RasterPairReader


@pytest.mark.parametrize(
    'src_file, ref_file, expected_proc_crs', [
        ('float_50cm_src_file', 'float_100cm_ref_file', ProcCrs.ref),
        ('float_100cm_src_file', 'float_50cm_ref_file', ProcCrs.src)
    ]
)  # yapf: disable
def test_creation(src_file: str, ref_file: str, expected_proc_crs: ProcCrs, request: FixtureRequest):
    """ Test RasterPair creation and proc_crs resolution. """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    raster_pair = RasterPairReader(src_file, ref_file)
    assert (raster_pair.proc_crs == expected_proc_crs)
    assert (raster_pair.src_bands == (1,))
    assert (raster_pair.ref_bands == (1,))

    # enter the raster pair context and test block(s) correspond to bands
    with raster_pair as rp:
        block_pairs = list(rp.block_pairs())
        assert (len(block_pairs) == 1)
        assert (block_pairs[0].src_out_block == Window(0, 0, rp.src_im.width, rp.src_im.height))
    assert rp.closed


@pytest.mark.parametrize(
    'src_file, ref_file', [
        ('float_50cm_ref_file', 'float_100cm_src_file'),
        ('float_50cm_ref_file', 'float_50cm_src_file'),
        ('float_100cm_ref_file', 'float_50cm_src_file'),
        ('float_100cm_ref_file', 'float_100cm_src_file')
    ]
)  # yapf: disable
def test_coverage_exception(src_file: str, ref_file: str, request: FixtureRequest):
    """ Test that ref not covering the extent of src raises an error. """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    with pytest.raises(ImageContentError):
        _ = RasterPairReader(src_file, ref_file)


def test_band_count_exception(rgba_file: Path, byte_file: Path):
    """ Test that src band count > ref band count raises an error. """
    with pytest.raises(ImageContentError):
        _ = RasterPairReader(rgba_file, byte_file)


def test_block_shape_exception(float_50cm_src_file: Path, float_100cm_ref_file: Path):
    """ Test block shape errors. """
    # test auto block shape smaller than a pixel
    raster_pair = RasterPairReader(float_50cm_src_file, float_100cm_ref_file)
    with pytest.raises(BlockSizeError):
        with raster_pair:
            block_pairs = [block_pair for block_pair in raster_pair.block_pairs(max_block_mem=1.e-5)]

    # test auto block shape smaller than overlap
    with pytest.raises(BlockSizeError):
        with raster_pair:
            block_pairs = [block_pair for block_pair in raster_pair.block_pairs(overlap=(5, 5), max_block_mem=1.e-4)]


def test_not_open_exception(float_50cm_src_file: Path, float_100cm_ref_file: Path):
    """ Test that using a raster pair before entering the context raises an error. """
    raster_pair = RasterPairReader(float_50cm_src_file, float_100cm_ref_file)
    with pytest.raises(IoError):
        _ = raster_pair.ref_im
    with pytest.raises(IoError):
        _ = raster_pair.src_im
    with pytest.raises(IoError):
        _ = list(raster_pair.block_pairs())
    with pytest.raises(IoError):
        _ = raster_pair.read(None)


@pytest.mark.parametrize(
    'src_file, ref_file, proc_crs, blk_overlap, max_block_mem', [
        ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
        ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
        ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 2.e-4),
        ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (1, 1), 2.e-4),
        ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
        ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
        ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 2.e-4),
        ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (1, 1), 2.e-4),
    ]
)  # yapf: disable
def test_block_pair_continuity(
    src_file: str, ref_file: str, proc_crs: ProcCrs, blk_overlap: Tuple[int, int], max_block_mem: float,
    request: FixtureRequest
):
    """ Test the continuity of block pairs for different src/ref etc combinations. """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    raster_pair = RasterPairReader(src_file, ref_file, proc_crs=proc_crs)

    def compare_blocks(block: Window, prev_block: Window, overlap: Tuple[int, int] = (0, 0), compare=np.equal):
        """ Test block continuity. """
        if block.row_off == prev_block.row_off:  # blocks in the same row
            assert compare(block.col_off, prev_block.col_off + prev_block.width - 2 * overlap[1])
        else:
            assert compare(block.row_off, prev_block.row_off + prev_block.height - 2 * overlap[0])

    with raster_pair:
        # Create lists of compare_blocks() parameters for each block in a BlockPair.
        # NOTE: the <'other' crs>_in_block may overlap more than ``overlap`, otherwise <proc_crs>_in_block's should
        # overlap by exactly blk_overlap, and *out_blocks should be exactly adjacent.  max_block_mem, and blk_overlap
        # should however be chosen to give a block shape > 2*overlap.
        block_keys = ['src_in_block', 'ref_in_block', 'src_out_block', 'ref_out_block']
        # *in_blocks overlap, *out_blocks don't
        overlaps = [blk_overlap, blk_overlap, (0, 0), (0, 0)]
        if raster_pair.proc_crs == ProcCrs.ref:
            # the src_in_block can overlap by more than blk_overlap, the other blocks should be exact
            compares = [np.less_equal, np.equal, np.equal, np.equal]
        else:
            # the ref_in_block can overlap by more than blk_overlap, the other blocks should be exact
            compares = [np.equal, np.less_equal, np.equal, np.equal]
        block_pairs = list(raster_pair.block_pairs(overlap=blk_overlap, max_block_mem=max_block_mem))
        prev_block_pair = block_pairs[0]
        for block_pair in block_pairs[1:]:
            if block_pair.band_i == prev_block_pair.band_i:  # blocks in the same band
                # compare each block type with its previous version, using the appropriate compare_block params
                for block_key, overlap, compare in zip(block_keys, overlaps, compares):
                    block = block_pair.__getattribute__(block_key)
                    prev_block = prev_block_pair.__getattribute__(block_key)
                    compare_blocks(block, prev_block, overlap, compare)
            else:
                assert (block_pair.band_i == prev_block_pair.band_i + 1)
            prev_block_pair = block_pair


@pytest.mark.parametrize(
    'src_file, ref_file, proc_crs, overlap, max_block_mem', [
        ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
        ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
        ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 2.e-4),
        ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 2.e-4),
        ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
        ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
        ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 2.e-4),
        ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (2, 2), 2.e-4),
    ]
)  # yapf: disable
def test_block_pair_coverage(
    src_file: str, ref_file: str, proc_crs: ProcCrs, overlap: Tuple[int, int], max_block_mem: float,
    request: FixtureRequest,
):
    """ Test that combined block pairs cover the processing window for different src/ref etc combinations. """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    raster_pair = RasterPairReader(src_file, ref_file, proc_crs=proc_crs)
    with raster_pair:
        block_pairs = list(raster_pair.block_pairs(overlap=overlap, max_block_mem=max_block_mem))
        accum_block_pair = block_pairs[0]._asdict()  # a dict to hold combined windows

        # find the combined windows for the block pairs
        for block_pair in block_pairs[1:]:
            for field in ['src_in_block', 'ref_in_block', 'src_out_block', 'ref_out_block']:
                accum_block_pair[field] = union(block_pair.__getattribute__(field), accum_block_pair[field])

        # test coverage of the combined windows
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


@pytest.mark.parametrize(
    'src_file, ref_file, proc_crs, overlap, max_block_mem', [
        ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
        ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
        ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (0, 0), 2.e-4),
        ('float_45cm_src_file', 'float_100cm_ref_file', ProcCrs.auto, (2, 2), 2.e-4),
        ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 1.e-3),
        ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (2, 2), 1.e-3),
        ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (0, 0), 2.e-4),
        ('float_100cm_src_file', 'float_45cm_ref_file', ProcCrs.auto, (2, 2), 2.e-4),
    ]
)  # yapf: disable
def test_block_pair_io(
    src_file: str, ref_file: str, proc_crs: ProcCrs, overlap: Tuple[int, int], max_block_mem: float,
    request: FixtureRequest,
):
    """
    Test block pairs can be read, reprojected and written as raster arrays without loss of data

    This is more an integration test with raster array than a raster pair unit test.  It simulates the way raster
    arrays are reprojected in *KernelModel.    .
    """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    raster_pair = RasterPairReader(src_file, ref_file, proc_crs=proc_crs)

    if raster_pair.proc_crs == ProcCrs.ref:
        ref_sampling = Resampling.average
        src_sampling = Resampling.cubic_spline
    else:
        ref_sampling = Resampling.cubic_spline
        src_sampling = Resampling.average

    # test re-projections from src->ref->src and ref->src->ref for both proc_crs=ref & src
    for reproj_ra in ['src', 'ref']:
        # create src and ref test datasets for writing, and enter the raster pair context
        with MemoryFile() as src_mf, MemoryFile() as ref_mf, raster_pair:
            with src_mf.open(**raster_pair.src_im.profile) as src_ds, ref_mf.open(
                **raster_pair.ref_im.profile
            ) as ref_ds:
                # read, reproject and write block pairs to their respective datasets
                block_pairs = list(raster_pair.block_pairs(overlap=overlap, max_block_mem=max_block_mem))
                for block_pair in block_pairs:
                    src_ra, ref_ra = raster_pair.read(block_pair)
                    if reproj_ra == 'src':
                        _src_ra = src_ra.reproject(**ref_ra.proj_profile, resampling=ref_sampling)
                        __src_ra = _src_ra.reproject(**src_ra.proj_profile, resampling=src_sampling)
                        _src_ra.to_rio_dataset(ref_ds, indexes=1, window=block_pair.ref_out_block)
                        __src_ra.to_rio_dataset(src_ds, indexes=1, window=block_pair.src_out_block)
                    else:
                        _ref_ra = ref_ra.reproject(**src_ra.proj_profile, resampling=src_sampling)
                        __ref_ra = _ref_ra.reproject(**ref_ra.proj_profile, resampling=ref_sampling)
                        __ref_ra.to_rio_dataset(ref_ds, indexes=1, window=block_pair.ref_out_block)
                        _ref_ra.to_rio_dataset(src_ds, indexes=1, window=block_pair.src_out_block)

            # test the written datasets contain same valid areas as the original src/ref files
            with rio.open(src_file, 'r') as src_ds, src_mf.open() as test_src_ds:
                src_mask = src_ds.dataset_mask().astype('bool', copy=False)
                test_mask = test_src_ds.dataset_mask().astype('bool', copy=False)
                assert (test_mask[src_mask]).all()

            with rio.open(ref_file, 'r') as ref_ds, ref_mf.open() as test_ref_ds:
                ref_mask = ref_ds.dataset_mask().astype('bool', copy=False)
                test_mask = test_ref_ds.dataset_mask().astype('bool', copy=False)
                assert (test_mask[ref_mask]).all()
