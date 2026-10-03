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

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from pytest import FixtureRequest
from rasterio import MemoryFile
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.windows import Window, union

from homonim import utils
from homonim.enums import ProcCrs
from homonim.errors import HomonimError
from homonim.raster_pair import RasterPairReader


@pytest.mark.parametrize(
    'src_file, ref_file, expected_proc_crs',
    [
        ('src_file_50cm_float', 'ref_file_100cm_float', ProcCrs.ref),
        ('src_file_100cm_float', 'ref_file_50cm_float', ProcCrs.src),
    ],
)
def test_init(
    src_file: str, ref_file: str, expected_proc_crs: ProcCrs, request: FixtureRequest
):
    """Test RasterPair initialisation and proc_crs resolution."""
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    raster_pair = RasterPairReader(src_file, ref_file)
    assert raster_pair.proc_crs == expected_proc_crs
    assert raster_pair.src_bands == (1,)
    assert raster_pair.ref_bands == (1,)

    # enter the context and test block(s) correspond to bands
    with raster_pair as rp:
        block_pairs = list(rp.block_pairs())
        assert len(block_pairs) == 1
        assert block_pairs[0].src_out_block == Window(
            0, 0, rp.src_im.width, rp.src_im.height
        )
    assert rp.closed


def test_coverage_error(ref_file_100cm_float, src_file_100cm_float):
    """Test an error is raised when the reference does not cover the source."""
    with pytest.raises(HomonimError, match='does not cover'):
        _ = RasterPairReader(ref_file_100cm_float, src_file_100cm_float)


def test_band_count_error(file_rgba, file_byte):
    """Test an error is raised when there are more source than reference bands."""
    with pytest.raises(HomonimError, match='fewer non-alpha bands'):
        _ = RasterPairReader(file_rgba, file_byte)


def test_block_shape_error(src_file_50cm_float, ref_file_100cm_float):
    """Test block shape errors."""
    # test auto block shape smaller than a pixel
    with RasterPairReader(src_file_50cm_float, ref_file_100cm_float) as rp:
        with pytest.raises(HomonimError, match='smaller than a pixel'):
            _ = list(rp.block_pairs(max_block_mem=1.0e-5))

    # test auto block shape smaller than overlap
    with RasterPairReader(src_file_50cm_float, ref_file_100cm_float) as rp:
        with pytest.raises(HomonimError, match='smaller than the overlap'):
            _ = list(rp.block_pairs(overlap=(5, 5), max_block_mem=1.0e-4))


def test_non_nup_error(src_file_100cm_float, src_file_sup_100cm_float):
    """Test non North-up images raise errors."""
    with pytest.raises(HomonimError, match='North-up'):
        with RasterPairReader(src_file_100cm_float, src_file_sup_100cm_float):
            pass
    with pytest.raises(HomonimError, match='North-up'):
        with RasterPairReader(src_file_sup_100cm_float, src_file_100cm_float):
            pass


def test_ctx_closed_error(src_file_50cm_float, ref_file_100cm_float):
    """Test that using a closed context raises an error."""
    rp = RasterPairReader(src_file_50cm_float, ref_file_100cm_float)
    rp.close()
    with pytest.raises(RuntimeError, match='closed'):
        _ = list(rp.block_pairs())
    with pytest.raises(RuntimeError, match='closed'):
        _ = list(rp.read())
    with pytest.raises(RuntimeError, match='closed'):
        with rp:
            pass


@pytest.mark.parametrize(
    'src_file, ref_file, overlap, max_block_mem',
    [
        ('src_file_45cm_float', 'ref_file_100cm_float', (0, 0), 1.0e-3),
        ('src_file_45cm_float', 'ref_file_100cm_float', (2, 2), 1.0e-3),
        ('src_file_45cm_float', 'ref_file_100cm_float', (0, 0), 2.0e-4),
        ('src_file_45cm_float', 'ref_file_100cm_float', (1, 1), 2.0e-4),
        ('src_file_100cm_float', 'ref_file_45cm_float', (0, 0), 1.0e-3),
        ('src_file_100cm_float', 'ref_file_45cm_float', (2, 2), 1.0e-3),
        ('src_file_100cm_float', 'ref_file_45cm_float', (0, 0), 2.0e-4),
        ('src_file_100cm_float', 'ref_file_45cm_float', (1, 1), 2.0e-4),
    ],
)
def test_block_pair_continuity(
    src_file: str,
    ref_file: str,
    overlap: tuple[int, int],
    max_block_mem: float,
    request: FixtureRequest,
):
    """Test the continuity of block pairs for different source and reference etc.
    combinations.
    """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)

    def compare_blocks(
        block: Window,
        prev_block: Window,
        overlap: tuple[int, int] = (0, 0),
        compare: Callable = np.equal,
    ):
        """Test block continuity."""
        if block.row_off == prev_block.row_off:  # blocks in the same row
            assert compare(
                block.col_off, prev_block.col_off + prev_block.width - 2 * overlap[1]
            )
        else:
            assert compare(
                block.row_off, prev_block.row_off + prev_block.height - 2 * overlap[0]
            )

    with RasterPairReader(src_file, ref_file) as rp:
        # Create lists of compare_blocks() parameters for each block in a BlockPair.
        # NOTE: the <other crs>_in_block may overlap more than overlap, but
        # <proc_crs>_in_block's should overlap by exactly overlap, and *out_blocks
        # should be exactly adjacent.  max_block_mem and overlap should be chosen to
        # give a block shape > 2*overlap.
        block_keys = ['src_in_block', 'ref_in_block', 'src_out_block', 'ref_out_block']
        # *in_blocks overlap, *out_blocks don't
        overlaps = [overlap, overlap, (0, 0), (0, 0)]
        if rp.proc_crs is ProcCrs.ref:
            # the src_in_block can overlap by more than overlap, the other blocks
            # should be exact
            compares = [np.less_equal, np.equal, np.equal, np.equal]
        else:
            # the ref_in_block can overlap by more than overlap, the other blocks
            # should be exact
            compares = [np.equal, np.less_equal, np.equal, np.equal]

        block_pairs = list(rp.block_pairs(overlap=overlap, max_block_mem=max_block_mem))
        prev_block_pair = block_pairs[0]
        for block_pair in block_pairs[1:]:
            if block_pair.band_i == prev_block_pair.band_i:
                # compare each block type with its previous version
                for block_key, overlap, compare in zip(
                    block_keys, overlaps, compares, strict=True
                ):
                    block = getattr(block_pair, block_key)
                    prev_block = getattr(prev_block_pair, block_key)
                    compare_blocks(block, prev_block, overlap, compare)
            else:
                assert block_pair.band_i == prev_block_pair.band_i + 1
            prev_block_pair = block_pair


@pytest.mark.parametrize(
    'src_file, ref_file, overlap, max_block_mem',
    [
        ('src_file_45cm_float', 'ref_file_100cm_float', (0, 0), 1.0e-3),
        ('src_file_45cm_float', 'ref_file_100cm_float', (2, 2), 1.0e-3),
        ('src_file_45cm_float', 'ref_file_100cm_float', (0, 0), 2.0e-4),
        ('src_file_45cm_float', 'ref_file_100cm_float', (2, 2), 2.0e-4),
        ('src_file_100cm_float', 'ref_file_45cm_float', (0, 0), 1.0e-3),
        ('src_file_100cm_float', 'ref_file_45cm_float', (2, 2), 1.0e-3),
        ('src_file_100cm_float', 'ref_file_45cm_float', (0, 0), 2.0e-4),
        ('src_file_100cm_float', 'ref_file_45cm_float', (2, 2), 2.0e-4),
    ],
)
def test_block_pair_coverage(
    src_file: str,
    ref_file: str,
    overlap: tuple[int, int],
    max_block_mem: float,
    request: FixtureRequest,
):
    """Test that combined block pairs cover the processing window for different
    source and reference etc combinations.
    """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)

    with RasterPairReader(src_file, ref_file) as rp:
        block_pairs = list(rp.block_pairs(overlap=overlap, max_block_mem=max_block_mem))
        # a dict to hold combined windows
        accum_block_pair = block_pairs[0]._asdict()

        # find the combined windows for the block pairs
        for block_pair in block_pairs[1:]:
            for field in [
                'src_in_block',
                'ref_in_block',
                'src_out_block',
                'ref_out_block',
            ]:
                accum_block_pair[field] = union(
                    getattr(block_pair, field), accum_block_pair[field]
                )

        # test coverage of the combined windows
        if rp.proc_crs is ProcCrs.ref:
            assert accum_block_pair['ref_in_block'] == rp._ref_win
            assert accum_block_pair['ref_out_block'] == rp._ref_win
            src_win = Window(0, 0, rp.src_im.width, rp.src_im.height)
            assert accum_block_pair['src_in_block'].intersection(src_win) == src_win
            assert accum_block_pair['src_out_block'].intersection(src_win) == src_win
        elif rp.proc_crs is ProcCrs.src:
            assert accum_block_pair['src_in_block'] == rp._src_win
            assert accum_block_pair['src_out_block'] == rp._src_win
            assert (
                accum_block_pair['ref_in_block'].intersection(rp._ref_win)
                == rp._ref_win
            )
            assert (
                accum_block_pair['ref_out_block'].intersection(rp._ref_win)
                == rp._ref_win
            )


@pytest.mark.parametrize(
    'src_file, ref_file, overlap, max_block_mem',
    [
        ('src_file_45cm_float', 'ref_file_100cm_float', (0, 0), 1.0e-3),
        ('src_file_45cm_float', 'ref_file_100cm_float', (2, 2), 1.0e-3),
        ('src_file_45cm_float', 'ref_file_100cm_float', (0, 0), 2.0e-4),
        ('src_file_45cm_float', 'ref_file_100cm_float', (2, 2), 2.0e-4),
        ('src_file_100cm_float', 'ref_file_45cm_float', (0, 0), 1.0e-3),
        ('src_file_100cm_float', 'ref_file_45cm_float', (2, 2), 1.0e-3),
        ('src_file_100cm_float', 'ref_file_45cm_float', (0, 0), 2.0e-4),
        ('src_file_100cm_float', 'ref_file_45cm_float', (2, 2), 2.0e-4),
    ],
)
def test_block_pair_io(
    src_file: str,
    ref_file: str,
    overlap: tuple[int, int],
    max_block_mem: float,
    request: FixtureRequest,
):
    """
    Test block pairs can be read, reprojected and written as RasterArrays without
    loss of data.

    This is more an integration test with RasterArray than a RasterPairReader unit
    test. It simulates the way RasterArrays are reprojected in *KernelModel.
    """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)

    # test re-projections from src->ref->src and ref->src->ref
    for reproj_ra in ['src', 'ref']:
        with (
            RasterPairReader(src_file, ref_file) as rp,
            MemoryFile() as src_mf,
            MemoryFile() as ref_mf,
        ):
            if rp.proc_crs is ProcCrs.ref:
                ref_sampling = Resampling.average
                src_sampling = Resampling.cubic_spline
            else:
                ref_sampling = Resampling.cubic_spline
                src_sampling = Resampling.average

            with (
                src_mf.open(**rp.src_im.meta) as test_src_ds,
                ref_mf.open(**rp.ref_im.meta) as test_ref_ds,
            ):
                # read, reproject and write block pairs to their respective datasets
                for block_pair in rp.block_pairs(
                    overlap=overlap, max_block_mem=max_block_mem
                ):
                    src_ra, ref_ra = rp.read(block_pair)
                    if reproj_ra == 'src':
                        src_ra_ = src_ra.reproject_like(ref_ra, resampling=ref_sampling)
                        src_ra__ = src_ra_.reproject_like(
                            src_ra, resampling=src_sampling
                        )
                        src_ra_.to_rio_dataset(
                            test_ref_ds, window=block_pair.ref_out_block
                        )
                        src_ra__.to_rio_dataset(
                            test_src_ds, window=block_pair.src_out_block
                        )
                    else:
                        ref_ra_ = ref_ra.reproject_like(src_ra, resampling=src_sampling)
                        ref_ra__ = ref_ra_.reproject_like(
                            ref_ra, resampling=ref_sampling
                        )
                        ref_ra__.to_rio_dataset(
                            test_ref_ds, window=block_pair.ref_out_block
                        )
                        ref_ra_.to_rio_dataset(
                            test_src_ds, window=block_pair.src_out_block
                        )

            # test the written datasets contain same valid areas as the original
            # source and reference files
            with rio.open(src_file, 'r') as src_ds, src_mf.open() as test_src_ds:
                src_mask = src_ds.read_masks(indexes=1).view('bool')
                test_mask = test_src_ds.read_masks(indexes=1).view('bool')
                assert (test_mask[src_mask]).all()

            with rio.open(ref_file, 'r') as ref_ds, ref_mf.open() as test_ref_ds:
                ref_mask = ref_ds.read_masks(indexes=1).view('bool')
                test_mask = test_ref_ds.read_masks(indexes=1).view('bool')
                assert (test_mask[ref_mask]).all()


@pytest.mark.parametrize(
    'src_file, ref_file, proc_crs',
    [
        ('src_file_100cm_float', 'ref_file_100cm_float', ProcCrs.ref),
        ('src_file_100cm_float', 'ref_file_100cm_float', ProcCrs.src),
        ('src_file_wgs84_100cm_float', 'ref_file_100cm_float', ProcCrs.ref),
        ('src_file_wgs84_100cm_float', 'ref_file_100cm_float', ProcCrs.src),
        ('src_file_100cm_float', 'ref_file_wgs84_100cm_float', ProcCrs.ref),
        ('src_file_100cm_float', 'ref_file_wgs84_100cm_float', ProcCrs.src),
    ],
)
def test_crs(src_file: str, ref_file: str, proc_crs: ProcCrs, request: FixtureRequest):
    """Test data is not lost when reprojecting blocks with source and reference in
    different CRSs.
    """
    src_file: Path = request.getfixturevalue(src_file)
    ref_file: Path = request.getfixturevalue(ref_file)
    with RasterPairReader(src_file, ref_file, proc_crs=proc_crs) as rp:
        for block_pair in rp.block_pairs():
            src_ra, ref_ra = rp.read(block_pair)
            src_ra_ = src_ra.reproject_like(ref_ra, resampling=Resampling.bilinear)
            assert ref_ra.array == pytest.approx(src_ra_.array, abs=1, nan_ok=True)
            assert np.all(src_ra_.mask()[ref_ra.mask()])
            ref_ra_ = ref_ra.reproject_like(src_ra, resampling=Resampling.bilinear)
            assert src_ra.array == pytest.approx(ref_ra_.array, abs=1, nan_ok=True)
            assert np.all(ref_ra_.mask()[src_ra.mask()])


def test_url():
    """Test source and reference as URLs rather than files."""
    data_url = 'https://raw.githubusercontent.com/leftfield-geospatial/homonim/main/tests/data/'
    ngi_url = data_url + 'source/ngi_rgb_byte_1.tif'
    modis_url = data_url + '/reference/modis_nbar.tif'
    with RasterPairReader(ngi_url, modis_url) as rp:
        assert not rp.closed


def test_file_exists_error(ngi_src_file, landsat_ref_file):
    """Test an error is raised if a source or reference file doesn't exist."""
    for src_file, ref_file in zip(
        [ngi_src_file, 'dummy.tif'], ['dummy.tif', landsat_ref_file], strict=True
    ):
        with pytest.raises(RasterioIOError, match='No such file or directory'):
            with RasterPairReader(src_file, ref_file):
                pass
