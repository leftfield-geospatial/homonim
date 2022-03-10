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
import os
import pathlib
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
from click.testing import CliRunner
from rasterio.windows import Window
from tqdm import tqdm

from homonim import cli
from homonim import utils
from homonim.compare import RasterCompare
from homonim.enums import ProcCrs, Method
from homonim.fuse import RasterFuse
from homonim.kernel_model import KernelModel
from homonim.raster_pair import RasterPairReader
from tests.common import TestBase


@pytest.mark.parametrize('src_file, ref_file, method', [
    ('float_50cm_src_file', 'float_100cm_ref_file', Method.gain),
    ('float_50cm_src_file', 'float_100cm_ref_file', Method.gain_blk_offset),
    ('float_50cm_src_file', 'float_100cm_ref_file', Method.gain_offset),
    ('float_100cm_src_file', 'float_50cm_ref_file', Method.gain),
    ('float_100cm_src_file', 'float_50cm_ref_file', Method.gain_blk_offset),
    ('float_100cm_src_file', 'float_50cm_ref_file', Method.gain_offset),
])
def test_creation(src_file, ref_file, method, tmp_path, request):
    """Test creation of RasterFuse"""
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    model_config = KernelModel.default_config.copy()
    model_config.update(mask_partial=True)
    homo_config = RasterFuse.default_homo_config.copy()
    homo_config.update(param_image=True)
    out_profile = RasterFuse.default_out_profile.copy()
    out_profile.update(driver='HFA', creation_options={})

    raster_fuse = RasterFuse(src_file, ref_file, tmp_path, method, (5, 5), homo_config=homo_config,
                             model_config=model_config, out_profile=out_profile)
    with raster_fuse:
        assert (raster_fuse.method == method)
        assert (raster_fuse.kernel_shape == (5, 5))
        assert (raster_fuse.proc_crs != ProcCrs.auto)
        assert (raster_fuse.homo_filename is not None)
        assert (raster_fuse.param_filename is not None)
        assert (not raster_fuse.closed)

        assert (raster_fuse._config == homo_config)
        assert (raster_fuse._model_config == model_config)
        for k, v in model_config.items():
            assert (raster_fuse._model.__getattribute__(f'_{k}') == v)
        assert (raster_fuse._out_profile == out_profile)

    assert (raster_fuse.closed)


@pytest.mark.parametrize('overwrite', [False, True])
def test_set_overwrite(tmp_path, float_50cm_src_file, float_100cm_ref_file, overwrite):
    homo_config = RasterFuse.default_homo_config.copy()
    homo_config.update(param_image=True)
    params = dict(src_filename=float_50cm_src_file, ref_filename=float_100cm_ref_file, homo_path=tmp_path,
                  method=Method.gain_blk_offset, kernel_shape=(5, 5), homo_config=homo_config, overwrite=overwrite)

    raster_fuse = RasterFuse(**params)
    raster_fuse.homo_filename.touch()
    if not overwrite:
        with pytest.raises(FileExistsError):
            _ = RasterFuse(**params)
    else:
        _ = RasterFuse(**params)

    os.remove(raster_fuse.homo_filename)
    raster_fuse.param_filename.touch()
    if not overwrite:
        with pytest.raises(FileExistsError):
            _ = RasterFuse(**params)
    else:
        _ = RasterFuse(**params)

@pytest.mark.parametrize('src_file, ref_file, method, kernel_shape, max_block_mem', [
    ('float_45cm_src_file', 'float_100cm_ref_file', Method.gain, (1, 1), 2.e-4),
    ('float_45cm_src_file', 'float_100cm_ref_file', Method.gain_blk_offset, (1, 1), 1.e-3),
    ('float_45cm_src_file', 'float_100cm_ref_file', Method.gain_offset, (5, 5), 1.e-3),
    ('float_100cm_src_file', 'float_45cm_ref_file', Method.gain, (1, 1), 2.e-4),
    ('float_100cm_src_file', 'float_45cm_ref_file', Method.gain_blk_offset, (1, 1), 1.e-3),
    ('float_100cm_src_file', 'float_45cm_ref_file', Method.gain_offset, (5, 5), 1.e-3),
])
def test_homo_file(src_file, ref_file, method, kernel_shape, max_block_mem, tmp_path, request):
    """"""
    src_file = request.getfixturevalue(src_file)
    ref_file = request.getfixturevalue(ref_file)
    homo_config = RasterFuse.default_homo_config
    homo_config.update(max_block_mem=max_block_mem)
    raster_fuse = RasterFuse(src_file, ref_file, tmp_path, method, kernel_shape, homo_config=homo_config)
    with raster_fuse:
        raster_fuse.process()
    assert(raster_fuse.homo_filename.exists())
    with rio.open(src_file, 'r') as src_ds, rio.open(raster_fuse.homo_filename, 'r') as out_ds:
        src_array = src_ds.read(indexes=1)
        src_mask = src_ds.dataset_mask().astype('bool', copy=False)
        out_array = src_ds.read(indexes=1)
        out_mask = out_ds.dataset_mask().astype('bool', copy=False)
        assert(out_mask == src_mask).all()
        assert(out_array[out_mask] == pytest.approx(src_array[src_mask], abs=1.e-3))

@pytest.mark.parametrize('out_profile', [
    dict(driver='GTiff', dtype='float32', nodata=float('nan'),
         creation_options=dict(tiled=True, blockxsize=512, blockysize=512, compress='deflate', interleave='band',
                               photometric=None)),
    dict(driver='GTiff', dtype='uint8', nodata=0,
         creation_options=dict(tiled=True, blockxsize=64, blockysize=64, compress='jpeg', interleave='pixel',
                               photometric='ycbcr')),
    dict(driver='PNG', dtype='uint16', nodata=0,
         creation_options=dict()),
])
def test_out_profile(float_100cm_rgb_file, tmp_path, out_profile):
    """"""
    raster_fuse = RasterFuse(float_100cm_rgb_file, float_100cm_rgb_file, tmp_path, Method.gain_blk_offset, (3,3),
                             out_profile=out_profile)
    with raster_fuse:
        raster_fuse.process()
    assert(raster_fuse.homo_filename.exists())
    out_profile.update(**out_profile['creation_options'])
    out_profile.pop('creation_options')
    with rio.open(float_100cm_rgb_file, 'r') as src_ds, rio.open(raster_fuse.homo_filename, 'r') as out_ds:
        # test output image has been set with out_profile properties
        for k, v in out_profile.items():
            assert ((v is None and k not in out_ds.profile) or (out_ds.profile[k] == v) or
                    (str(out_ds.profile[k]) == str(v)))

        if src_ds.profile['driver'].lower() == out_profile['driver'].lower():
            # source image keys including driver specific creation options, not present in out_profile
            src_keys = set(src_ds.profile.keys()).difference(out_profile.keys())
        else:
            # source image keys excluding driver specific creation options, not present in out_profile
            src_keys = {'width', 'height', 'count', 'dtype', 'crs', 'transform'}.difference(out_profile.keys())
        # test output image has been set with src image properties not in out_profile
        for k in src_keys:
            v = src_ds.profile[k]
            assert ((v is None and k not in out_ds.profile) or (out_ds.profile[k] == v) or
                    (str(out_ds.profile[k]) == str(v)))


# TO DO:
# - Test config:
#   - param_image, max_block_mem
#   - mask_partial?  has been tested via KernelModel, perhaps test that KernelModel has been set correctly
#   - out_profile: e.g. non-geotiff

class TestFuse(TestBase):
    """Integrations tests for fuse API and CLI."""

    def _test_homo_mask(self, src_filename: pathlib.Path, homo_filename: pathlib.Path, mask_partial: bool = False):
        """Validate the mask of a homogenised file."""
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            with rio.open(homo_filename, 'r') as homo_im, rio.open(src_filename, 'r') as src_im:
                # read masks
                # note that there is a bug / memory issue with cv2.connectedComponents, or perhaps datasset_mask(),
                # which the copy() below works around
                src_mask = src_im.dataset_mask().copy()
                homo_mask = homo_im.dataset_mask().copy()

                # check homogenised mask has similar area to source mask
                if mask_partial:
                    self.assertTrue(np.abs(src_mask.mean() - homo_mask.mean()) / 255 < .2,
                                    'Source and homogenised images have similar valid areas')
                else:
                    # strangely, compression adds some artifacts in the homo mask, so test for near equality rather
                    #  than exact equality
                    self.assertTrue(np.abs(src_mask.mean() - homo_mask.mean()) / 255 < .001,
                                    'Source and homogenised images have same valid area')

                # check for gaps due to overlapped block processing
                n_src_labels, src_labels = cv2.connectedComponents(src_mask, None, 4, cv2.CV_16U)
                n_homo_labels, homo_labels = cv2.connectedComponents(homo_mask, None, 4, cv2.CV_16U)
                self.assertTrue(n_src_labels == n_homo_labels,
                                'Number of source and homgenised valid/nodata areas match')

    def _test_homo_profile(self, src_filename: pathlib.Path, homo_filename: pathlib.Path):
        """Validate the homogenised properties."""
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            with rio.open(homo_filename, 'r') as homo_im, rio.open(src_filename, 'r') as src_im:
                # validate the homogenised image configuration
                out_profile = utils.combine_profiles(src_im.profile, self._out_profile)
                for key, val in out_profile.items():
                    match = (homo_im.profile[key] == val) | (str(homo_im.profile[key]) == str(val))
                    self.assertTrue(match, f'{key} set to {val}')
                self.assertTrue(src_im.crs.to_proj4() == homo_im.crs.to_proj4(), 'Source and homogenised crs match')
                self.assertTrue(src_im.count == homo_im.count, 'Source and homogenised band counts match')
                self.assertTrue(utils.covers_bounds(src_im, homo_im), 'Source bounds cover homogenised bounds')

    def _test_homo_correlation(self, src_filename: pathlib.Path, homo_filename: pathlib.Path,
                               ref_filename: pathlib.Path):
        """Test correlation between image and reference, before and after homogenisation."""
        # create dataframe of per-band r2 values for src-ref and homo-ref image pairs
        im_ref_r2 = []
        for im_i, im_filename in enumerate([src_filename, homo_filename]):
            cmp = RasterCompare(im_filename, ref_filename)
            cmp_dict = cmp.compare()
            im_ref_r2.append({band_key: band_val['r2'] for band_key, band_val in cmp_dict.items()})
        im_ref_r2_df = pd.DataFrame(im_ref_r2, index=['Source r2', 'Homo r2'])
        tqdm.write(im_ref_r2_df.to_string())

        # check the homogenised correlation is sane
        self.assertTrue(np.all(im_ref_r2_df.iloc[1, :] > 0.6), 'Homogenised r2 > 0.6')
        # check the homogenised correlation improves on the source correlation
        self.assertTrue(np.all(im_ref_r2_df.iloc[1, :] > im_ref_r2_df.iloc[0, :]),
                        'Homogenised vs reference r2 improvement')

    def _test_block_pair(self, block: Window, prev_block: Window, overlap: Tuple[int, int] = (0, 0), in_block=True):
        """Helper function for _test_block_pairs, to validate a window against the previous one in the sequence."""
        cmp = np.less_equal if in_block else np.equal

        if block.row_off == prev_block.row_off:
            self.assertTrue(cmp(block.col_off, prev_block.col_off + prev_block.width - 2 * overlap[1]),
                            f'columns overlap')
        else:
            self.assertTrue(cmp(block.row_off, prev_block.row_off + prev_block.height - 2 * overlap[0]),
                            f'rows overlap')

    def _test_block_pairs(self, raster_pair: RasterPairReader):
        """Validate block pairs from RasterPairReader."""
        ovl_blocks = list(raster_pair.block_pairs())

        # find the overlap in source and reference pixels
        proc_overlap = np.array(raster_pair.overlap)
        res_ratio = np.divide(raster_pair.ref_im.res, raster_pair.src_im.res)
        if raster_pair._proc_crs == ProcCrs.ref:
            other_overlap = proc_overlap * res_ratio
        else:
            other_overlap = proc_overlap / res_ratio
        other_overlap = np.round(other_overlap).astype('int')
        src_overlap = other_overlap if raster_pair._proc_crs == ProcCrs.ref else proc_overlap
        ref_overlap = proc_overlap if raster_pair._proc_crs == ProcCrs.ref else other_overlap

        # validate the BlockPairs are contiguous and overlapping
        prev_ovl_block = ovl_blocks[0]
        for ovl_block in ovl_blocks[1:]:
            if ovl_block.band_i == prev_ovl_block.band_i:
                self._test_block_pair(ovl_block.src_in_block, prev_ovl_block.src_in_block, src_overlap)
                self._test_block_pair(ovl_block.src_out_block, prev_ovl_block.src_out_block, in_block=False)
                self._test_block_pair(ovl_block.ref_in_block, prev_ovl_block.ref_in_block, ref_overlap)
                self._test_block_pair(ovl_block.ref_out_block, prev_ovl_block.ref_out_block, in_block=False)
            else:
                self.assertTrue(ovl_block.band_i == prev_ovl_block.band_i + 1, f'bands are contiguous')
            prev_ovl_block = ovl_block

    def _test_api(self, src_filename: pathlib.Path, ref_filename: pathlib.Path, test_filename: pathlib.Path, **kwargs):
        """Helper function to call the fuse API and validate results."""
        # generate and validate block pairs
        overlap = utils.overlap_for_kernel(kwargs['kernel_shape'])
        with self.subTest(ref_filename=ref_filename.name, src_filename=src_filename.name, overlap=overlap):
            with RasterPairReader(src_filename, ref_filename, proc_crs=kwargs['proc_crs'], overlap=overlap,
                                  max_block_mem=self._homo_config['max_block_mem']) as raster_pair:
                self._test_block_pairs(raster_pair)

        # homogenise and validate the results
        post_fix = utils.create_homo_postfix(driver=self._out_profile['driver'], **kwargs)
        homo_filename = self.homo_root.joinpath(src_filename.stem + post_fix)
        with RasterFuse(src_filename, ref_filename, homo_filename, **kwargs, homo_config=self._homo_config,
                        model_config=self._model_config, out_profile=self._out_profile) as raster_fuse:
            raster_fuse.process()
            raster_fuse.build_overviews()
        self.assertTrue(homo_filename.exists(), 'Homogenised file exists')
        self._test_homo_profile(src_filename, homo_filename)
        self._test_homo_mask(src_filename, homo_filename, mask_partial=self._model_config['mask_partial'])
        self._test_homo_correlation(src_filename, homo_filename, test_filename)

    def test_api_ref_space(self):
        """Test fuse API with proc-crs==ref."""
        param_list = [
            dict(method=Method.gain, kernel_shape=(3, 3), proc_crs=ProcCrs.ref),
            dict(method=Method.gain_blk_offset, kernel_shape=(5, 5), proc_crs=ProcCrs.ref),
            dict(method=Method.gain_offset, kernel_shape=(9, 9), proc_crs=ProcCrs.ref),
        ]
        for param_dict in param_list:
            with self.subTest(src_filename=self.aerial_filename.name, ref_filename=self.landsat_filename.name,
                              param_dict=param_dict):
                self._test_api(self.aerial_filename, self.landsat_filename, self.s2_filename, **param_dict)

    def test_api_src_space(self):
        """Test fuse API with proc-crs=src."""
        param_dict = dict(method=Method.gain, kernel_shape=(5, 5), proc_crs=ProcCrs.src)
        self._test_api(self.landsat_vrt, self.s2_filename, self.s2_filename, **param_dict)

    def test_cli(self):
        """Test fuse CLI."""
        param_list = [
            dict(method=Method.gain, kernel_shape=(1, 1), proc_crs=ProcCrs.ref),
            dict(method=Method.gain_blk_offset, kernel_shape=(3, 3), proc_crs=ProcCrs.ref),
            dict(method=Method.gain_offset, kernel_shape=(15, 15), proc_crs=ProcCrs.ref),
        ]

        for param_dict in param_list:
            cli_str = (f'-v fuse {" ".join(self.aerial_filenames)} {self.landsat_filename} '
                       f'-k {param_dict["kernel_shape"][0]} {param_dict["kernel_shape"][1]} -m {param_dict["method"]} '
                       f'-od {self.homo_root} -c {self._conf_filename} -pc {param_dict["proc_crs"]}')

            result = CliRunner().invoke(cli.cli, cli_str.split(), terminal_width=100, catch_exceptions=True)
            self.assertTrue(result.exit_code == 0, result.output)

            homo_post_fix = utils.create_homo_postfix(driver=self._out_profile['driver'], **param_dict)
            for src_filename in self.aerial_filenames:
                src_filename = pathlib.Path(src_filename)
                homo_filename = self.homo_root.joinpath(src_filename.stem + homo_post_fix)
                self.assertTrue(homo_filename.exists(), 'Homogenised file exists')
                with self.subTest(ref_filename=self.landsat_filename.name, homo_filename=homo_filename.name,
                                  param_dict=param_dict):
                    self._test_homo_profile(src_filename, homo_filename)
                    self._test_homo_mask(src_filename, homo_filename, mask_partial=self._model_config['mask_partial'])
                    self._test_homo_correlation(src_filename, homo_filename, self.s2_filename)

    def test_cli2(self):
        """Test fuse CLI."""
        param_list = [
            dict(method=Method.gain_blk_offset, kernel_shape=(5, 5), proc_crs=ProcCrs.ref),
        ]

        co_str = ''
        for co_key, co_val in self._out_profile['creation_options'].items():
            co_str += f'-co {co_key.upper()}={co_val} '

        for param_dict in param_list:
            cli_str = (f'-v fuse {" ".join(self.aerial_filenames)} {self.landsat_filename} '
                       f'-k {param_dict["kernel_shape"][0]} {param_dict["kernel_shape"][1]} -m {param_dict["method"]} '
                       f'-od {self.homo_root} {co_str} -pc {param_dict["proc_crs"]} -ovw')

            result = CliRunner().invoke(cli.cli, cli_str.split(), terminal_width=100, catch_exceptions=True)
            self.assertTrue(result.exit_code == 0, result.output)

            homo_post_fix = utils.create_homo_postfix(driver=self._out_profile['driver'], **param_dict)
            for src_filename in self.aerial_filenames:
                src_filename = pathlib.Path(src_filename)
                homo_filename = self.homo_root.joinpath(src_filename.stem + homo_post_fix)
                self.assertTrue(homo_filename.exists(), 'Homogenised file exists')
                with self.subTest(ref_filename=self.landsat_filename.name, homo_filename=homo_filename.name,
                                  param_dict=param_dict):
                    self._test_homo_profile(src_filename, homo_filename)
                    self._test_homo_mask(src_filename, homo_filename, mask_partial=self._model_config['mask_partial'])
                    self._test_homo_correlation(src_filename, homo_filename, self.s2_filename)

##
