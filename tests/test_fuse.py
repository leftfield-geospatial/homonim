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
import glob
import os
import pathlib
import unittest
import warnings

import cv2
import numpy as np
import rasterio as rio
import yaml
from click.testing import CliRunner
from shapely.geometry import box
from tqdm import tqdm

from homonim import root_path, cli
from homonim.compare import RasterCompare
from homonim.fuse import RasterFuse
from homonim.raster_pair import RasterPairReader
from homonim.enums import Method, ProcCrs


class TestFuse(unittest.TestCase):
    """ Test fuse API and CLI """

    def setUp(self):
        """Delete old test outputs and load config"""
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        test_out_dir = root_path.joinpath('data/outputs/test_example/homogenised')
        if not test_out_dir.exists():
            os.makedirs(test_out_dir)
        file_list = glob.glob(str(test_out_dir.joinpath('*')))
        for f in file_list:
            os.remove(f)

        self._conf_filename = root_path.joinpath('data/inputs/test_example/config.yaml')
        with open(self._conf_filename, 'r') as f:
            config = yaml.safe_load(f)
        self._homo_config = cli._update_existing_keys(RasterFuse.default_homo_config, **config)
        self._out_profile = cli._update_existing_keys(RasterFuse.default_out_profile, **config)
        self._model_config = cli._update_existing_keys(RasterFuse.default_model_config, **config)

    def _test_homo_against_src(self, src_filename, homo_filename):
        """Test homogenised against source image"""
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            with rio.open(homo_filename, 'r') as homo_im, rio.open(src_filename, 'r') as src_im:
                # check homo_filename configured correctly
                def flatten_profile(in_profile, out_profile={}):
                    for k, v in in_profile.items():
                        if isinstance(v, dict):
                            out_profile = flatten_profile(v, out_profile=out_profile)
                        else:
                            out_profile[k] = v
                    return out_profile
                out_profile = flatten_profile(self._out_profile.copy())
                for attr in out_profile.keys():
                    if (out_profile[attr] is not None):
                        out_attr = out_profile[attr]
                    elif (attr in src_im.profile) and (src_im.profile['driver'] == out_profile['driver']):
                        out_attr = src_im.profile[attr]
                    else:
                        continue
                    is_set = (homo_im.profile[attr] == out_attr) | (str(homo_im.profile[attr]) == str(out_attr))
                    self.assertTrue(is_set, f'{attr} set')
                # check homo_filename against source
                self.assertTrue(src_im.crs.to_proj4() == homo_im.crs.to_proj4(), 'Source and homogenised crs match')
                self.assertTrue(src_im.count == homo_im.count, 'Source and homogenised band counts match')
                src_box = box(*src_im.bounds)
                homo_box = box(*homo_im.bounds)
                self.assertTrue(src_box.covers(homo_box), 'Source bounds cover homogenised bounds')
                for bi in range(src_im.count):
                    src_mask = src_im.read_masks(bi + 1)
                    homo_mask = homo_im.read_masks(bi + 1)
                    if self._model_config['mask_partial']:
                        self.assertTrue(np.abs(src_mask.mean() - homo_mask.mean()) / 255 < .2,
                                        'Source and homogenised images have similar valid areas')
                    else:
                        # strangely compression adds some artifacts in the homo mask, so test for near equality rather
                        #  than exact equality
                        self.assertTrue(np.abs(src_mask.mean() - homo_mask.mean()) / 255 < .001,
                                        'Source and homogenised images have same valid area')

                    for fn in [lambda x: x, lambda x: np.bitwise_not(x)]:
                        n_src_labels, src_labels = cv2.connectedComponents(fn(src_mask), None, 4, cv2.CV_16U)
                        n_homo_labels, homo_labels = cv2.connectedComponents(fn(homo_mask), None, 4, cv2.CV_16U)
                        # there can be small areas that become separated from main mask with mask_partial, so make it
                        # a near, rather than exact match.
                        self.assertTrue(np.abs(n_src_labels - n_homo_labels) <= 2,
                                        'Number of source and homgenised valid/nodata areas match')

    def _test_homo_against_ref(self, src_filename, homo_filename, ref_filename):
        """Test R2 against reference before and after homogenisation"""
        im_ref_r2 = []
        for im_i, im_filename in enumerate([src_filename, homo_filename]):
            cmp = RasterCompare(im_filename, ref_filename)
            cmp_dict = cmp.compare()
            im_ref_r2.append([band_dict['r2'] for band_dict in cmp_dict.values()])

        im_ref_r2 = np.array(im_ref_r2)
        tqdm.write(f'Pre-homogensied R2 : {im_ref_r2[0, :]}')
        tqdm.write(f'Post-homogenised R2: {im_ref_r2[1, :]}')
        self.assertTrue(np.all(im_ref_r2[1, :] > 0.6), 'Homogenised R2 > 0.6')
        self.assertTrue(np.all(im_ref_r2[1, :] > im_ref_r2[0, :]), 'Homogenised vs reference R2 improvement')

    def _test_ovl_blocks(self, raster_pair: RasterPairReader):
        """ Test overlap blocks for sanity """
        ovl_blocks = list(raster_pair.block_pairs())
        proc_overlap = np.array(raster_pair._overlap)
        res_ratio = np.divide(raster_pair.ref_im.res, raster_pair.src_im.res)
        other_overlap = proc_overlap * res_ratio if raster_pair._proc_crs == ProcCrs.ref else proc_overlap / res_ratio
        other_overlap = np.round(other_overlap).astype('int')
        ref_overlap = proc_overlap if raster_pair._proc_crs == ProcCrs.ref else other_overlap
        src_overlap = other_overlap if raster_pair._proc_crs == ProcCrs.ref else proc_overlap
        prev_ovl_block = ovl_blocks[0]
        for ovl_block in ovl_blocks[1:]:
            ovl_block = ovl_block
            if ovl_block.band_i == prev_ovl_block.band_i:
                for out_blk_fld in ('src_out_block', 'ref_out_block'):
                    curr_blk = ovl_block.__getattribute__(out_blk_fld)
                    prev_blk = prev_ovl_block.__getattribute__(out_blk_fld)
                    if curr_blk.row_off == prev_blk.row_off:
                        self.assertTrue(curr_blk.col_off == prev_blk.col_off + prev_blk.width,
                                        f'{out_blk_fld} col consecutive')
                    else:
                        self.assertTrue(curr_blk.row_off == prev_blk.row_off + prev_blk.height,
                                        f'{out_blk_fld} row consecutive')
                for in_blk_fld, overlap in zip(('src_in_block', 'ref_in_block'), (src_overlap, ref_overlap)):
                    curr_blk = ovl_block.__getattribute__(in_blk_fld)
                    prev_blk = prev_ovl_block.__getattribute__(in_blk_fld)
                    if curr_blk.row_off == prev_blk.row_off:
                        self.assertTrue(curr_blk.col_off == prev_blk.col_off + prev_blk.width - 2*overlap[1],
                                        f'{in_blk_fld} col overlap')
                    else:
                        self.assertTrue(curr_blk.row_off == prev_blk.row_off + prev_blk.height - 2*overlap[0],
                                        f'{in_blk_fld} row overlap')
            else:
                self.assertTrue(ovl_block.band_i == prev_ovl_block.band_i + 1, f'band consecutive')

            prev_ovl_block = ovl_block

    def _test_api(self, src_filename, ref_filename, test_filename, **kwargs):
        with self.subTest('Overlapped Blocks', src_filename=src_filename):
            overlap = np.floor(np.array(kwargs['kernel_shape'])/2).astype('int')
            with RasterPairReader(src_filename, ref_filename, proc_crs=kwargs['proc_crs'], overlap=overlap,
                                  max_block_mem=self._homo_config['max_block_mem']) as raster_pair:
                self._test_ovl_blocks(raster_pair)

        homo_root = root_path.joinpath('data/outputs/test_example/homogenised')

        post_fix = cli._create_homo_postfix(driver=self._out_profile['driver'], **kwargs)
        homo_filename = homo_root.joinpath(src_filename.stem + post_fix)
        him = RasterFuse(src_filename, ref_filename, **kwargs, homo_config=self._homo_config,
                         model_config=self._model_config, out_profile=self._out_profile)
        him.homogenise(homo_filename)
        him.build_overviews(homo_filename)
        self.assertTrue(homo_filename.exists(), 'Homogenised file exists')
        with self.subTest('Homogenised vs Source', src_filename=src_filename, homo_filename=homo_filename):
            self._test_homo_against_src(src_filename, homo_filename)
        with self.subTest('Homogenised vs Reference', src_filename=src_filename, homo_filename=homo_filename,
                          ref_filename=test_filename):
            self._test_homo_against_ref(src_filename, homo_filename, test_filename)

    def test_api_ref_space(self):
        """Test homogenisation API with model-crs=ref"""
        src_filename = root_path.joinpath('data/inputs/test_example/source/3324c_2015_1004_05_0182_RGB.tif')
        ref_filename = root_path.joinpath(
            'data/inputs/test_example/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.tif')
        ref2_filename = root_path.joinpath(
            'data/inputs/test_example/reference/COPERNICUS-S2-20151003T075826_20151003T082014_T35HKC_B432_Byte.tif')

        param_list = [
            dict(method=Method.gain, kernel_shape=(3, 3), proc_crs=ProcCrs.ref),
            dict(method=Method.gain_im_offset, kernel_shape=(5, 5), proc_crs=ProcCrs.ref),
            dict(method=Method.gain_offset, kernel_shape=(9, 9), proc_crs=ProcCrs.ref),
        ]
        for param_dict in param_list:
            self._test_api(src_filename, ref_filename, ref2_filename, **param_dict)
        # -m unittest test_fuse.TestFuse.test_api_ref_space

    def test_api_src_space(self):
        """Test homogenisation API with model-crs=src and src res > ref res"""
        src_filename = root_path.joinpath(
            'data/inputs/test_example/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.vrt')
        ref_filename = root_path.joinpath(
            'data/inputs/test_example/reference/COPERNICUS-S2-20151003T075826_20151003T082014_T35HKC_B432_Byte.tif')
        # src_filename = root_path.joinpath(
        #     'data/inputs/test_example/reference/COPERNICUS-S2-20151003T075826_20151003T082014_T35HKC_B432_Byte.vrt')
        # ref_filename = root_path.joinpath(
        #     'data/inputs/test_example/source/NGI_Baviaanskloof_3324c_2015_RGB.vrt')

        param_dict = dict(method=Method.gain, kernel_shape=(5, 5), proc_crs=ProcCrs.src)
        self._test_api(src_filename, ref_filename, ref_filename, **param_dict)

    def test_cli(self):
        """Test homogenisation CLI"""
        src_wildcard = root_path.joinpath('data/inputs/test_example/source/3324c_2015_*_RGB.tif')
        ref_filename = root_path.joinpath(
            'data/inputs/test_example/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.tif')
        ref2_filename = root_path.joinpath(
            'data/inputs/test_example/reference/COPERNICUS-S2-20151003T075826_20151003T082014_T35HKC_B432_Byte.tif')
        homo_root = root_path.joinpath('data/outputs/test_example/homogenised')

        param_list = [
            dict(method=Method.gain, kernel_shape=(1, 1), proc_crs=ProcCrs.ref),
            dict(method=Method.gain_im_offset, kernel_shape=(3, 3), proc_crs=ProcCrs.ref),
            dict(method=Method.gain_offset, kernel_shape=(9, 9), proc_crs=ProcCrs.ref),
        ]

        for param_dict in param_list:
            cli_str = (f'fuse {src_wildcard} {ref_filename} -k {param_dict["kernel_shape"][0]} '
                       f'{param_dict["kernel_shape"][1]} -m {param_dict["method"]} -od {homo_root} -c '
                       f'{self._conf_filename} -pc {param_dict["proc_crs"]}')

            result = CliRunner().invoke(cli.cli, cli_str.split(), terminal_width=100, catch_exceptions=True)
            self.assertTrue(result.exit_code == 0, result.output)

            src_file_list = glob.glob(str(src_wildcard))
            homo_post_fix = cli._create_homo_postfix(driver=self._out_profile['driver'], **param_dict)
            for src_filename in src_file_list:
                src_filename = pathlib.Path(src_filename)
                homo_filename = homo_root.joinpath(src_filename.stem + homo_post_fix)
                self.assertTrue(homo_filename.exists(), 'Homogenised file exists')
                with self.subTest('Homogenised vs Source', src_filename=src_filename, homo_filename=homo_filename):
                    self._test_homo_against_src(src_filename, homo_filename)
                with self.subTest('Homogenised vs Reference', src_filename=src_filename, homo_filename=homo_filename,
                                  ref_filename=ref2_filename):
                    self._test_homo_against_ref(src_filename, homo_filename, ref2_filename)
##
