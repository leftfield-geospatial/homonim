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
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from shapely.geometry import box
from tqdm import tqdm

from homonim import homonim, root_path, cli
from homonim.raster_array import RasterArray, expand_window_to_grid


def _read_ref(src_filename, ref_filename):
    """
    Read the source region from the reference image in the source CRS.
    """
    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(src_filename, 'r') as src_im:
        with WarpedVRT(rio.open(ref_filename, 'r'), crs=src_im.crs, resampling=Resampling.bilinear) as ref_im:
            ref_win = expand_window_to_grid(ref_im.window(*src_im.bounds))
            ref_bands = range(1, src_im.count + 1)
            _ref_array = ref_im.read(ref_bands, window=ref_win).astype(homonim.hom_dtype)

            if (ref_im.nodata is not None) and (ref_im.nodata != homonim.hom_nodata):
                _ref_array[_ref_array == ref_im.nodata] = homonim.hom_nodata
            ref_array = RasterArray.from_profile(_ref_array, ref_im.profile, ref_win)

    return ref_array


class TestHomonim(unittest.TestCase):
    """ Class to test homonim API """

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
        self._homo_config = config['homogenisation']
        self._out_config = config['output']

    def _test_homo_against_src(self, src_filename, homo_filename):
        """Test homogenised against source image"""
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            with rio.open(homo_filename, 'r') as homo_im, rio.open(src_filename, 'r') as src_im:
                # check homo_filename configured correctly
                for attr in ['nodata', 'dtype', 'compress', 'interleave', 'blockxsize', 'blockysize']:
                    if self._out_config[attr] is None:
                        out_attr = src_im.profile[attr]
                    else:
                        out_attr = self._out_config[attr]
                    self.assertTrue(str(homo_im.profile[attr]) == str(out_attr), f'{attr} set')
                # check homo_filename against source
                self.assertTrue(src_im.crs.to_proj4() == homo_im.crs.to_proj4(), 'Source and homogenised crs match')
                self.assertTrue(src_im.count == homo_im.count, 'Source and homogenised band counts match')
                src_box = box(*src_im.bounds)
                homo_box = box(*homo_im.bounds)
                self.assertTrue(src_box.covers(homo_box), 'Source bounds cover homogenised bounds')
                for bi in range(src_im.count):
                    src_mask = src_im.read_masks(bi + 1)
                    homo_mask = homo_im.read_masks(bi + 1)
                    self.assertTrue(np.abs(src_mask.mean() - homo_mask.mean()) / 255 < .2,
                                    'Source and homgenised have similar valid areas')

                    for fn in [lambda x: x, lambda x: np.bitwise_not(x)]:
                        n_src_labels, src_labels = cv2.connectedComponents(fn(src_mask), None, 4, cv2.CV_16U)
                        n_homo_labels, homo_labels = cv2.connectedComponents(fn(homo_mask), None, 4, cv2.CV_16U)
                        self.assertTrue(n_src_labels == n_homo_labels,
                                        'Number of source and homgenised valid/nodata areas match')

    def _test_homo_against_ref(self, src_filename, homo_filename, ref_filename):
        """Test R2 against reference before and after homogenisation"""
        him = homonim.HomonimRefSpace(src_filename, ref_filename)
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            im_ref_r2 = None
            ref_array = _read_ref(src_filename, ref_filename)
            for im_i, im_filename in enumerate([src_filename, homo_filename]):
                with rio.open(im_filename, 'r') as im:
                    if im_ref_r2 is None:
                        im_ref_r2 = np.zeros((2, im.count))
                    for band_i in range(im.count):
                        _im_array = im.read(band_i + 1)
                        im_array = RasterArray.from_profile(_im_array, im.profile)
                        im_ds_array = im_array.reproject(transform=ref_array.transform, shape=ref_array.shape[-2:],
                                                         resampling=him._homo_config['src2ref_interp'])
                        mask = im_ds_array.mask
                        im_ref_cc = np.corrcoef(im_ds_array.array[mask], ref_array.array[band_i, mask])
                        im_ref_r2[im_i, band_i] = im_ref_cc[0, 1] ** 2

            tqdm.write(f'Pre-homogensied R2 : {im_ref_r2[0, :]}')
            tqdm.write(f'Post-homogenised R2: {im_ref_r2[1, :]}')
            self.assertTrue(np.all(im_ref_r2[1, :] > 0.6), 'Homogenised R2 > 0.6')
            self.assertTrue(np.all(im_ref_r2[1, :] > im_ref_r2[0, :]), 'Homogenised vs reference R2 improvement')

    def _test_ovl_blocks(self, ovl_blocks):
        """ Test overlap blocks for sanity """
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
                for in_blk_fld in ('src_in_block', 'ref_in_block'):
                    curr_blk = ovl_block.__getattribute__(in_blk_fld)
                    prev_blk = prev_ovl_block.__getattribute__(in_blk_fld)
                    if curr_blk.row_off == prev_blk.row_off:
                        self.assertTrue(curr_blk.col_off < prev_blk.col_off + prev_blk.width,
                                        f'{in_blk_fld} col overlap')
                    else:
                        self.assertTrue(curr_blk.row_off < prev_blk.row_off + prev_blk.height,
                                        f'{in_blk_fld} row overlap')
            else:
                self.assertTrue(ovl_block.band_i == prev_ovl_block.band_i + 1, f'band consecutive')

            prev_ovl_block = ovl_block

    def test_api(self):
        """Test homogenisation API"""
        src_filename = root_path.joinpath('data/inputs/test_example/source/3324c_2015_1004_05_0182_RGB.tif')
        ref_filename = root_path.joinpath(
            'data/inputs/test_example/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.tif')
        ref2_filename = root_path.joinpath(
            'data/inputs/test_example/reference/COPERNICUS-S2-20151003T075826_20151003T082014_T35HKC_B432_Byte.tif')
        homo_root = root_path.joinpath('data/outputs/test_example/homogenised')

        param_list = [
            dict(method='gain', kernel_shape=(3, 3)),
            dict(method='gain_im_offset', kernel_shape=(5, 5)),
            dict(method='gain_offset', kernel_shape=(9, 9)),
        ]

        for param_dict in param_list:
            post_fix = cli._create_homo_postfix(space='ref-space', **param_dict)
            homo_filename = homo_root.joinpath(src_filename.stem + post_fix)
            him = homonim.HomonimRefSpace(src_filename, ref_filename, homo_config=self._homo_config,
                                          out_config=self._out_config)
            with self.subTest('Overlapped Blocks', src_filename=src_filename):
                self._test_ovl_blocks(him._create_ovl_blocks(param_dict['kernel_shape']))
            him.homogenise(homo_filename, **param_dict)
            him.set_homo_metadata(homo_filename)
            him.build_overviews(homo_filename)
            self.assertTrue(homo_filename.exists(), 'Homogenised file exists')
            with self.subTest('Homogenised vs Source', src_filename=src_filename, homo_filename=homo_filename):
                self._test_homo_against_src(src_filename, homo_filename)
            with self.subTest('Homogenised vs Reference', src_filename=src_filename, homo_filename=homo_filename,
                              ref_filename=ref2_filename):
                self._test_homo_against_ref(src_filename, homo_filename, ref2_filename)

    def test_cli(self):
        """Test homogenisation CLI"""
        src_wildcard = root_path.joinpath('data/inputs/test_example/source/3324c_2015_*_RGB.tif')
        ref_filename = root_path.joinpath(
            'data/inputs/test_example/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.tif')
        ref2_filename = root_path.joinpath(
            'data/inputs/test_example/reference/COPERNICUS-S2-20151003T075826_20151003T082014_T35HKC_B432_Byte.tif')
        homo_root = root_path.joinpath('data/outputs/test_example/homogenised')

        param_list = [
            dict(method='gain', kernel_shape=(3, 3)),
            dict(method='gain_im_offset', kernel_shape=(5, 5)),
            dict(method='gain_offset', kernel_shape=(9, 9)),
        ]

        for param_dict in param_list:
            kernel_shape_str = [str(param_dict["kernel_shape"][0]), str(param_dict["kernel_shape"][1])]
            cli_params = ['-s', str(src_wildcard), '-r', str(ref_filename), '--ref-space', '-k', *kernel_shape_str,
                          '-m', param_dict['method'], '-od', str(homo_root), '-c', str(self._conf_filename)]
            result = CliRunner().invoke(cli.cli, cli_params, terminal_width=100)
            self.assertTrue(result.exit_code == 0, result.exception)

            src_file_list = glob.glob(str(src_wildcard))
            homo_post_fix = cli._create_homo_postfix(space='ref-space', **param_dict)
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
