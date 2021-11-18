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
import unittest
import warnings

import numpy as np
import rasterio as rio
import yaml
from homonim import homonim, root_path, cli
from shapely.geometry import box
from tqdm import tqdm


def _setup_test(cls):
    """ Test initialisation """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    test_out_dir = root_path.joinpath('data/outputs/test_example/homogenised')
    if not test_out_dir.exists():
        os.makedirs(test_out_dir)
    file_list = glob.glob(str(test_out_dir.joinpath('*')))
    for f in file_list:
        os.remove(f)

    conf_filename = root_path.joinpath('data/inputs/test_example/config.yaml')
    with open(conf_filename, 'r') as f:
        config = yaml.safe_load(f)
    cls._homo_config = config['homogenisation']
    cls._out_config = config['output']


class TestApi(unittest.TestCase):
    """ Class to test homonim API """
    _homo_config = None
    _out_config = None

    @classmethod
    def setUpClass(cls):
        """ """
        _setup_test(cls)

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
                    self.assertTrue(homo_im.profile[attr] == out_attr, f'{attr} set')
                # check homo_filename against source
                self.assertTrue(src_im.crs.to_proj4() == homo_im.crs.to_proj4(), 'Source and homogenised crs match')
                self.assertTrue(src_im.count == homo_im.count, 'Source and homogenised band counts match')
                src_box = box(*src_im.bounds)
                homo_box = box(*homo_im.bounds)
                self.assertTrue(src_box.covers(homo_box), 'Source bounds cover homogenised bounds')

    def _test_homo_against_ref(self, src_filename, homo_filename, ref_filename):
        """Test homogenised vs source image R2, against reference image"""
        him = homonim.HomonimRefSpace(src_filename, ref_filename)
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            im_ref_r2 = None
            for im_i, im_filename in enumerate([src_filename, homo_filename]):
                with rio.open(im_filename, 'r') as im:
                    if im_ref_r2 is None:
                        im_ref_r2 = np.zeros((2, im.count))
                    for band_i in range(im.count):
                        im_array = im.read(band_i + 1)
                        im_ds_array = him._project_src_to_ref(im_array, src_nodata=im.nodata)
                        mask = im_ds_array != homonim.hom_nodata
                        im_ref_cc = np.corrcoef(im_ds_array[mask], him.ref_array[band_i, mask])
                        im_ref_r2[im_i, band_i] = im_ref_cc[0, 1] ** 2

            self.assertTrue(np.all(im_ref_r2[1, :] > 0.7), 'Homogenised R2 > 0.6')
            self.assertTrue(np.all(im_ref_r2[1, :] > im_ref_r2[0, :]), 'Homogenised vs reference R2 improvement')
            tqdm.write(f'Pre-homogensied R2 : {im_ref_r2[0, :]}')
            tqdm.write(f'Post-homogenised R2: {im_ref_r2[1, :]}')

    def test_homogenise(self):
        """ """
        src_filename = root_path.joinpath('data/inputs/test_example/source/3324c_2015_1004_05_0182_RGB.tif')
        ref_filename = root_path.joinpath(
            'data/inputs/test_example/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.tif')
        ref2_filename = root_path.joinpath(
            'data/inputs/test_example/reference/COPERNICUS-S2-20151003T075826_20151003T082014_T35HKC_B432_Byte.tif')
        homo_root = root_path.joinpath('data/outputs/test_example/homogenised')

        param_list = [
            dict(method='gain_only', win_size=(3, 3), normalise=False),
            dict(method='gain_only', win_size=(5, 5), normalise=True),
            dict(method='gain_offset', win_size=(15, 15), normalise=False),
        ]

        for param_dict in param_list:
            post_fix = cli._create_homo_postfix(space='ref-space', **param_dict)
            homo_filename = homo_root.joinpath(src_filename.stem + post_fix)
            him = homonim.HomonimRefSpace(src_filename, ref_filename, homo_config=self._homo_config,
                                          out_config=self._out_config)
            him.homogenise(homo_filename, **param_dict)
            him.build_overviews(homo_filename)
            self.assertTrue(homo_filename.exists(), 'Homogenised file exists')
            with self.subTest('Homogenised vs Source', src_filename=src_filename, homo_filename=homo_filename):
                self._test_homo_against_src(src_filename, homo_filename)
            with self.subTest('Homogenised vs Reference', src_filename=src_filename, homo_filename=homo_filename,
                              ref_filename=ref2_filename):
                self._test_homo_against_ref(src_filename, homo_filename, ref2_filename)

##
