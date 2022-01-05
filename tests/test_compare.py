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
import json
import os
import pathlib
import unittest

import pandas as pd
import rasterio as rio
import yaml
from click.testing import CliRunner

from homonim import root_path, cli
from homonim.compare import ImCompare


class TestHomonim(unittest.TestCase):
    """ Class to test homonim API """

    def setUp(self):
        """Delete old test outputs and load config"""
        test_out_dir = root_path.joinpath('data/outputs/test_example')
        file_list = glob.glob(str(test_out_dir.joinpath('*.json')))
        for f in file_list:
            os.remove(f)

        self._conf_filename = root_path.joinpath('data/inputs/test_example/config.yaml')
        with open(self._conf_filename, 'r') as f:
            config = yaml.safe_load(f)
        self._config = cli._update_existing_keys(ImCompare.default_config, **config)

    def _test_compare_dict(self, src_filename, band_dict):
        band_df = pd.DataFrame.from_dict(band_dict, orient='index')
        self.assertTrue('Mean' in band_df.index, 'Comparison contains row of means')
        self.assertTrue('r2' in band_df, 'Comparison contains r2 column')
        with rio.open(src_filename, 'r') as src_im:
            self.assertEqual(band_df.shape[0], src_im.count + 1, 'Comparison contains correct number of bands')
        self.assertTrue(all(band_df.loc['Mean'] == band_df.iloc[-1]), 'Means are correct')
        self.assertTrue(all(band_df['r2'] >= 0) and all(band_df['r2'] <= 1), 'r2 in range')

    def _test_compare_api(self, src_filename, ref_filename, proc_crs='auto'):
        cmp = ImCompare(src_filename, ref_filename, proc_crs=proc_crs, multithread=self._config['multithread'])
        band_dict = cmp.compare()
        self._test_compare_dict(src_filename, band_dict)

    def test_compare_api_ref_space(self):
        """Test compare API with model_crs=ref"""
        src_filename = root_path.joinpath('data/inputs/test_example/source/3324c_2015_1004_05_0182_RGB.tif')
        ref_filename = root_path.joinpath(
            'data/inputs/test_example/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.tif')
        self._test_compare_api(src_filename, ref_filename, proc_crs='ref')

    def test_compare_api_src_space(self):
        """Test compare API with model_crs=ref"""
        src_filename = root_path.joinpath(
            'data/inputs/test_example/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.vrt')
        ref_filename = root_path.joinpath(
            'data/inputs/test_example/reference/COPERNICUS-S2-20151003T075826_20151003T082014_T35HKC_B432_Byte.tif')
        self._test_compare_api(src_filename, ref_filename, proc_crs='src')

    def test_cli(self):
        """Test compare CLI"""
        src_wildcard = root_path.joinpath('data/inputs/test_example/source/3324c_2015_*_RGB.tif')
        ref_filename = root_path.joinpath(
            'data/inputs/test_example/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.tif')
        cmp_filename = root_path.joinpath('data/outputs/test_example/comparison.json')

        cli_str = (f'compare {src_wildcard} {ref_filename} -pc ref --output {cmp_filename}')
        result = CliRunner().invoke(cli.cli, cli_str.split(), terminal_width=100, catch_exceptions=True)
        self.assertTrue(result.exit_code == 0, result.output)
        self.assertTrue(cmp_filename.exists(), 'Comparison results file exists')
        with open(cmp_filename) as f:
            cmp_dict = json.load(f)

        src_file_list = glob.glob(str(src_wildcard))
        for src_filename in src_file_list:
            src_filename = pathlib.Path(src_filename)
            self.assertTrue(src_filename.stem in cmp_dict, 'Comparison results file contain src file key')
            band_dict = cmp_dict[src_filename.stem]
            self._test_compare_dict(src_filename, band_dict)
