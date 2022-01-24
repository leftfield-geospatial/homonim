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
import pathlib

import pandas as pd
import rasterio as rio
from click.testing import CliRunner

from homonim import root_path, cli
from homonim.compare import RasterCompare
from homonim.enums import ProcCrs
from tests.common import TestBase


class TestCompare(TestBase):
    """Integrations tests for compare API and CLI."""

    def _test_compare_dict(self, src_filename: pathlib.Path, band_dict: dict):
        """Validate the band stats dictionary returned by RasterCompare.compare()."""
        band_df = pd.DataFrame.from_dict(band_dict, orient='index')
        self.assertTrue('Mean' in band_df.index, 'Comparison contains row of means')
        self.assertTrue('r2' in band_df, 'Comparison contains r2 column')
        with rio.open(src_filename, 'r') as src_im:
            self.assertEqual(band_df.shape[0], src_im.count + 1, 'Comparison contains correct number of bands')
        self.assertTrue(all(band_df.loc['Mean'] == band_df.iloc[:-1].mean()), 'Means are correct')
        self.assertTrue(all(band_df['r2'] >= 0) and all(band_df['r2'] <= 1), 'r2 in range')

    def _test_compare_api(self, src_filename: pathlib.Path, ref_filename: pathlib.Path,
                          proc_crs: ProcCrs = ProcCrs.auto):
        """Helper function to call the compare API and validate results."""
        cmp = RasterCompare(src_filename, ref_filename, proc_crs=proc_crs, threads=self._homo_config['threads'])
        band_dict = cmp.compare()
        self._test_compare_dict(src_filename, band_dict)

    def test_compare_api_ref_space(self):
        """Test compare API with proc-crs==ref."""
        self._test_compare_api(self.aerial_filename, self.landsat_filename, proc_crs=ProcCrs.ref)

    def test_compare_api_src_space(self):
        """Test compare API with proc-crs==src."""
        self._test_compare_api(self.landsat_vrt, self.s2_filename, proc_crs=ProcCrs.src)

    def test_cli(self):
        """Test compare CLI"""
        cmp_filename = root_path.joinpath('data/test_example/comparison.json')

        cli_str = f'compare {" ".join(self.aerial_filenames)} {self.landsat_filename} -pc ref --output {cmp_filename}'
        result = CliRunner().invoke(cli.cli, cli_str.split(), terminal_width=100, catch_exceptions=True)
        self.assertTrue(result.exit_code == 0, result.output)
        self.assertTrue(cmp_filename.exists(), 'Comparison results file exists')
        with open(cmp_filename) as f:
            cmp_dict = json.load(f)

        for src_filename in self.aerial_filenames:
            self.assertTrue(src_filename in cmp_dict, 'Comparison results file contain src file key')
            band_dict = cmp_dict[src_filename]
            self._test_compare_dict(pathlib.Path(src_filename), band_dict)
