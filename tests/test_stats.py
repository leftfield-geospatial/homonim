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
import json
import pathlib

import pandas as pd
import rasterio as rio
from click.testing import CliRunner

from homonim import root_path, cli
from homonim.compare import RasterCompare
from homonim.enums import ProcCrs
from tests.common import TestBase


class TestStats(TestBase):
    """Integrations tests for stats CLI."""

    def _test_stats_dict(self, param_filename: pathlib.Path, band_dict: dict):
        """Validate the band stats dictionary returned by RasterCompare.compare()."""
        band_df = pd.DataFrame.from_dict(band_dict, orient='index')
        self.assertTrue({'Mean', 'Std', 'Min', 'Max', 'Inpaint (%)'} <= set(band_df.columns),
                        'Stats contains required columns')
        self.assertTrue('Mean' in band_df, 'Stats contains mean column')
        with rio.open(param_filename, 'r') as param_im:
            self.assertEqual(band_df.shape[0], param_im.count, 'Stats contains correct number of bands')

        r2_df = band_df.iloc[-int(band_df.shape[0] / 3):][['Mean', 'Std', 'Min', 'Max']]
        self.assertTrue(all(r2_df >= 0) and all(r2_df <= 1), 'r2 in range')

    def test_cli(self):
        """Test compare CLI"""
        stats_filename = root_path.joinpath('data/test_example/stats.json')

        cli_str = f'stats {self.param_filename} --output {stats_filename}'
        result = CliRunner().invoke(cli.cli, cli_str.split(), terminal_width=100, catch_exceptions=True)
        self.assertTrue(result.exit_code == 0, result.output)
        self.assertTrue(stats_filename.exists(), 'Comparison results file exists')
        with open(stats_filename) as f:
            stats_dict = json.load(f)

        param_filename = str(self.param_filename)
        self.assertTrue(param_filename in stats_dict, 'Stats results file contain param file key')
        band_dict = stats_dict[param_filename]
        self._test_stats_dict(self.param_filename, band_dict)
