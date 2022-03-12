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

import numpy as np
import pytest

from homonim import root_path
from homonim.stats import ParamStats
from homonim.errors import ImageFormatError


def test_stats():
    param_file = root_path.joinpath(
        'data/test_example/param/3324c_2015_1004_05_0182_RGB_HOMO_cREF_mGAIN-OFFSET_k15_15_PARAM.tif')
    stats = ParamStats(param_file)
    assert  (len(stats.metadata) > 0)
    param_stats = stats.stats()
    assert len(param_stats) == 9
    for band_name, band_stats in param_stats.items():
        assert ({'Mean', 'Std', 'Min', 'Max'} <= set(band_stats.keys()))
    for r2_stats in list(param_stats.values())[-3:]:
        assert ('Inpaint (%)' in r2_stats)
        assert (r2_stats['Inpaint (%)'] >= 0 and r2_stats['Inpaint (%)'] <= 100)
        r2_stats.pop('Inpaint (%)')
        r2_stat_vals = np.array(list(r2_stats.values()))
        assert ((r2_stat_vals >= 0) & (r2_stat_vals <= 1)).all()

def test_file_format_error(float_100cm_rgb_file):
    with pytest.raises(ImageFormatError):
        _ = ParamStats(float_100cm_rgb_file)