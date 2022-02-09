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

import yaml

from homonim import root_path, cli
from homonim.fuse import RasterFuse


class TestBase(unittest.TestCase):
    """Base class for homonim integration tests."""
    @classmethod
    def setUpClass(cls) -> None:
        """Delete old test files and load common configuration."""
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        cls.source_root = root_path.joinpath('data/test_example/source/')
        cls.ref_root = root_path.joinpath('data/test_example/reference/')
        cls.homo_root = root_path.joinpath('data/test_example/homogenised/')
        cls.param_root = root_path.joinpath('data/test_example/param/')
        cls.aerial_filename = cls.source_root.joinpath('3324c_2015_1004_05_0182_RGB.tif')
        cls.aerial_filenames = [str(fn) for fn in cls.aerial_filename.parent.glob('3324c_2015_*_RGB.tif')]
        cls.landsat_filename = cls.ref_root.joinpath('LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.tif')
        cls.landsat_vrt = cls.ref_root.joinpath('LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.vrt')
        cls.s2_filename = cls.ref_root.joinpath('COPERNICUS-S2-20151003T075826_20151003T082014_T35HKC_B432_Byte.tif')
        cls.param_filename = cls.param_root.joinpath(
            '3324c_2015_1004_05_0182_RGB_HOMO_cREF_mGAIN-OFFSET_k15_15_PARAM.tif')

        # delete old output files
        if not cls.homo_root.exists():
            os.makedirs(cls.homo_root)
        file_list = glob.glob(str(cls.homo_root.joinpath('*')))
        for f in file_list:
            os.remove(f)

        # load config
        cls._conf_filename = root_path.joinpath('data/test_example/config.yaml')
        with open(cls._conf_filename, 'r') as f:
            config = yaml.safe_load(f)
        cls._homo_config = cli._update_existing_keys(RasterFuse.default_homo_config, **config)
        cls._out_profile = cli._update_existing_keys(RasterFuse.default_out_profile, **config)
        cls._model_config = cli._update_existing_keys(RasterFuse.default_model_config, **config)
        return cls
