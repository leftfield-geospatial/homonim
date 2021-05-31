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
import numpy
import rasterio as rio
from homonim import homonim

# src_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3324C_2015_1004\RGBN\simple_ortho_eg\3324c_2015_1004_05_0182_RGBN_CMP_ORTHO.tif")#2015-09-06
src_filename = pathlib.Path(r"V:\Data\UCT\combined 0005 0006\odm_orthophoto\odm_orthophoto - with cam radiom and with global mosaic adj 4Band.tif")
ref_filename = pathlib.Path(r"V:\Data\UCT\combined 0005 0006\S2_UctJonasPeak_UTM34S_SeqBands.tif")

hom = homonim.HomonIm(src_filename, ref_filename)
hom.homogenise(src_filename.parent.joinpath(src_filename.stem + '_homonim' + src_filename.suffix))