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
from homonim import root_path, get_logger

# src_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3324C_2015_1004\RGBN\simple_ortho_eg\3324c_2015_1004_05_0182_RGBN_CMP_ORTHO.tif")#2015-09-06
src_filename = pathlib.Path(r"V:\Data\UCT\combined 0005 0006\odm_orthophoto\odm_orthophoto - with cam radiom and with global mosaic adj 4Band.tif")
# src_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3322A_2010_Subsection\Source\3322a_320_02_0041_rgbn_CMP.tif")
ref_filename = pathlib.Path(r"V:\Data\UCT\combined 0005 0006\S2_UctJonasPeak_UTM34S_SeqBands.tif")
# ref_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3322A_2010_Subsection\Reference\MCD43A4.A2010025.h19v12.005.2010043064233.Lo23.RGBN.tif")

hom = homonim.HomonimSrcSpace(src_filename, ref_filename, win_size=[3, 3])
hom.homogenise(src_filename.parent.joinpath(src_filename.stem + '_HOMO_SRC' + src_filename.suffix))


##

