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
import pathlib
from homonim import homonim

# src_filename = pathlib.Path(r"V:\Data\NGI\UnRectified\3324C_2015_1004\RGBN\simple_ortho_eg\3324c_2015_1004_05_0182_RGBN_CMP_ORTHO.tif")#2015-09-06
src_filename = pathlib.Path(r"V:\Data\HomonimEgs\Sequoia_2017_02_28_JonasKop\Source\odm_orthophoto - with cam radiom and with global mosaic adj 4Band.tif")
# src_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3322A_2010_HotSpotSeamLineEg\Source\3322a_320_13_0478_rgbn_CMP.tif")
# src_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3322A_2010_HotSpotSeamLineEg\Source\3322a_320_02_0041_rgbn_CMP.tif")
# src_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3322A_2010_HotSpotSeamLineEg\Source\3322a_320_13_0475_rgbn_CMP.tif")
# src_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3322A_2010_HotSpotSeamLineEg\Source\3322a_320_03_0103_rgbn_CMP.tif")


ref_filename = pathlib.Path(r"V:/Data/HomonimEgs/Sequoia_2017_02_28_JonasKop/Reference/S2_UctJonasPeak_UTM34S_SeqBands.tif")
# ref_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3322A_2010_HotSpotSeamLineEg\Reference\MCD43A4.A2010025.h19v12.005.2010043064233.Lo23.RGBN.tif")
# ref_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3322A_2010_HotSpotSeamLineEg\Reference\LANDSAT-LE07-C02-T1_L2-2010_02_03-2010_02_19-Q_MOSAIC_COMP_B3214.tif")

hom = homonim.HomonimRefSpace(src_filename, ref_filename, win_size=[15, 15])
hom.homogenise(src_filename.parent.joinpath(src_filename.stem + '_HOMO_REF_L7_GainAndOffset' + src_filename.suffix))


##
import numpy as np
import cv2
from datetime import datetime

srange = np.arange(100, 2000, 200)
durations = np.zeros(srange.shape)
ksize = (15,15)
for si, s in enumerate(srange):
    x = np.random.randn(s, s).astype('float32')
    start = datetime.now()
    x_sum = cv2.boxFilter(x, -1, ksize, normalize=False, borderType=cv2.BORDER_CONSTANT)
    durations[si] = (datetime.now() - start).total_seconds()
    print(s)

from matplotlib import pyplot

pyplot.figure()
pyplot.plot(srange**2, durations, 'o-')
pyplot.xlabel('Array size')
pyplot.ylabel('Time (secs)')