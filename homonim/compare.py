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

import logging
import warnings
import cProfile
import concurrent.futures
import multiprocessing
import pathlib
import pstats
import threading
import tracemalloc
from collections import namedtuple
from itertools import product

import numpy as np
import rasterio as rio
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from homonim.kernel_model import KernelModel, RefSpaceModel, SrcSpaceModel
from homonim.raster_array import RasterArray, round_window_to_grid, expand_window_to_grid
from homonim.errors import (UnsupportedImageError, ImageContentError, BlockSizeError)
from homonim.inspect import _inspect_image, _inspect_image_pair

logger = logging.getLogger(__name__)

class ImCompare():
    def __init__(self, src_filename, ref_filename, proc_crs='auto'):
        """
        Class for comparing images

        Parameters
        ----------
        src_filename : str, pathlib.Path
            Source image filename.
        ref_filename: str, pathlib.Path
            Reference image filename.
        """
        self._src_filename = pathlib.Path(src_filename)
        self._ref_filename = pathlib.Path(ref_filename)

        self._ref_bands = None
        self._src_bands = None
        self._ref_warped_vrt_dict = None
        self._profile = False
        self._proc_crs = proc_crs
        self._image_init()


    def _image_init(self):
        """Check bounds, band count, and compression type of source and reference images"""
        self._src_bands, self._ref_bands = _inspect_image_pair(self._src_filename, self._ref_filename)

        with rio.open(self._src_filename, 'r') as src_im, rio.open(self._ref_filename, 'r') as _ref_im:
            if src_im.crs.to_proj4() != _ref_im.crs.to_proj4():  # re-project the reference image to source CRS
                logger.info(f'Re-projecting reference image to source CRS.')
            with WarpedVRT(_ref_im, crs=src_im.crs, resampling=Resampling.bilinear) as ref_im:
                src_pixel_smaller = np.prod(src_im.res) < np.prod(ref_im.res)
                if self._model_crs == 'auto':
                    self._model_crs = 'ref' if src_pixel_smaller else 'src'
                elif ((self._model_crs == 'src' and src_pixel_smaller) or
                      (self._model_crs == 'ref' and not src_pixel_smaller)):
                    logger.warning(f'model_crs == "{"ref" if src_pixel_smaller else "src"}" is recommended when '
                                  f'the source image pixel size is '
                                  f'{"smaller" if src_pixel_smaller else "larger"} than the reference.')

                ref_win = expand_window_to_grid(
                    ref_im.window(*src_im.bounds),
                    expand_pixels=np.ceil(np.divide(src_im.res, ref_im.res)).astype('int')
                )
                ref_transform = ref_im.window_transform(ref_win)
                self._ref_warped_vrt_dict = dict(crs=src_im.crs, transform=ref_transform, width=ref_win.width,
                                                 height=ref_win.height, resampling=Resampling.bilinear)
