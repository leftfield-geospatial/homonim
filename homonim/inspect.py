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
import pathlib

import numpy as np
import rasterio as rio
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from shapely.geometry import box

from homonim.errors import (UnsupportedImageError, ImageContentError)

logger = logging.getLogger(__name__)


def _inspect_image(im_filename):
    im_filename = pathlib.Path(im_filename)
    if not im_filename.exists():
        raise FileNotFoundError(f'{im_filename.name} does not exist')

    with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(im_filename, 'r') as im:
        try:
            tmp_array = im.read(1, window=im.block_window(1, 0, 0))
        except Exception as ex:
            if im.profile['compress'] == 'jpeg':  # assume it is a 12bit JPEG
                raise UnsupportedImageError(
                    f'Could not read image {im_filename.name}.  JPEG compression with NBITS==12 is unsupported, '
                    f'you probably need to recompress this image.'
                )
            else:
                raise ex
        is_masked = any([MaskFlags.all_valid not in im.mask_flag_enums[bi] for bi in range(im.count)])
        if im.nodata is None and not is_masked:
            logger.warning(f'{im_filename.name} has no mask or nodata value, '
                           f'any invalid pixels should be masked before homogenising.')
        im_bands = [bi + 1 for bi in range(im.count) if im.colorinterp[bi] != ColorInterp.alpha]
    return im_bands


def _inspect_image_pair(src_filename, ref_filename, proc_crs='auto'):
    # check ref_filename has enough bands
    cmp_bands = _inspect_image(src_filename)
    ref_bands = _inspect_image(ref_filename)
    if len(cmp_bands) > len(ref_bands):
        raise ImageContentError(f'{ref_filename.name} has fewer non-alpha bands than {src_filename.name}.')

    # warn if band counts don't match
    if len(cmp_bands) != len(ref_bands):
        logger.warning(f'Image non-alpha band counts don`t match. Using the first {len(cmp_bands)} non-alpha bands '
                       f'of {ref_filename.name}.')

    with rio.open(src_filename, 'r') as cmp_im:
        with WarpedVRT(rio.open(ref_filename, 'r'), crs=cmp_im.crs, resampling=Resampling.bilinear) as ref_im:
            # check coverage
            if not box(*ref_im.bounds).covers(box(*cmp_im.bounds)):
                raise ImageContentError('Reference extent does not cover image.')

            src_pixel_smaller = np.prod(cmp_im.res) < np.prod(ref_im.res)
            cmp_str = "smaller" if src_pixel_smaller else "larger"
            if proc_crs == 'auto':
                proc_crs = 'ref' if src_pixel_smaller else 'src'
                logger.debug(f'Source pixel size is {cmp_str} than the reference, '
                             f'using model_crs="{proc_crs}"')
            elif ((proc_crs == 'src' and src_pixel_smaller) or
                  (proc_crs == 'ref' and not src_pixel_smaller)):
                rec_crs_str = "ref" if src_pixel_smaller else "src"
                logger.warning(f'model_crs="{rec_crs_str}" is recommended when '
                               f'the source image pixel size is {cmp_str} than the reference.')

    return cmp_bands, ref_bands, proc_crs
