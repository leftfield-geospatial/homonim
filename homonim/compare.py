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

import concurrent.futures
import logging
import multiprocessing
import pathlib
import threading

import numpy as np
import rasterio as rio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling

from homonim.inspect import _inspect_image_pair
from homonim.raster_array import RasterArray, expand_window_to_grid

logger = logging.getLogger(__name__)


class ImCompare():
    def __init__(self, cmp_filename, ref_filename, proc_crs='auto'):
        """
        Class for comparing images

        Parameters
        ----------
        cmp_filename : str, pathlib.Path
            Source image filename.
        ref_filename: str, pathlib.Path
            Reference image filename.
        """
        self._cmp_filename = pathlib.Path(cmp_filename)
        self._ref_filename = pathlib.Path(ref_filename)

        self._ref_bands = None
        self._cmp_bands = None
        self._warped_vrt_dict = None
        self._profile = False
        self._proc_crs = proc_crs
        self._cmp_bands, self._ref_bands = _inspect_image_pair(self._cmp_filename, self._ref_filename)

    def _image_init(self):
        """Check bounds, band count, and compression type of source and reference images"""
        self._cmp_bands, self._ref_bands = _inspect_image_pair(self._cmp_filename, self._ref_filename)

        with rio.open(self._cmp_filename, 'r') as cmp_im:
            with rio.open(self._ref_filename, 'r') as ref_im:
                cmp_pixel_smaller = np.prod(cmp_im.res) < np.prod(ref_im.res)
                if cmp_pixel_smaller:
                    self._im_filenames = [self._cmp_filename, self._ref_filename]
                    self._im_bands = [self._cmp_bands, self._ref_bands]
                    proc_im = ref_im
                else:
                    self._im_filenames = [self._ref_filename, self._cmp_filename]
                    self._im_bands = [self._ref_bands, self._cmp_bands]
                    proc_im = cmp_im

                self._proc_win = expand_window_to_grid(
                    proc_im.window(*cmp_im.bounds),
                    expand_pixels=(1, 1)  # np.ceil(np.divide(cmp_im.res, ref_im.res)).astype('int')
                )
                proc_transform = proc_im.window_transform(self._proc_win)
                self._warped_vrt_dict = dict(crs=proc_im.crs, transform=proc_transform, width=self._proc_win.width,
                                             height=self._proc_win.height, resampling=Resampling.average)

    def compare(self):
        cmp_read_lock = threading.Lock()
        ref_read_lock = threading.Lock()
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._ref_filename, 'r') as ref_im:
            with rio.open(self._cmp_filename, 'r') as cmp_im:
                proc_in_ref_crs = np.prod(cmp_im.res) < np.prod(ref_im.res)

                def process_band(band_i):
                    with cmp_read_lock:
                        cmp_ra = RasterArray.from_rio_dataset(cmp_im, indexes=self._cmp_bands[band_i])
                    expand_pixels = np.ceil(np.divide(cmp_im.res, ref_im.res)).astype('int')
                    ref_win = expand_window_to_grid(ref_im.window(*cmp_ra.bounds), expand_pixels=expand_pixels)
                    with ref_read_lock:
                        ref_ra = RasterArray.from_rio_dataset(ref_im, indexes=self._ref_bands[band_i], window=ref_win,
                                                              boundless=True)
                    if proc_in_ref_crs:
                        cmp_ra = cmp_ra.reproject(**ref_ra.proj_profile, resampling=Resampling.average)
                    else:
                        ref_ra = ref_ra.reproject(**cmp_ra.proj_profile, resampling=Resampling.average)

                    mask = cmp_ra.mask & ref_ra.mask
                    cmp_vec = cmp_ra.array[mask]
                    ref_vec = ref_ra.array[mask]
                    r = np.corrcoef(cmp_vec, ref_vec)[0, 1]
                    mse = np.mean((cmp_vec - ref_vec) ** 2)
                    return dict(r=r, mse=mse)

                if True:
                    future_list = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                        for band_i in range(len(self._cmp_bands)):
                            future = executor.submit(process_band, band_i)
                            future_list.append(future)

                        # wait for threads and raise any thread generated exceptions
                        res_list = []
                        for future in future_list:
                            res_dict = future.result()
                            res_list.append(res_dict)
                        print(res_list)
                else:
                    res_list = []
                    for band_i in range(len(self._cmp_bands)):
                        res_dict = process_band(band_i)
                        res_list.append(res_dict)
                    print(res_list)

    def _compare(self):
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            with WarpedVRT(rio.open(self._im_filenames[0], 'r'), **self._warped_vrt_dict) as im0:
                with rio.open(self._im_filenames[1], 'r') as im1:
                    for bi in range(len(self._cmp_bands)):
                        im0_ra = RasterArray.from_rio_dataset(im0, indexes=self._im_bands[0][bi])
                        im1_ra = RasterArray.from_rio_dataset(im1, indexes=self._im_bands[0][bi], window=self._proc_win,
                                                              boundless=True)
                        mask = im0_ra.mask & im1_ra.mask
                        r = np.corrcoef(im0_ra.array[mask], im1_ra.array[mask])[0, 1]
                        print(im0_ra.shape)
