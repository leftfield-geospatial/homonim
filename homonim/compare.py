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
from collections import OrderedDict

import numpy as np
import rasterio as rio
from rasterio.warp import Resampling
from tqdm import tqdm
import pandas as pd

from homonim.inspect import _inspect_image_pair
from homonim.raster_array import RasterArray, expand_window_to_grid

logger = logging.getLogger(__name__)


class ImCompare():
    default_config = dict(multithread=True)

    def __init__(self, src_filename, ref_filename, proc_crs='auto', multithread=default_config['multithread']):
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
        self._multithread = multithread
        self._src_bands, self._ref_bands, self._proc_crs = _inspect_image_pair(self._src_filename, self._ref_filename,
                                                                               proc_crs)
    def _compare_im(self):
        """Read all bands and compare"""
        src_read_lock = threading.Lock()
        ref_read_lock = threading.Lock()
        res_dict = {}
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._ref_filename, 'r') as ref_im:
            with rio.open(self._src_filename, 'r') as src_im:
                with src_read_lock:
                    src_ra = RasterArray.from_rio_dataset(src_im, indexes=self._src_bands)
                expand_pixels = np.ceil(np.divide(src_im.res, ref_im.res)).astype(
                    'int')  # TODO: compare to fuse - is this necessary
                ref_win = expand_window_to_grid(ref_im.window(*src_ra.bounds), expand_pixels=expand_pixels)
                with ref_read_lock:
                    ref_ra = RasterArray.from_rio_dataset(ref_im, indexes=self._ref_bands, window=ref_win,
                                                          boundless=True)
                if self._proc_crs == 'ref':
                    src_ra = src_ra.reproject(**ref_ra.proj_profile, resampling=Resampling.average)
                else:
                    ref_ra = ref_ra.reproject(**src_ra.proj_profile, resampling=Resampling.average)

                def get_stats(src_vec, ref_vec):
                    r = float(np.corrcoef(src_vec, ref_vec)[0, 1])
                    rmse = float(np.sqrt(np.mean((src_vec - ref_vec) ** 2)))
                    rrmse = float(rmse/np.mean(ref_vec))
                    return OrderedDict(r2=r ** 2, RMSE=rmse, rRMSE=rrmse, N=len(src_vec))

                mask = src_ra.mask & ref_ra.mask
                for band_i in range(src_ra.count):
                    src_vec = src_ra.array[band_i, mask]
                    ref_vec = ref_ra.array[band_i, mask]
                    band_desc = (ref_im.descriptions[self._ref_bands[band_i] - 1] or
                                 src_im.descriptions[self._src_bands[band_i] - 1] or f'Band {band_i + 1}')
                    res_dict[band_desc] = get_stats(src_vec, ref_vec)
                res_df = pd.DataFrame.from_dict(res_dict, orient='index')
                # mean_ds = res_df.mean()
                src_vec = src_ra.array[:, mask].flatten()
                ref_vec = ref_ra.array[:, mask].flatten()
                res_dict['All'] = get_stats(src_vec, ref_vec)
        return res_dict

    def compare(self):
        src_read_lock = threading.Lock()
        ref_read_lock = threading.Lock()
        res_dict = {}
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} bands [{elapsed}<{remaining}]'
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._ref_filename, 'r') as ref_im:
            with rio.open(self._src_filename, 'r') as src_im:
                def process_band(band_i):
                    with src_read_lock:
                        src_ra = RasterArray.from_rio_dataset(src_im, indexes=self._src_bands[band_i])
                    expand_pixels = np.ceil(np.divide(src_im.res, ref_im.res)).astype(
                        'int')  # TODO: compare to fuse - is this necessary
                    ref_win = expand_window_to_grid(ref_im.window(*src_ra.bounds), expand_pixels=expand_pixels)
                    with ref_read_lock:
                        ref_ra = RasterArray.from_rio_dataset(ref_im, indexes=self._ref_bands[band_i], window=ref_win,
                                                              boundless=True)
                    if self._proc_crs == 'ref':
                        src_ra = src_ra.reproject(**ref_ra.proj_profile, resampling=Resampling.average)
                    else:
                        ref_ra = ref_ra.reproject(**src_ra.proj_profile, resampling=Resampling.average)

                    mask = src_ra.mask & ref_ra.mask
                    src_vec = src_ra.array[mask]
                    ref_vec = ref_ra.array[mask]
                    r = float(np.corrcoef(src_vec, ref_vec)[0, 1])
                    rmse = float(np.sqrt(np.mean((src_vec - ref_vec) ** 2)))
                    rrmse = float(rmse/np.mean(ref_vec))
                    band_desc = (ref_im.descriptions[self._ref_bands[band_i] - 1] or
                                 src_im.descriptions[self._src_bands[band_i] - 1] or f'Band {band_i + 1}')
                    return band_desc, OrderedDict(r2=r ** 2, RMSE=rmse, rRMSE=rrmse, N=len(src_vec))

                if self._multithread:
                    future_list = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                        for band_i in range(len(self._src_bands)):
                            future = executor.submit(process_band, band_i)
                            future_list.append(future)

                        # wait for threads and raise any thread generated exceptions
                        for future in tqdm(future_list, bar_format=bar_format):
                            band_desc, band_dict = future.result()
                            res_dict[band_desc] = band_dict
                else:
                    for band_i in tqdm(range(len(self._src_bands)), bar_format=bar_format):
                        band_desc, band_dict = process_band(band_i)
                        res_dict[band_desc] = band_dict

        res_df = pd.DataFrame.from_dict(res_dict, orient='index')
        res_df.loc['Mean'] = res_df.mean()
        res_df.N = res_df.N.astype('int')

        return res_df.to_dict(orient='index')
