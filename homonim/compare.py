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
from collections import OrderedDict

import numpy as np
import pandas as pd
from rasterio.warp import Resampling
from tqdm import tqdm

from homonim.raster_pair import _inspect_image_pair, ImPairReader

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

    def compare(self):
        res_dict = OrderedDict()
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} bands [{elapsed}<{remaining}]'
        with ImPairReader(self._src_filename, self._ref_filename, proc_crs=self._proc_crs) as im_pair:
            def process_band(block_pair):
                src_ra, ref_ra = im_pair.read(block_pair)

                if self._proc_crs == 'ref':
                    src_ra = src_ra.reproject(**ref_ra.proj_profile, resampling=Resampling.average)
                else:
                    ref_ra = ref_ra.reproject(**src_ra.proj_profile, resampling=Resampling.average)

                def get_stats(src_vec, ref_vec):
                    r = float(np.corrcoef(src_vec, ref_vec)[0, 1])
                    rmse = float(np.sqrt(np.mean((src_vec - ref_vec) ** 2)))
                    rrmse = float(rmse / np.mean(ref_vec))
                    return OrderedDict(r2=r ** 2, RMSE=rmse, rRMSE=rrmse, N=len(src_vec))

                mask = src_ra.mask & ref_ra.mask
                src_vec = src_ra.array[mask]
                ref_vec = ref_ra.array[mask]
                stats_dict = get_stats(src_vec, ref_vec)
                band_desc = (im_pair.ref_im.descriptions[im_pair.ref_bands[block_pair.band_i] - 1] or
                             im_pair.src_im.descriptions[im_pair.src_bands[block_pair.band_i] - 1] or
                             f'Band {block_pair.band_i + 1}')
                return band_desc, stats_dict

            if self._multithread:
                future_list = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    for block_pair in im_pair.block_pairs():
                        future = executor.submit(process_band, block_pair)
                        future_list.append(future)

                    # wait for threads and raise any thread generated exceptions
                    for future in tqdm(future_list, bar_format=bar_format):
                        band_desc, band_dict = future.result()
                        res_dict[band_desc] = band_dict
            else:
                for block_pair in tqdm(im_pair.block_pairs(), bar_format=bar_format):
                    band_desc, band_dict = process_band(block_pair)
                    res_dict[band_desc] = band_dict

        res_df = pd.DataFrame.from_dict(res_dict, orient='index')
        res_df.loc['Mean'] = res_df.mean()
        res_df.N = res_df.N.astype('int')

        return res_df.to_dict(orient='index')
