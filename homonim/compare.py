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

from homonim.enums import ProcCrs
from homonim.raster_pair import RasterPairReader

logger = logging.getLogger(__name__)


class RasterCompare():
    """Class to statistically compare an image against a reference"""
    default_config = dict(multithread=True)

    def __init__(self, src_filename, ref_filename, proc_crs=ProcCrs.auto, multithread=default_config['multithread']):
        """
        Construct the RasterCompare object

        Parameters
        ----------
        src_filename: pathlib.Path, str
                      Path to the source image file
        ref_filename: pathlib.Path, str
                      Path to the reference image file (whose spatial extent should cover that of src_filename)
        proc_crs: homonim.enums.ProcCrs, optional
                  The image CRS and resolution in which to perform the comparison.  proc_crs=ProcCrs.auto will
                  automatically choose the lowest resolution of the source and reference CRS's (recommended)
        multithread: bool, optional
                     Compare image bands concurrently (requires more memory).
        """
        self._src_filename = pathlib.Path(src_filename)
        self._ref_filename = pathlib.Path(ref_filename)
        self._multithread = multithread

        # check src and ref image validity via RasterPairReader and get proc_crs
        # self._raster_pair is initialised to read in bands (not blocks)
        if not isinstance(proc_crs, ProcCrs):
            raise ValueError("'proc_crs' must be an instance of homonim.enums.ProcCrs")
        self._raster_pair = RasterPairReader(self._src_filename, self._ref_filename, proc_crs=proc_crs)
        self._proc_crs = self._raster_pair.proc_crs

    def compare(self):
        """
        Statistically compare source and reference images and return results.

        Returns
        -------
        res_dict: dict[dict]
                  A dictionary representing the results.
        """
        res_dict = OrderedDict()
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} bands [{elapsed}<{remaining}]'

        with self._raster_pair as raster_pair:

            def process_band(block_pair):
                """Thread-safe function to process a block (that encompasses the full band)"""
                src_ra, ref_ra = raster_pair.read(block_pair)   # read src and ref bands

                # re-project into the lowest resolution (_proc_crs) space
                if self._proc_crs == ProcCrs.ref:
                    src_ra = src_ra.reproject(**ref_ra.proj_profile, resampling=Resampling.average)
                else:
                    ref_ra = ref_ra.reproject(**src_ra.proj_profile, resampling=Resampling.average)

                def get_stats(src_vec, ref_vec):
                    """Find comparison statistics between two vectors"""
                    r = float(np.corrcoef(src_vec, ref_vec)[0, 1])  # Pearson's correlation coefficient
                    rmse = float(np.sqrt(np.mean((src_vec - ref_vec) ** 2)))    # Root mean square error
                    rrmse = float(rmse / np.mean(ref_vec))  # Relative RMSE
                    return OrderedDict(r2=r ** 2, RMSE=rmse, rRMSE=rrmse, N=len(src_vec))

                mask = src_ra.mask & ref_ra.mask    # combined src and ref mask

                # find stats of valid data
                src_vec = src_ra.array[mask]
                ref_vec = ref_ra.array[mask]
                stats_dict = get_stats(src_vec, ref_vec)

                # form a string desribing the band
                band_desc = (raster_pair.ref_im.descriptions[raster_pair.ref_bands[block_pair.band_i] - 1] or
                             raster_pair.src_im.descriptions[raster_pair.src_bands[block_pair.band_i] - 1] or
                             f'Band {block_pair.band_i + 1}')
                return band_desc, stats_dict

            if self._multithread:
                # process bands in parallel
                future_list = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    for block_pair in raster_pair.block_pairs():
                        future = executor.submit(process_band, block_pair)
                        future_list.append(future)

                    # wait for threads, get results and raise any thread generated exceptions
                    for future in tqdm(future_list, bar_format=bar_format):
                        band_desc, band_dict = future.result()
                        res_dict[band_desc] = band_dict
            else:
                # process bands consecutively
                for block_pair in tqdm(raster_pair.block_pairs(), bar_format=bar_format):
                    band_desc, band_dict = process_band(block_pair)
                    res_dict[band_desc] = band_dict

        # use a pandas dataframe to find the mean of the statistics over the bands
        res_df = pd.DataFrame.from_dict(res_dict, orient='index')
        res_df.loc['Mean'] = res_df.mean()
        res_df.N = res_df.N.astype('int')

        return res_df.to_dict(orient='index')
