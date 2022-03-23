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
import pathlib
import threading
from collections import OrderedDict
from multiprocessing import cpu_count

import numpy as np
import rasterio as rio
import yaml
from rasterio.windows import get_data_window
from tqdm import tqdm

from homonim import utils
from homonim.enums import Method
from homonim.errors import ImageFormatError

logger = logging.getLogger(__name__)


class ParamStats:
    """Class to find statistics of a parameter image."""
    default_config = dict(threads=cpu_count() - 1)

    def __init__(self, param_filename, threads=default_config['threads']):
        """
        Construct the ParamStats object

        Parameters
        ----------
        param_filename: pathlib.Path, str
            Path to the parameter image file
        threads: int, optional
            The number of threads to use for concurrent processing of bands (requires more memory).  0 = use all cpus.
        """
        self._param_filename = pathlib.Path(param_filename)
        self._threads = utils.validate_threads(threads)

        utils.validate_param_image(self._param_filename)

        # read some parameters from the metadata
        with rio.open(self._param_filename, 'r') as param_im:
            self._tags = param_im.tags()
            self._method = self._tags['HOMO_METHOD'].replace('_', '-')
            self._r2_inpaint_thresh = yaml.safe_load(self._tags['HOMO_MODEL_CONF'])['r2_inpaint_thresh']

    @property
    def metadata(self):
        """Return a printable string of parameter metadata"""
        res_str = (f'Method: {self._method}\n'
                   f'Kernel shape: {self._tags["HOMO_KERNEL_SHAPE"]}\n'
                   f'Processing CRS: {self._tags["HOMO_PROC_CRS"]}')
        if self._method == 'gain-offset':
            res_str += f'\nR\N{SUPERSCRIPT TWO} inpaint threshold: {self._r2_inpaint_thresh}'
        return res_str

    def stats(self):
        """
        Find parameter image statistics.

        Returns
        -------
        param_dict: dict[dict]
            A dictionary representing the results.
        """

        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} bands [{elapsed}<{remaining}]'
        read_lock = threading.Lock()

        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._param_filename) as param_im:
            # initialise the results dictionary in correct band order
            param_dict = OrderedDict([(param_desc, None) for param_desc in param_im.descriptions])

            # read mask once, assuming the same for all bands
            _mask = param_im.read_masks(indexes=1)

            # get window of valid pixels and re-read windowed mask
            valid_win = get_data_window(_mask, nodata=0)
            del _mask
            mask = param_im.read_masks(indexes=1, window=valid_win).astype('bool', copy=False)

            def process_band(band_i):
                with read_lock:
                    param_array = param_im.read(indexes=band_i + 1, window=valid_win, out_dtype='float32')
                param_vec = param_array[mask]  # vector of valid parameter values
                del param_array
                param_vec = np.ma.masked_invalid(param_vec).astype('float')  # mask out nan and inf values

                def stats(v: np.ma.core.MaskedArray):
                    """Find mean, std, min & max statistics for a vector."""
                    if v.mask.all():     # all values are invalid (nan / +-inf)
                        return OrderedDict(Mean=float('nan'), Std=float('nan'), Min=float('nan'), Max=float('nan'))
                    else:
                        return OrderedDict(Mean=v.mean(), Std=v.std(), Min=v.min(), Max=v.max())

                stats_dict = stats(param_vec)
                if (self._method == Method.gain_offset) and (band_i >= param_im.count * 2 / 3):
                    # Find the r2 inpaint portion if these parameters are from Method.gain_offset
                    inpaint_portion = np.sum(param_vec < self._r2_inpaint_thresh) / len(param_vec)
                    stats_dict['Inpaint (%)'] = inpaint_portion * 100

                return param_im.descriptions[band_i], stats_dict

            # loop over bands (i.e. parameters), finding statistics
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._threads) as executor:
                futures = [executor.submit(process_band, band_i) for band_i in range(param_im.count)]

                # wait for threads, get results and raise any thread generated exceptions
                for future in tqdm(concurrent.futures.as_completed(futures), bar_format=bar_format, total=len(futures)):
                    param_desc, stats_dict = future.result()
                    param_dict[param_desc] = stats_dict

        return param_dict
