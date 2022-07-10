"""
    Homonim: Correction of aerial and satellite imagery to surface relfectance
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
import threading
from collections import OrderedDict
from multiprocessing import cpu_count
from contextlib import ExitStack
from concurrent import futures
from typing import List, Dict, Tuple

import numpy as np
import rasterio as rio
import yaml
from rasterio.windows import get_data_window, intersect, union, Window, transform
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from tabulate import TableFormat, Line, DataRow, tabulate

from homonim import utils
from homonim.enums import Model
from homonim import errors

logger = logging.getLogger(__name__)

# TODO: make it clear here, and in compare what is R2 coeff of det, and PCC
class ParamStats:
    def __init__(self, param_filename):
        """
        Class to calculate the statistics of a parameter image.

        Parameters
        ----------
        param_filename: pathlib.Path, str
            Path to the parameter image file, as produced by :meth:`homonim.fuse.RasterFuse.process` with
        """
        self._param_filename = pathlib.Path(param_filename)

        utils.validate_param_image(self._param_filename)

        # read some parameters from the metadata
        with rio.open(self._param_filename, 'r') as self._param_im:
            self._tags = self._param_im.tags()
            self._model = self._tags['FUSE_MODEL'].replace('_', '-')
            self._r2_inpaint_thresh = yaml.safe_load(self._tags['FUSE_R2_INPAINT_THRESH'])
        self._param_im: rio.DatasetReader = self._param_im

    @property
    def closed(self):
        """ True if the parameter file is closed, otherwise False. """
        return not self._param_im or self._param_im.closed

    @property
    def metadata(self):
        """ A printable string of parameter metadata. """
        res_str = (
            f'Model: {self._model}\n'
            f'Kernel shape: {self._tags["FUSE_KERNEL_SHAPE"]}\n'
            f'Processing CRS: {self._tags["FUSE_PROC_CRS"]}\n'
            f'Reference: {self._tags["FUSE_REF_FILE"]}\n'
        )
        if self._model == 'gain-offset':
            res_str += f'R\N{SUPERSCRIPT TWO} inpaint threshold: {self._r2_inpaint_thresh}\n'
        return res_str

    @staticmethod
    def stats_table(stats_list: List[Dict]):
        """
        Return a table string of the provided parameter statistics.

        Parameters
        ----------
        stats_list: list of dict
            Parameter statistics to tabulate, as returned by :meth:`ParamStats.stats`.

        Returns
        -------
        str
            A table string.
        """
        headers = dict(band='Band', mean='Mean', std='Std.', min='Min', max='Max', inpaint_p='Inpaint (%)')
        return tabulate(stats_list, headers=headers, floatfmt='.3f', stralign='right', tablefmt=utils.table_format)

    def __enter__(self):
        self._stack = ExitStack()
        self._stack.enter_context(rio.Env(GDAL_NUM_THREADS='ALL_CPUs'))
        self._stack.enter_context(logging_redirect_tqdm([logging.getLogger(__package__)]))
        self._param_im = rio.open(self._param_filename, 'r')
        self._stack.enter_context(self._param_im)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.__exit__(exc_type, exc_val, exc_tb)

    def _assert_open(self):
        """ Utility method to raise an exception if the parameter file is not open. """
        if self.closed:
            raise errors.IoError(f'The parameter file has not been opened: {self._param_filename.name}')

    def _get_data_window(self, threads: int = cpu_count()) -> Window:
        """ Threaded function to accumulate the parameter image data window over blocks.  """
        self._assert_open()
        read_lock = threading.Lock()

        def get_block_data_window(block_win: Window):
            """ Return a window of valid data corresponding to a given block.  """
            with read_lock:
                mask = self._param_im.read_masks(indexes=1, window=block_win)
            block_data_win = get_data_window(mask, nodata=0)
            # offset block_data_win to the UL corner of block_win and return
            return Window(
                block_win.col_off + block_data_win.col_off, block_win.row_off + block_data_win.row_off,
                block_data_win.width, block_data_win.height
            )

        # combine valid block windows into a valid image window
        data_win: Window = None
        bar_format = 'Finding window: {l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'
        with futures.ThreadPoolExecutor(max_workers=threads) as executor:
            # create threads to get valid block windows
            win_futures = [
                executor.submit(get_block_data_window, block_win)
                for block_ij, block_win in self._param_im.block_windows(1)
            ]  # yapf: disable

            # wait for threads and combine returned windows
            for future in tqdm(
                futures.as_completed(win_futures), bar_format=bar_format, total=len(win_futures), dynamic_ncols=True,
                leave=False,
            ):
                block_data_win: Window = future.result()
                if block_data_win.width > 0 and block_data_win.height > 0:
                    data_win = union(data_win, block_data_win) if data_win else block_data_win
        return data_win

    def _get_image_stats(self, image_accum: List[Dict]) -> List[Dict]:
        """ Utility method to calculate image statistics from accumulated results. """
        image_stats = []
        for band_i, band_accum in enumerate(image_accum):
            band_stats = dict(
                band=self._param_im.descriptions[band_i],
                mean=band_accum['sum'] / band_accum['n'],
                std=np.sqrt((band_accum['sum2'] / band_accum['n']) - (band_accum['sum']**2 / band_accum['n']**2)),
                min=band_accum['min'],
                max=band_accum['max'],
            )  # yapf: disable
            if 'inpaint_sum' in band_accum:
                band_stats['inpaint_p'] = 100 * band_accum['inpaint_sum'] / band_accum['n']
            image_stats.append(band_stats)
        return image_stats

    def stats(self, threads: int=cpu_count()) -> Dict[str, Dict]:
        """
        Find parameter image statistics.

        Returns
        -------
        param_dict: dict[dict]
            A dictionary representing the results.
        """
        threads = utils.validate_threads(threads)
        self._assert_open()
        data_win = self._get_data_window(threads=threads)  # get valid data window
        read_lock = threading.Lock()

        def get_block_sums(band_i: int, block_win: rio.windows.Window) -> Tuple[Dict, int]:
            """ Thread-safe function to find sums etc for a given block and band. """
            with read_lock:
                array: np.ma.masked_array = self._param_im.read(
                    indexes=band_i + 1, window=block_win, masked=True, out_dtype='float64',
                )
            block_dict = dict(min=array.min(), max=array.max(), sum=array.sum(), sum2=(array**2).sum(), n=array.count())
            if (self._model == Model.gain_offset) and (band_i >= self._param_im.count * 2 / 3):
                # find the sum of inpainted pixels, if this is a R2 band and a gain-offset model
                block_dict.update(inpaint_sum=(array < self._r2_inpaint_thresh).sum())
            return block_dict, band_i

        # accumulate the block sums, min, max etc over the image
        image_accum = [{} for i in range(self._param_im.count)]
        bar_format = 'Finding stats: {l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'
        with futures.ThreadPoolExecutor(max_workers=threads) as executor:
            # create threads to get block sums etc
            stats_futures = [
                executor.submit(get_block_sums, band_i, block_win)
                for band_i in range(self._param_im.count)
                for block_ij, block_win in self._param_im.block_windows(band_i + 1)
                if intersect(data_win, block_win)
            ]  # yapf: disable

            # wait for threads, and accumulate block results over the image
            for future in tqdm(
                futures.as_completed(stats_futures), bar_format=bar_format, total=len(stats_futures),
                dynamic_ncols=True,
            ):
                block_dict, band_i = future.result()
                image_accum[band_i].update(
                    min=np.nanmin((image_accum[band_i].get('min', np.inf), block_dict['min'])),
                    max=np.nanmax((image_accum[band_i].get('max', -np.inf), block_dict['max'])),
                    sum=np.nansum((image_accum[band_i].get('sum', 0), block_dict['sum'])),
                    sum2=np.nansum((image_accum[band_i].get('sum2', 0), block_dict['sum2'])),
                    n=np.nansum((image_accum[band_i].get('n', 0), block_dict['n']))
                )
                if 'inpaint_sum' in block_dict:
                    image_accum[band_i].update(
                        inpaint_sum=np.nansum((image_accum[band_i].get('inpaint_sum', 0), block_dict['inpaint_sum']))
                    )

        # find the image statistics and return
        return self._get_image_stats(image_accum)

    def _stats(self):
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
            # param_dict = OrderedDict([(param_desc, None) for param_desc in param_im.descriptions])
            param_list = []

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
                    """ Find mean, std, min & max statistics for a vector. """
                    if v.mask.all():  # all values are invalid (nan / +-inf)
                        return OrderedDict(Mean=float('nan'), Std=float('nan'), Min=float('nan'), Max=float('nan'))
                    else:
                        return OrderedDict(Mean=v.mean(), Std=v.std(), Min=v.min(), Max=v.max())

                stats_dict = stats(param_vec)
                if (self._model == Model.gain_offset) and (band_i >= param_im.count * 2 / 3):
                    # Find the r2 inpaint portion if these parameters are from Model.gain_offset
                    inpaint_portion = np.sum(param_vec < self._r2_inpaint_thresh) / len(param_vec)
                    stats_dict['Inpaint (%)'] = inpaint_portion * 100

                return param_im.descriptions[band_i], stats_dict

            # loop over bands (i.e. parameters), finding statistics
            with futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                stats_futures = [executor.submit(process_band, band_i) for band_i in range(param_im.count)]

                # wait for threads, get results and raise any thread generated exceptions
                for future in tqdm(futures.as_completed(stats_futures), bar_format=bar_format, total=len(stats_futures)):
                    param_desc, stats_dict = future.result()
                    stats_dict.update(Band=param_desc)
                    param_list.append(stats_dict)

        return param_list
