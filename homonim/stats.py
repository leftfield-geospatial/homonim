"""
    Homonim: Correction of aerial and satellite imagery to surface reflectance
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
from concurrent import futures
from contextlib import ExitStack
from multiprocessing import cpu_count
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import rasterio as rio
import yaml
from rasterio.windows import get_data_window, intersect, union, Window
from tabulate import tabulate
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from homonim import utils, errors
from homonim.enums import Model

logger = logging.getLogger(__name__)


class ParamStats:

    def __init__(self, param_filename: Union[pathlib.Path, str]):
        """
        Class to calculate the statistics of a parameter image.

        Parameters
        ----------
        param_filename: pathlib.Path, str
            Path to a parameter image file, as created by :meth:`homonim.RasterFuse.process` with the
            ``param_filename`` argument specified.
        """
        self._param_filename = pathlib.Path(param_filename)

        utils.validate_param_image(self._param_filename)

        # read some parameters from the metadata
        self._param_im: rio.DatasetReader
        with rio.open(self._param_filename, 'r') as self._param_im:
            self._tags = self._param_im.tags()
            self._model = self._tags['FUSE_MODEL'].replace('_', '-')
            self._r2_inpaint_thresh = yaml.safe_load(self._tags['FUSE_R2_INPAINT_THRESH'])

    schema = dict(
        band=dict(abbrev='Band'),
        mean=dict(abbrev='Mean'),
        std=dict(abbrev='Std.'),
        min=dict(abbrev='Min.'),
        max=dict(abbrev='Max.'),
        inpaint_p=dict(abbrev='Inpaint (%)', description='Portion of inpainted pixels (%).'),
    )  # yapf: disable
    """ Dictionary describing the statistics returned by :meth:`ParamStats.stats`. """

    @property
    def closed(self) -> bool:
        """ True if the parameter file is closed, otherwise False. """
        return not self._param_im or self._param_im.closed

    @property
    def metadata(self) -> str:
        """ Parameter metadata string. """
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
    def schema_table() -> str:
        """ Return a table string describing statistics returned by :meth:`ParamStats.stats`. """
        schema_list = [v for k, v in ParamStats.schema.items() if 'description' in v]
        schema_list.append(dict(abbrev='*_R2', description='R\N{SUPERSCRIPT TWO} coefficient of determination.'))
        headers = {k: k.upper() for k in schema_list[0].keys()}
        return tabulate(schema_list, headers=headers, tablefmt=utils.table_format)

    @staticmethod
    def stats_table(stats_list: List[Dict]) -> str:
        """
        Return a table string for the provided parameter statistics.

        Parameters
        ----------
        stats_list: list of dict
            Parameter statistics to tabulate, as returned by :meth:`ParamStats.stats`.

        Returns
        -------
        str
            A table string.
        """
        headers = {k: v['abbrev'] for k, v in ParamStats.schema.items() if k in stats_list[-1]}
        return tabulate(stats_list, headers=headers, floatfmt='.3f', stralign='right', tablefmt=utils.table_format)

    def __enter__(self):
        self._stack = ExitStack()
        self._stack.enter_context(rio.Env(GDAL_NUM_THREADS='ALL_CPUs', GTIFF_FORCE_RGBA=False))
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

        def get_block_data_window(block_win: Window) -> Optional[Window]:
            """ Return a window of valid data corresponding to a given block.  """
            with read_lock:
                mask = self._param_im.read_masks(indexes=1, window=block_win)
            _block_data_win = get_data_window(mask, nodata=0)

            if _block_data_win.width == 0 or _block_data_win.height == 0:
                # return None if there is no valid data in the block
                return None
            # offset _block_data_win to the UL corner of block_win and return
            return Window(
                block_win.col_off + _block_data_win.col_off, block_win.row_off + _block_data_win.row_off,
                _block_data_win.width, _block_data_win.height
            )

        # combine valid block windows into a valid image window
        im_data_win: Optional[Window] = None
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
            ):  # yapf: disable
                block_data_win: Window = future.result()
                if block_data_win:
                    im_data_win = union(im_data_win, block_data_win) if im_data_win else block_data_win
        return im_data_win

    def _get_image_stats(self, image_accum: List[Dict]) -> List[Dict]:
        """ Utility method to calculate image statistics from accumulated results. """
        image_stats = []
        for band_i, band_accum in enumerate(image_accum):
            band_stats = dict(
                band=self._param_im.descriptions[band_i],
                mean=band_accum['sum'] / band_accum['n'],
                # formula for cumulative std dev from:
                # https://rosettacode.org/wiki/Cumulative_standard_deviation#Python
                std=np.sqrt((band_accum['sum2'] / band_accum['n']) - (band_accum['sum'] ** 2 / band_accum['n'] ** 2)),
                min=band_accum['min'],
                max=band_accum['max'],
            )  # yapf: disable

            if 'inpaint_sum' in band_accum:
                band_stats['inpaint_p'] = 100 * band_accum['inpaint_sum'] / band_accum['n']
            image_stats.append(band_stats)
        return image_stats

    def stats(self, threads: int = 0) -> List[Dict]:
        """
        Find parameter image statistics.

        Statistics are accumulated over image blocks to limit memory usage for large images.

        Parameters
        ----------
        threads: int, optional
            Number of image blocks to process concurrently.  A maximum of the number of processors on your
            system is allowed.  Increasing this number will increase the memory required for processing.
            0 = use all processors.

        Returns
        -------
        list of dict
            A list of parameter band statistics.
        """
        self._assert_open()
        threads = utils.validate_threads(threads)
        data_win = self._get_data_window(threads=threads)  # get valid data window
        read_lock = threading.Lock()

        def get_block_sums(band_i: int, block_win: rio.windows.Window) -> Tuple[Dict, int]:
            """ Thread-safe function read a block and find its sums etc. """
            with read_lock:
                array: np.ma.masked_array = self._param_im.read(
                    indexes=band_i + 1, window=block_win, masked=True, out_dtype='float64',
                )
            _block_dict = dict(
                min=array.min(), max=array.max(), sum=array.sum(), sum2=(array ** 2).sum(), n=array.count()
            )
            if (self._model == Model.gain_offset) and (band_i >= self._param_im.count * 2 / 3):
                # find the sum of inpainted pixels, if this is a R2 band and a gain-offset model
                _block_dict.update(inpaint_sum=(array < self._r2_inpaint_thresh).sum())
            return _block_dict, band_i

        # find block sums, min, max etc. in threads, and accumulate over the image
        image_accum = [{} for _ in range(self._param_im.count)]
        bar_format = 'Finding stats: {l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'
        with futures.ThreadPoolExecutor(max_workers=threads) as executor:
            # create threads to get block sums etc.
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
            ):  # yapf: disable
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
