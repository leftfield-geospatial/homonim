# Copyright Leftfield Geospatial
#
# This file is part of Homonim.
#
# Homonim is free software: you can redistribute it and/or modify it under the terms
# of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# Homonim is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with
# Homonim. If not, see <https://www.gnu.org/licenses/>.

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from itertools import product
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
import yaml
from rasterio.windows import Window
from tabulate import tabulate
from tqdm.auto import tqdm

from homonim import utils
from homonim.enums import Model
from homonim.errors import HomonimError

logger = logging.getLogger(__name__)

# description of the statistics returned by ParamStats.stats()
_stats_schema = dict(
    band=dict(abbrev='Band'),
    mean=dict(abbrev='Mean'),
    std=dict(abbrev='Std.', description='Standard deviation'),
    min=dict(abbrev='Min.', description='Minimum'),
    max=dict(abbrev='Max.', description='Maximum'),
    inpaint_p=dict(
        abbrev='Inpaint (%)', description='Portion of inpainted pixels (%).'
    ),
)


def _validate_param_image(param_file: str | PathLike):
    """Validate the given parameter image."""
    param_file = os.fspath(param_file)
    param_name = Path(param_file).name

    err_msg = f"'{param_name}' is not a valid parameter image."
    with rio.open(param_file) as param_im:
        tags = param_im.tags()
        # check band count is a multiple of 3 and that expected metadata tags exist
        if (
            not param_im.count
            or param_im.count % 3 != 0
            or not {'FUSE_MODEL', 'FUSE_KERNEL_SHAPE', 'FUSE_PROC_CRS', 'FUSE_REF_FILE'}
            <= set(tags)
        ):
            raise HomonimError(err_msg)

        # check band descriptions end with the expected suffixes
        n_refl_bands = int(param_im.count / 3)
        suffixes = (
            ['gain'] * n_refl_bands + ['offset'] * n_refl_bands + ['r2'] * n_refl_bands
        )
        if not all(
            desc.lower().endswith(suffix)
            for suffix, desc in zip(suffixes, param_im.descriptions, strict=True)
        ):
            raise HomonimError(err_msg)

        if not np.isnan(param_im.nodata):
            raise HomonimError(err_msg)


class ParamStats:
    def __init__(self, param_filename: str | PathLike):
        """
        Class to calculate the statistics of a parameter image.

        Can be used as a context manager to automatically close the parameter image
        on exiting.

        :param param_filename:
            Path or URI of a parameter image, as created by
            :meth:`homonim.fuse.RasterFuse.process` with the ``param_filename``
            argument provided.
        """
        param_filename = os.fspath(param_filename)
        _validate_param_image(param_filename)
        self._stack = ExitStack()
        try:
            self._stack.enter_context(rio.Env(GDAL_NUM_THREADS='ALL_CPUs'))
            self._param_im = self._stack.enter_context(rio.open(param_filename, 'r'))

            # read some parameters from the tags
            self._tags = self._param_im.tags()
            self._model = self._tags['FUSE_MODEL'].replace('_', '-')
            self._r2_inpaint_thresh = yaml.safe_load(
                self._tags['FUSE_R2_INPAINT_THRESH']
            )

        except Exception:
            self._stack.close()
            raise

    @property
    def closed(self) -> bool:
        """Whether the parameter image is closed."""
        return self._param_im.closed

    @property
    def metadata(self) -> str:
        """Parameter metadata string."""
        res_str = (
            f'Model: {self._model}\n'
            f'Kernel shape: {self._tags["FUSE_KERNEL_SHAPE"]}\n'
            f'Processing CRS: {self._tags["FUSE_PROC_CRS"]}\n'
            f'Reference: {self._tags["FUSE_REF_FILE"]}\n'
        )
        if self._model == Model.gain_offset:
            res_str += (
                f'R\N{SUPERSCRIPT TWO} inpaint threshold: {self._r2_inpaint_thresh}\n'
            )
        return res_str

    @staticmethod
    def schema_table() -> str:
        """Return a table string describing the statistics returned by
        :meth:`~ParamStats.stats`.
        """
        schema_list = [v for k, v in _stats_schema.items() if 'description' in v]
        schema_list.append(
            dict(
                abbrev='*_R2',
                description='R\N{SUPERSCRIPT TWO} coefficient of determination.',
            )
        )
        headers = {k: k.upper() for k in schema_list[0].keys()}
        return tabulate(schema_list, headers=headers, tablefmt=utils.table_format)

    @staticmethod
    def stats_table(stats_list: list[dict[str, float]]) -> str:
        """
        Tabulate the provided parameter image statistics.

        :param stats_list:
            Parameter statistics, as returned by :meth:`~ParamStats.stats`.

        :return:
            Table string.
        """
        headers = {
            k: v['abbrev'] for k, v in _stats_schema.items() if k in stats_list[-1]
        }
        return tabulate(
            stats_list,
            headers=headers,
            floatfmt='.3f',
            stralign='right',
            tablefmt=utils.table_format,
        )

    @utils.assert_open
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.__exit__(exc_type, exc_val, exc_tb)

    def _get_image_stats(
        self, image_accum: list[dict[str, Any]]
    ) -> list[dict[str, float]]:
        """Calculate the image statistics from the accumulated sum etc. values."""
        image_stats = []
        for band_i, band_accum in enumerate(image_accum):
            band_stats = dict(
                band=self._param_im.descriptions[band_i],
                mean=float(band_accum['sum'] / band_accum['n']),
                # formula for cumulative std dev from:
                # https://rosettacode.org/wiki/Cumulative_standard_deviation#Python
                std=float(
                    np.sqrt(
                        (band_accum['sum2'] / band_accum['n'])
                        - (band_accum['sum'] ** 2 / band_accum['n'] ** 2)
                    )
                ),
                min=float(band_accum['min']),
                max=float(band_accum['max']),
            )

            if 'inpaint_sum' in band_accum:
                band_stats['inpaint_p'] = float(
                    100 * band_accum['inpaint_sum'] / band_accum['n']
                )

            image_stats.append(band_stats)

        return image_stats

    def close(self):
        """Close the parameter image."""
        self._stack.close()

    @utils.assert_open
    def stats(self, threads: int = 0) -> list[dict[str, float]]:
        """
        Find parameter image statistics.

        :param threads:
            Number of image blocks to process concurrently.  If ``0``, the number of
            CPUs is used.

        :return:
            List of dictionaries of parameter band statistics.
        """
        threads = utils.validate_threads(threads)
        read_lock = threading.Lock()

        def get_block_stats(band_i: int, block_win: Window) -> dict[str, Any]:
            """Thread-safe function read a block and find its sums etc."""
            with read_lock:
                array = self._param_im.read(
                    indexes=band_i + 1, window=block_win, out_dtype='float64'
                )

            mask = ~np.isnan(array)
            array_ = array[mask]
            block_dict = dict(
                min=array_.min(initial=np.inf),
                max=array_.max(initial=-np.inf),
                sum=array_.sum(initial=0),
                sum2=(array_**2).sum(initial=0),
                n=mask.sum(initial=0),
            )
            if (self._model == Model.gain_offset) and (
                band_i >= self._param_im.count * 2 / 3
            ):
                # find the sum of inpainted pixels if this is a R2 band and a
                # gain-offset model
                inpaint_sum = (array_ < self._r2_inpaint_thresh).sum(initial=0)
                block_dict.update(inpaint_sum=inpaint_sum)

            return block_dict

        # find block sums etc. in a thread pool, and accumulate over the image in the
        # main thread
        image_accum = [{} for _ in range(self._param_im.count)]
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures_dict = {
                executor.submit(get_block_stats, band_i, block_win): band_i
                for band_i, (_, block_win) in product(
                    range(self._param_im.count), self._param_im.block_windows(1)
                )
            }

            for future in tqdm(
                as_completed(futures_dict),
                bar_format=bar_format,
                total=len(futures_dict),
                dynamic_ncols=True,
            ):
                try:
                    block_dict = future.result()
                    band_accum = image_accum[futures_dict[future]]
                    band_accum.update(
                        min=np.min((band_accum.get('min', np.inf), block_dict['min'])),
                        max=np.max((band_accum.get('max', -np.inf), block_dict['max'])),
                        sum=np.sum((band_accum.get('sum', 0), block_dict['sum'])),
                        sum2=np.sum((band_accum.get('sum2', 0), block_dict['sum2'])),
                        n=np.sum((band_accum.get('n', 0), block_dict['n'])),
                    )

                    if 'inpaint_sum' in block_dict:
                        inpaint_sum = np.sum(
                            (
                                band_accum.get('inpaint_sum', 0),
                                block_dict['inpaint_sum'],
                            )
                        )
                        band_accum.update(inpaint_sum=inpaint_sum)
                except Exception as ex:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError('Could not process block.') from ex

        # convert the accumulated sums etc. to image stats
        return self._get_image_stats(image_accum)
