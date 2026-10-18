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
from contextlib import ExitStack
from os import PathLike
from pathlib import Path

import numpy as np
import rasterio as rio
import yaml
from tabulate import tabulate

from homonim import utils
from homonim.enums import Model
from homonim.errors import HomonimError

logger = logging.getLogger(__name__)


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
            rit = yaml.safe_load(self._tags['FUSE_R2_INPAINT_THRESH'])
            self._r2_inpaint_thresh = (
                float(rit) if isinstance(rit, (float, int)) else None
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
    def stats_table(stats_list: list[dict[str, float]]) -> str:
        """
        Tabulate the provided parameter image statistics.

        :param stats_list:
            Parameter statistics, as returned by :meth:`~ParamStats.stats`.

        :return:
            Table string.
        """
        headers = dict(
            band='Band',
            mean='Mean',
            std='StdDev',
            min='Minimum',
            max='Maximum',
            inpaint_p='Inpaint (%)',
        )
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

    def close(self):
        """Close the parameter image."""
        self._stack.close()

    @utils.assert_open
    def stats(self) -> list[dict[str, float]]:
        """
        Find parameter image statistics.

        :return:
            List of per-band statistics dictionaries
        """
        stats_list = self._param_im.stats(approx=False)
        # convert Statistics objects to dicts
        stats_keys = ['mean', 'std', 'min', 'max']
        stats_list = [{k: getattr(s, k) for k in stats_keys} for s in stats_list]
        # add band descriptions to the dicts
        stats_list = [
            {'band': self._param_im.descriptions[i], **sd}
            for i, sd in enumerate(stats_list)
        ]

        if self._model == Model.gain_offset and self._r2_inpaint_thresh is not None:
            # add the inpaint % to the dicts for the R2 bands
            r2_start_idx = int(2 * self._param_im.count / 3)
            for i, stats_dict in enumerate(stats_list[r2_start_idx:]):
                array = self._param_im.read(indexes=r2_start_idx + i + 1)
                mask = ~np.isnan(array)
                inpaint_p = 100 * (array < self._r2_inpaint_thresh).sum() / mask.sum()
                stats_dict.update(inpaint_p=float(inpaint_p))

        return stats_list
