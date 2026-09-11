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
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from os import PathLike
from typing import Any, ClassVar

import numpy as np
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import Resampling
from tabulate import tabulate
from tqdm.auto import tqdm

from homonim import utils
from homonim.enums import ProcCrs
from homonim.errors import HomonimError
from homonim.matched_pair import MatchedPairReader
from homonim.raster_pair import BlockPair

logger = logging.getLogger(__name__)


class RasterCompare(MatchedPairReader):
    # description of the stats returned by process()
    _schema: ClassVar[dict[str, str]] = dict(
        r2=dict(
            abbrev='r\N{SUPERSCRIPT TWO}',
            description="Pearson's correlation coefficient squared",
        ),
        rmse=dict(abbrev='RMSE', description='Root Mean Square Error'),
        rrmse=dict(abbrev='rRMSE', description='Relative RMSE (RMSE/mean(ref))'),
        n=dict(abbrev='N', description='Number of pixels'),
    )
    # default values for process() kwargs
    _default_config: ClassVar[dict[str, str]] = dict(
        threads=0,
        max_block_mem=512,
        downsampling=Resampling.average,
        upsampling=Resampling.cubic_spline,
    )

    def __init__(
        self,
        src_filename: str | PathLike,
        ref_filename: str | PathLike,
        proc_crs: str | ProcCrs = ProcCrs.auto,
        src_bands: tuple[int, ...] | None = None,
        ref_bands: tuple[int, ...] | None = None,
        force: bool = False,
    ):
        """
        Class to compare source and reference images.

        Reference extents must encompass those of the source.

        The reference should contain bands that are approximate wavelength matches to
        the source bands.  When source and reference bands are RGB, or have
        ``center_wavelength`` tags, bands are matched automatically based on
        wavelength.  Otherwise, source and reference bands are assumed to be in
        matching order.  Subsets and ordering of bands can be specified with the
        ``src_bands`` and ``ref_bands`` parameters.

        :param src_filename:
            Path or URI of a source image.
        :param ref_filename:
            Path or URI of a reference image.
        :param proc_crs:
            Which of the source or reference CRS and pixel grids should be used for
            performing the comparison.  By default, the CRS and pixel grid with the
            lowest resolution is used (recommended).
        :param src_bands:
            Indexes of source bands to be compared (1 based).  Defaults to all bands
            with a ``center_wavelength`` tag if any exist, otherwise to all non-alpha
            bands.
        :param ref_bands:
            Indexes of reference bands to match with source bands (1 based).
            Defaults to all bands with a ``center_wavelength`` tag if any exist,
            otherwise to all non-alpha bands.
        :param force:
            Whether to bypass wavelength band matching, and match source and reference
            bands in their given order.
        """
        super().__init__(
            src_filename,
            ref_filename,
            proc_crs=proc_crs,
            src_bands=src_bands,
            ref_bands=ref_bands,
            force=force,
        )
        self._lock = threading.Lock()

    @staticmethod
    def schema_table() -> str:
        """Return a table string describing the :meth:`RasterCompare.process`
        statistics.
        """
        headers = {
            key: key.upper()
            for key in next(iter(RasterCompare._schema.values())).keys()
        }
        return tabulate(
            RasterCompare._schema.values(), headers=headers, tablefmt=utils.table_format
        )

    @staticmethod
    def create_config(
        threads: int = _default_config['threads'],
        max_block_mem: float = _default_config['max_block_mem'],
        downsampling: Resampling = _default_config['downsampling'],
        upsampling: Resampling = _default_config['upsampling'],
    ) -> dict[str, Any]:
        """
        Return a comparison configuration whose items can be passed as keyword
        arguments to :meth:`~RasterCompare.process`.

        .. deprecated:: 0.5.0

            This method is deprecated and will be removed in a future release.

        :param threads:
            Number of image blocks to process concurrently.  If ``0``, the number of
            CPUs is used.
        :param max_block_mem:
            Maximum size of an image block in megabytes.  If ``0``, a block will
            correspond to a whole image band.
        :param downsampling:
            Resampling method to use when downsampling.
        :param upsampling:
            Resampling method to use when upsampling.

        :return:
            Configuration dictionary.
        """
        warnings.warn(
            'This method is deprecated and will be removed in a future release.',
            category=DeprecationWarning,
            stacklevel=2,
        )
        return dict(
            threads=utils.validate_threads(threads),
            max_block_mem=max_block_mem,
            downsampling=downsampling,
            upsampling=upsampling,
        )

    def _get_resampling(
        self,
        from_res: tuple[float, float],
        to_res: tuple[float, float],
        downsampling: Resampling,
        upsampling: Resampling,
    ) -> Resampling:
        """Return the resampling method for re-projecting from resolution
        ``from_res`` to resolution ``to_res``.
        """
        return (
            downsampling
            if np.prod(np.abs(from_res)) <= np.prod(np.abs(to_res))
            else upsampling
        )

    def _get_image_stats(
        self, image_sums: list[dict[str, int | float]]
    ) -> list[dict[str, int | float]]:
        """Return the image comparison statistics, given a list of source /
        reference etc. band sum dictionaries.
        """

        def get_band_stats(
            src_sum: float = 0,
            ref_sum: float = 0,
            src2_sum: float = 0,
            ref2_sum: float = 0,
            src_ref_sum: float = 0,
            res2_sum: float = 0,
            mask_sum: float = 0,
        ) -> dict:
            """Return the band comparison statistics, given the source / reference etc.
            band sums.
            """
            # find PCC using the 3rd equation down at
            # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
            src_mean = src_sum / mask_sum
            ref_mean = ref_sum / mask_sum
            pcc_num = src_ref_sum - (mask_sum * src_mean * ref_mean)
            pcc_den = np.sqrt(src2_sum - (mask_sum * (src_mean**2))) * np.sqrt(
                ref2_sum - (mask_sum * (ref_mean**2))
            )
            pcc = pcc_num / pcc_den

            # find RMSE and rRMSE
            rmse = np.sqrt(res2_sum / mask_sum).item()
            rrmse = rmse / ref_mean
            return dict(r2=pcc**2, rmse=rmse, rrmse=rrmse, n=int(mask_sum))

        image_stats = {}
        sum_over_bands = {}
        for band_i, band_sum_dict in enumerate(image_sums):
            band_stats = get_band_stats(**band_sum_dict)
            # prefer ref band name as key: with multiple source images compared with
            # one reference, this makes the comparison tables easier to interpret
            band_desc = (
                self.ref_im.descriptions[self.ref_bands[band_i] - 1]
                or self.src_im.descriptions[self.src_bands[band_i] - 1]
                or f'Ref. band {self.ref_bands[band_i]}'
            )
            image_stats[band_desc] = band_stats
            sum_over_bands = {
                k: sum_over_bands.get(k, 0) + v for k, v in band_stats.items()
            }

        # find mean of each statistic over the bands, retaining int types
        mean_stats = {
            k: int(v / len(image_sums)) if isinstance(v, int) else (v / len(image_sums))
            for k, v in sum_over_bands.items()
        }
        # add the means to the list of bands
        image_stats['Mean'] = mean_stats
        return image_stats

    @staticmethod
    def stats_table(
        stats_dict: dict[str, dict[str, float | int]], key_heading: str = 'band'
    ) -> str:
        """
        Tabulate the provided comparison statistics.

        :param stats_dict:
            Comparison statistics, as returned by :meth:`RasterCompare.process`.
        :param key_heading:
            Column heading for the ``stats_dict`` keys.

        :return:
            Table string.
        """
        stats_list = [
            dict(**{key_heading: key}, **val) for key, val in stats_dict.items()
        ]
        headers = {
            k: RasterCompare._schema[k]['abbrev']
            if k in RasterCompare._schema
            else str.capitalize(k)
            for k in stats_list[0].keys()
        }
        return tabulate(
            stats_list,
            headers=headers,
            floatfmt='.3f',
            stralign='right',
            tablefmt=utils.table_format,
        )

    def process(
        self,
        threads: int = _default_config['threads'],
        max_block_mem: float = _default_config['max_block_mem'],
        downsampling: Resampling = _default_config['downsampling'],
        upsampling: Resampling = _default_config['upsampling'],
    ) -> dict[str, dict[str, float | int]]:
        """
        Compare source and reference images.

        :param threads:
            Number of image blocks to process concurrently.  If ``0``, the number of
            CPUs is used.
        :param max_block_mem:
            Maximum size of an image block in megabytes.  If ``0``, a block will
            correspond to a whole image band.
        :param downsampling:
            Resampling method to use when downsampling.
        :param upsampling:
            Resampling method to use when upsampling.

        :return:
            Comparison statistic dictionary.
        """
        self._assert_open()
        if threads > os.cpu_count():
            raise HomonimError(
                "'threads' should be less than or equal to the number of CPUs"
            )
        threads = threads if threads > 0 else os.cpu_count()

        def get_block_sums(block_pair: BlockPair):
            """Thread-safe function to find the source / reference etc. sums for a
            block.
            """
            # read src and ref blocks
            src_ra, ref_ra = self.read(block_pair)

            # re-project so that both source and reference are in proc_crs
            if self.proc_crs == ProcCrs.ref:
                resampling = self._get_resampling(
                    src_ra.res, ref_ra.res, downsampling, upsampling
                )
                src_ra = src_ra.reproject(**ref_ra.proj_profile, resampling=resampling)
            else:
                resampling = self._get_resampling(
                    ref_ra.res, src_ra.res, downsampling, upsampling
                )
                ref_ra = ref_ra.reproject(**src_ra.proj_profile, resampling=resampling)

            # mask invalid pixels so they don't contribute to sums
            src_array = src_ra.array
            ref_array = ref_ra.array
            mask = ref_ra.mask & src_ra.mask
            src_array[~mask] = 0
            ref_array[~mask] = 0

            # find the required sums and return
            sums_dict = dict(
                src_sum=src_array.sum(),
                ref_sum=ref_array.sum(),
                src2_sum=(src_array**2).sum(),
                ref2_sum=(ref_array**2).sum(),
                src_ref_sum=(src_array * ref_array).sum(),
                res2_sum=((ref_array - src_array) ** 2).sum(),
                mask_sum=mask.sum(),
            )
            return sums_dict, block_pair

        with ExitStack() as stack:
            # ignore NotGeoreferencedWarning from RasterArray.reproject()
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter('ignore', category=NotGeoreferencedWarning)

            # read and sum image blocks in threads
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=threads))
            futures = [
                executor.submit(get_block_sums, block_pair)
                for block_pair in self.block_pairs(max_block_mem=max_block_mem)
            ]

            # wait for threads
            image_sums = [{} for _ in self.src_bands]
            bar_format = (
                '{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'
            )
            for future in tqdm(
                as_completed(futures),
                bar_format=bar_format,
                total=len(futures),
                dynamic_ncols=True,
            ):
                # get block sums and accumulate over the image
                block_sums_dict, block_pair = future.result()
                image_sums[block_pair.band_i] = {
                    k: image_sums[block_pair.band_i].get(k, 0) + v
                    for k, v in block_sums_dict.items()
                }

        # return the comparison statistics for the accumulated block sums
        return self._get_image_stats(image_sums)
