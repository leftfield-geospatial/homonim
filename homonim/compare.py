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

import concurrent.futures
import logging
import pathlib
import threading
from typing import Dict, List, Tuple

import numpy as np
from rasterio.warp import Resampling
from tabulate import tabulate
from tqdm import tqdm

from homonim import utils
from homonim.enums import ProcCrs
from homonim.raster_pair import RasterPairReader, BlockPair

logger = logging.getLogger(__name__)


class RasterCompare(RasterPairReader):

    def __init__(self, src_filename: pathlib.Path, ref_filename: pathlib.Path, proc_crs: ProcCrs = ProcCrs.auto):
        """
        Class to statistically compare source and reference images.

        Parameters
        ----------
        src_filename: str, pathlib.Path
            Path to the source image file.  Can be any raw, corrected, etc. multi-spectral image.
        ref_filename: str, pathlib.Path
            Path to the reference image file.  The extents of this image should cover the source with at least a 2
            pixel border.  The reference image should have at least as many bands as the source, and the
            ordering of the source and reference bands should match.
        proc_crs: homonim.enums.ProcCrs, optional
            :class:`~homonim.enums.ProcCrs` instance specifying which of the source/reference image spaces will be
            used for comparison.  In most cases, it can be left as the default of
            :attr:`~homonim.enums.ProcCrs.auto`,  where it will be resolved to the lowest resolution of the source and
            reference image CRS's.
        """
        RasterPairReader.__init__(self, src_filename, ref_filename, proc_crs=proc_crs)
        self._lock = threading.Lock()

    schema = dict(
        r2=dict(abbrev='r\N{SUPERSCRIPT TWO}', description='Pearson\'s correlation coefficient squared', ),
        rmse=dict(abbrev='RMSE', description='Root Mean Square Error', ),
        rrmse=dict(abbrev='rRMSE', description='Relative RMSE (RMSE/mean(ref))', ),
        n=dict(abbrev='N', description='Number of pixels', )
    )  # yapf: disable
    """ Dictionary describing the statistics returned by :attr:`RasterCompare.compare`. """

    @property
    def schema_table(self) -> str:
        """ Table string describing statistics returned by :attr:`RasterCompare.compare`. """
        headers = {key: key.upper() for key in list(self.schema.values())[0].keys()}
        return tabulate(self.schema.values(), headers=headers, tablefmt=utils.table_format)

    @staticmethod
    def create_config(
        threads: int = 0, max_block_mem: float = 512, downsampling: Resampling = Resampling.average,
        upsampling: Resampling = Resampling.cubic_spline,
    ) -> Dict:
        """
        Utility method to create a RasterCompare configuration dictionary that can be passed to
        :meth:`RasterCompare.process`.  Without arguments, the default configuration values are returned.

        Parameters
        ----------
        threads: int, optional
            Number of image blocks to process concurrently.  A maximum of the number of processors on your
            system is allowed.  Increasing this number will increase the memory required for processing.
            0 = use all processors.
        max_block_mem: float, optional
            Maximum size of an image block in megabytes. Note that the total memory consumed by a thread is
            proportional to, but a number of times larger than this number.
        downsampling: rasterio.enums.Resampling, optional
            Resampling method to use when downsampling. See the `rasterio docs
            <https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_ for
            available options.
        upsampling: rasterio.enums.Resampling, optional
            Resampling method to use when upsampling.  See the `rasterio docs
            <https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_ for
            available options.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        return dict(
            threads=utils.validate_threads(threads), max_block_mem=max_block_mem, downsampling=downsampling,
            upsampling=upsampling,
        )

    def _get_resampling(self, from_res: Tuple[float, float], to_res: Tuple[float, float], **kwargs) -> Resampling:
        """
        Utility method to return the resampling method for re-projecting from resolution `from_res` to resolution
        `to_res`.
        """
        config = self.create_config(**kwargs)
        return config['downsampling'] if np.prod(np.abs(from_res)) <= np.prod(np.abs(to_res)) else config['upsampling']

    def _get_image_stats(self, image_sums: List[Dict]) -> List[Dict]:
        """ Return the band comparison statistics, given src, ref, src**2 etc band sums. """

        def get_band_stats(
            src_sum: float = 0, ref_sum: float = 0, src2_sum: float = 0, ref2_sum: float = 0, src_ref_sum: float = 0,
            res2_sum: float = 0, mask_sum: float = 0
        ) -> Dict:
            """ Return the comparison statistics for a band, given the source, reference etc. sums. """
            # find PCC using the 3rd equation down at
            # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
            src_mean = src_sum / mask_sum
            ref_mean = ref_sum / mask_sum
            pcc_num = src_ref_sum - (mask_sum * src_mean * ref_mean)
            pcc_den = (
                np.sqrt(src2_sum - (mask_sum * (src_mean ** 2))) * np.sqrt(ref2_sum - (mask_sum * (ref_mean ** 2)))
            )
            pcc = pcc_num / pcc_den

            # find RMSE and rRMSE
            rmse = np.sqrt(res2_sum / mask_sum)
            rrmse = rmse / ref_mean
            return dict(r2=pcc ** 2, rmse=rmse, rrmse=rrmse, n=int(mask_sum))

        image_stats = []
        sum_over_bands = {}
        for band_i, band_sum_dict in enumerate(image_sums):
            band_stats = get_band_stats(**band_sum_dict)
            band_desc = (
                self.ref_im.descriptions[self.ref_bands[band_i] - 1] or
                self.src_im.descriptions[self.src_bands[band_i] - 1] or
                f'Band {band_i + 1}'
            )  # yapf: disable
            image_stats.append(dict(band=band_desc, **band_stats))
            sum_over_bands = {k: sum_over_bands.get(k, 0) + v for k, v in band_stats.items()}

        # find mean of each statistic over the bands, retaining int types
        mean_stats = {
            k: int(v / len(image_sums)) if isinstance(v, int) else (v / len(image_sums))
            for k, v in sum_over_bands.items()
        }  # yapf: disable
        # add the means to the list of bands
        image_stats.append(dict(band='Mean', **mean_stats))
        return image_stats

    def stats_table(self, stats_list: List[Dict]):
        """
        Create a table string from the provided comparison statistics.

        Parameters
        ----------
        stats_list: list of dict
            Comparison statistics to tabulate, as returned by :meth:`RasterCompare.compare`.

        Returns
        -------
        str
            Table string.
        """
        headers = {
            k: self.schema[k]['abbrev'] if k in self.schema else str.capitalize(k)
            for k in list(stats_list[0].keys())
        }  # yapf: disable
        return tabulate(stats_list, headers=headers, floatfmt='.3f', stralign='right', tablefmt=utils.table_format)

    def compare(self, **kwargs) -> List[Dict]:
        """
        Statistically compare source and reference images.

        To improve speed and reduce memory usage, images are divided into blocks for concurrent processing.

        Parameters
        ----------
        kwargs
            Optional configuration settings.  See :meth:`RasterCompare.create_config` for possible arguments and their
            default values.

        Returns
        -------
        list of dict
            List of dicts for each band, representing the comparison results.
        """
        self._assert_open()
        config = self.create_config(**kwargs)

        def get_block_sums(block_pair: BlockPair):
            """ Thread-safe function to find the source/reference sums for a block.  """
            src_ra, ref_ra = self.read(block_pair)  # read src and ref blocks

            # re-project so that both source and reference are in proc_crs
            if self.proc_crs == ProcCrs.ref:
                resampling = self._get_resampling(src_ra.res, ref_ra.res, **kwargs)
                src_ra = src_ra.reproject(**ref_ra.proj_profile, resampling=resampling)
            else:
                resampling = self._get_resampling(ref_ra.res, src_ra.res, **kwargs)
                ref_ra = ref_ra.reproject(**src_ra.proj_profile, resampling=resampling)

            # mask invalid pixels so they don't contribute to sums
            src_array = src_ra.array
            ref_array = ref_ra.array
            mask = ref_ra.mask & src_ra.mask
            src_array[~mask] = 0
            ref_array[~mask] = 0
            # find the required sums and return
            sums_dict = dict(
                src_sum=src_array.sum(), ref_sum=ref_array.sum(), src2_sum=(src_array ** 2).sum(),
                ref2_sum=(ref_array ** 2).sum(), src_ref_sum=(src_array * ref_array).sum(),
                res2_sum=((ref_array - src_array) ** 2).sum(), mask_sum=mask.sum()
            )
            return sums_dict, block_pair

        with concurrent.futures.ThreadPoolExecutor(max_workers=config['threads']) as executor:
            # read and sum image blocks in threads
            futures = [
                executor.submit(get_block_sums, block_pair)
                for block_pair in self.block_pairs(max_block_mem=config['max_block_mem'])
            ]  # yapf: disable

            # wait for threads
            image_sums = [{} for _ in self.src_bands]
            bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'
            for future in tqdm(
                concurrent.futures.as_completed(futures), bar_format=bar_format, total=len(futures), dynamic_ncols=True,
            ):  # yapf: disable
                # get block sums and accumulate over the image
                block_sums_dict, block_pair = future.result()
                image_sums[block_pair.band_i] = (
                    {k: image_sums[block_pair.band_i].get(k, 0) + v for k, v in block_sums_dict.items()}
                )  # yapf: disable

        # find the comparison statistics from the accumulated block sums, and return
        return self._get_image_stats(image_sums)
