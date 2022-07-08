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

import concurrent.futures
import logging
import threading
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from homonim import utils
from homonim.enums import ProcCrs
from homonim.raster_pair import RasterPairReader, BlockPair, RasterArray
from rasterio.warp import Resampling
from tabulate import TableFormat, Line, DataRow, tabulate
from tqdm import tqdm

logger = logging.getLogger(__name__)
tabulate.MIN_PADDING = 0

##
# tabulate format for comparison stats
_table_fmt = TableFormat(
    lineabove=Line("", "-", " ", ""),
    linebelowheader=Line("", "-", " ", ""),
    linebetweenrows=None,
    linebelow=Line("", "-", " ", ""),
    headerrow=DataRow("", " ", ""),
    datarow=DataRow("", " ", ""),
    padding=0,
    with_header_hide=["lineabove", "linebelow"]
)  # yapf: disable


class RasterCompare(RasterPairReader):

    def __init__(self, src_filename: Path, ref_filename: Path, proc_crs: ProcCrs = ProcCrs.auto):
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
            :attr:`~homonim.enums.ProcCrs.auto`,  where it will be resolved to lowest resolution of the source and
            reference image CRS's.
        """
        RasterPairReader.__init__(self, src_filename, ref_filename, proc_crs=proc_crs)
        self._lock = threading.Lock()

    schema = dict(
        r2=dict(abbrev='r\N{SUPERSCRIPT TWO}', description='Pearson\'s correlation coefficient squared', ),
        RMSE=dict(abbrev='RMSE', description='Root Mean Square Error', ),
        rRMSE=dict(abbrev='rRMSE', description='Relative RMSE (RMSE/mean(ref))', ),
        N=dict(abbrev='N', description='Number of pixels', )
    )  # yapf: disable
    """ Dictionary describing the statistics returned by :attr:`RasterCompare.compare`. """

    @property
    def schema_table(self) -> str:
        """ A printable table that describes the statistics returned by :attr:`RasterCompare.compare`. """
        headers = {key: key.upper() for key in list(self.schema.values())[0].keys()}
        return tabulate(self.schema.values(), headers=headers, tablefmt=_table_fmt)

    @staticmethod
    def create_config(
        threads: int = 0, max_block_mem: float = 100, downsampling: Resampling = Resampling.average,
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
        max_block_mem: float, optional
            Maximum size of an image block in megabytes. Note that the total memory consumed by a thread is
            proportional to, but a number of times larger than this number.
        downsampling: rasterio.enums.Resampling, optional
            The resampling method to use when downsampling.
        upsampling: rasterio.enums.Resampling, optional
            The resampling method to use when upsampling.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        return dict(
            threads=utils.validate_threads(threads), max_block_mem=max_block_mem, downsampling=downsampling,
            upsampling=upsampling,
        )

    def _get_resampling(self, from_res: Tuple[float, float], to_res: Tuple[float, float], **kwargs):
        """
        Utility method to return the resampling method for re-projecting from resolution `from_res` to resolution
        `to_res`.
        """
        config = self.create_config(**kwargs)
        return config['downsampling'] if np.prod(np.abs(from_res)) <= np.prod(np.abs(to_res)) else config['upsampling']

    def _get_image_stats(self, image_sums: List[Dict]) -> List[Dict]:
        """ Find the image comparison statistics, given src, ref, src**2 etc summations over blocks. """

        def get_band_stats(
            src_sum: float = 0, ref_sum: float = 0, src2_sum: float = 0, ref2_sum: float = 0, src_ref_sum: float = 0,
            res2_sum: float = 0, mask_sum: float = 0
        ) -> List[Dict]:
            # find PCC using the 3rd equation down at
            # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
            # TODO: incorporate stats defs in schema as they were
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
            return dict(r2=pcc ** 2, RMSE=rmse, rRMSE=rrmse, N=int(mask_sum))

        image_stats = []
        sum_over_bands = None
        for band_i, band_sum_dict in enumerate(image_sums):
            band_stats = get_band_stats(**band_sum_dict)
            band_desc = (
                self.ref_im.descriptions[self.ref_bands[band_i] - 1] or
                self.src_im.descriptions[self.src_bands[band_i] - 1] or
                f'Band {band_i + 1}'
            )  # yapf: disable
            image_stats.append(dict(band=band_desc, **band_stats))
            sum_over_bands = (
                dict(**band_stats) if sum_over_bands is None else {
                    k: sum_over_bands[k] + v for k, v in band_stats.items()
                }
            )  # yapf: disable

        mean_stats = {
            k: int(v / len(image_sums)) if isinstance(v, int) else (v / len(image_sums))
            for k, v in sum_over_bands.items()
        }  # yapf: disable
        image_stats.append(dict(band='Mean', **mean_stats))
        return image_stats

    def stats_table(self, stats_list: List[Dict]):
        """
        Create a printable table string of the provided comparison statistics, as returned by
        :meth:`RasterCompare.compare`.

        Parameters
        ----------
        stats_list: list of dict
            Comparison statistics to tabulate.

        Returns
        -------
        str
            A printable table string of the comparison statistics.
        """
        headers = {
            k: self.schema[k]['abbrev'] if k in self.schema else str.capitalize(k)
            for k in list(stats_list[0].keys())
        }  # yapf: disable
        return tabulate(stats_list, headers=headers, floatfmt='.3f', stralign='right', tablefmt=_table_fmt)

    def compare(self, **kwargs) -> List[Dict]:
        """
        Statistically compare source and reference images, displaying and returning results.

        Parameters
        ----------
        kwargs
            Optional configuration settings.  See :meth:`RasterCompare.create_config` for arguments and their default
            values.

        Returns
        -------
        list of dict
            A dictionary representing the comparison results.
        """
        self._assert_open()
        config = self.create_config(**kwargs)
        image_sums = [None] * len(self.src_bands)

        def accum_block(block_pair: BlockPair):
            """ Thread-safe function to accumulate image statistics, given a source-reference block pair.  """
            src_ra, ref_ra = self.read(block_pair)  # read src and ref bands
            # re-project into proc_crs
            if self.proc_crs == ProcCrs.ref:
                resampling = self._get_resampling(src_ra.res, ref_ra.res, **kwargs)
                src_ra = src_ra.reproject(**ref_ra.proj_profile, resampling=resampling)
            else:
                resampling = self._get_resampling(ref_ra.res, src_ra.res, **kwargs)
                ref_ra = ref_ra.reproject(**src_ra.proj_profile, resampling=resampling)

            def get_block_sums(src_ra: RasterArray, ref_ra: RasterArray):
                """ Return block sums of source, source**2, reference, etc. """
                src_array = src_ra.array
                ref_array = ref_ra.array
                mask = ref_ra.mask & src_ra.mask
                src_array[~mask] = 0
                ref_array[~mask] = 0
                return dict(
                    src_sum=src_array.sum(), ref_sum=ref_array.sum(), src2_sum=(src_array ** 2).sum(),
                    ref2_sum=(ref_array ** 2).sum(), src_ref_sum=(src_array * ref_array).sum(),
                    res2_sum=((ref_array - src_array) ** 2).sum(), mask_sum=mask.sum()
                )

            # get the sums for this block
            block_sums_dict = get_block_sums(src_ra, ref_ra)
            with self._lock:
                # accumulate sums for the image
                if image_sums[block_pair.band_i] is None:
                    # initialise
                    image_sums[block_pair.band_i] = block_sums_dict
                else:
                    # add the block sums to the totals for the block's band
                    image_sums[block_pair.band_i] = {
                        k: image_sums[block_pair.band_i][k] + v for k, v in block_sums_dict.items()
                    }  # yapf: disable

        with concurrent.futures.ThreadPoolExecutor(max_workers=config['threads']) as executor:
            futures = [
                executor.submit(accum_block, block_pair)
                for block_pair in self.block_pairs(max_block_mem=config['max_block_mem'])
            ]  # yapf: disable

            # wait for threads, get results and raise any thread generated exceptions
            bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'
            for future in tqdm(concurrent.futures.as_completed(futures), bar_format=bar_format, total=len(futures)):
                future.result()

        return self._get_image_stats(image_sums)
