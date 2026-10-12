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
import warnings
from os import PathLike
from pathlib import Path

import numpy as np
from rasterio.enums import ColorInterp
from rasterio.io import DatasetReader
from tabulate import tabulate

from homonim import utils
from homonim.enums import ProcCrs
from homonim.errors import HomonimError, HomonimWarning
from homonim.raster_pair import RasterPairReader

logger = logging.getLogger(__name__)


class MatchedPairReader(RasterPairReader):
    # maximum allowed relative distance between center wavelengths for matching
    _max_rel_wavelength_diff = 0.1

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
        Base class for processing matching blocks from a source and reference image
        pair.

        Reference extents must encompass those of the source.

        Source bands should be matched with reference bands of a similar wavelength.
        When source and reference bands are RGB, or have ``center_wavelength`` tags,
        bands are matched automatically.  Otherwise, source and reference bands are
        assumed to be in matching order.  Subsets and ordering of bands can be
        specified with the ``src_bands`` and ``ref_bands`` parameters.

        Can be used as a context manager to automatically close source and reference
        images on exiting.

        :param src_filename:
            Path or URI of a source image.
        :param ref_filename:
            Path or URI of a reference image.
        :param proc_crs:
            Which of the source or reference pixel grids to use for processing.  By
            default, the grid with the lowest resolution is used (recommended).
        :param src_bands:
            Indexes of source bands to be corrected (1 based).  Defaults to all bands
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
        self._src_bands = src_bands
        self._ref_bands = ref_bands
        self._force = force
        super().__init__(src_filename, ref_filename, proc_crs=proc_crs)
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug('Matched bands:\n\n' + self.matched_band_table() + '\n')

    @staticmethod
    def _get_band_info(
        im: DatasetReader,
        bands: tuple[int, ...] | None = None,
        param_name: str = 'bands',
    ) -> tuple[np.ndarray[int], np.ndarray[str], np.ndarray[float]]:
        """Return reflectance / non-alpha band indices, corresponding names,
        and center wavelengths for the given dataset, if bands is not provided.
        Otherwise validate the given bands and return them with their corresponding
        names and center wavelengths.
        """
        im_name = Path(im.name).name
        nonalpha_bands = utils.get_nonalpha_bands(im)
        refl_bands = [bi for bi in nonalpha_bands if 'center_wavelength' in im.tags(bi)]

        # form a list of center wavelengths for each band, using nan where no
        # center wavelength can be found

        # extract center wavelengths from any geedim tags
        center_wavelengths = [
            float(im.tags(bi).get('center_wavelength', 'nan'))
            for bi in range(1, im.count + 1)
        ]

        def no_center_wavelengths() -> bool:
            """Return whether no non-alpha bands have center wavelengths."""
            return np.isnan([center_wavelengths[bi - 1] for bi in nonalpha_bands]).all()

        # assign standard RGB wavelengths if there are 3 non-alpha bands with no
        # geedim tags
        if no_center_wavelengths() and len(nonalpha_bands) == 3:
            warnings.warn(
                f"Using standard RGB center wavelengths for '{im_name}'.",
                category=HomonimWarning,
                stacklevel=2,
            )
            # use the dataset colorinterp property to differentiate R/G/B
            std_rgb_cws = dict(
                zip(
                    [ColorInterp.red, ColorInterp.green, ColorInterp.blue],
                    [0.650, 0.560, 0.480],
                    strict=True,
                )
            )
            for bi in nonalpha_bands:
                center_wavelengths[bi - 1] = std_rgb_cws.get(
                    im.colorinterp[bi - 1], center_wavelengths[bi - 1]
                )

            if no_center_wavelengths():
                # if colorinterp was not R/G/B, assume bands are in RGB order
                for i, cw in enumerate(std_rgb_cws.values()):
                    center_wavelengths[nonalpha_bands[i] - 1] = cw

        if bands:
            # validate the provided bands
            invalid_bands = tuple(set(bands).difference(range(1, im.count + 1)))
            if invalid_bands:
                raise HomonimError(
                    f"'{param_name}' for '{im_name}' contains out of range indexes "
                    f'{invalid_bands}.'
                )
            alpha_bands = tuple(set(bands).difference(nonalpha_bands))
            if alpha_bands:
                raise HomonimError(
                    f"'{param_name}' for '{im_name}' contains alpha band indexes "
                    f'{alpha_bands}.'
                )
            nonrefl_bands = tuple(set(bands).difference(refl_bands))
            if refl_bands and nonrefl_bands:
                warnings.warn(
                    f"'{param_name}' for '{im_name}' contains non-reflectance "
                    f'indexes {nonrefl_bands}.',
                    category=HomonimWarning,
                    stacklevel=2,
                )
        else:
            # default bands to center wavelength ('reflectance') bands if any exist,
            # otherwise to non-alpha bands
            bands = refl_bands or nonalpha_bands
            if not bands:
                raise HomonimError(f"'{im_name}' has no non-alpha bands.")

        names = [im.descriptions[bi - 1] or str(bi) for bi in bands]
        # center wavelengths corresponding to bands
        center_wavelengths = [center_wavelengths[bi - 1] for bi in bands]

        return np.array(bands), np.array(names), np.array(center_wavelengths)

    @staticmethod
    def _greedy_match(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Greedy match of reference to source bands based on the provided wavelength
        distance matrix.  Source bands must be down the rows, and reference bands
        along the cols.

        Matches one reference band for each source band until either all source or
        all reference bands have been matched.  Works for all cases where
        len(src_bands) != len(ref_bands).

        Returns the matched reference band indexes and wavelength distances for each
        source band.
        """
        # match_idx[i] is the index of the reference band that matches with the ith
        # source band
        match_idx = np.array([np.nan] * dist.shape[0])
        # distances corresponding to the matches in match_idx
        match_dist = np.array([np.nan] * dist.shape[0])

        # use masked array rather than nan pass-through to avoid all-nan slice warnings
        dist = np.ma.array(dist, mask=np.isnan(dist))

        # repeat until all source or reference bands have been matched
        while not dist.mask.all():
            # find the row with the smallest distance in it
            min_dist = dist.min(axis=1)
            min_dist_row_idx = np.ma.argmin(min_dist)
            min_dist_row = dist[min_dist_row_idx, :]
            # store match idx and distance for this row
            match_idx[min_dist_row_idx] = np.ma.argmin(min_dist_row)
            match_dist[min_dist_row_idx] = min_dist[min_dist_row_idx]
            # mask the matched row and col, so that it is not used again
            dist[:, int(match_idx[min_dist_row_idx])] = np.ma.masked
            dist[min_dist_row_idx, :] = np.ma.masked

        return match_idx, match_dist

    def _match_bands(
        self, src_im: DatasetReader, ref_im: DatasetReader
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return matched source and reference band indices for the given datasets."""
        # get source & reference bands, names and center wavelengths
        src_bands, src_names, src_wavelengths = MatchedPairReader._get_band_info(
            src_im, bands=self._src_bands, param_name='src_bands'
        )
        ref_bands, ref_names, ref_wavelengths = MatchedPairReader._get_band_info(
            ref_im, bands=self._ref_bands, param_name='ref_bands'
        )
        src_name = Path(src_im.name).name
        ref_name = Path(ref_im.name).name

        if self._force:
            # match bands in their given order
            min_num_bands = min(len(src_bands), len(ref_bands))
            src_bands = src_bands[:min_num_bands]
            ref_bands = ref_bands[:min_num_bands]
            return tuple(src_bands.tolist()), tuple(ref_bands.tolist())

        if len(src_bands) > len(ref_bands):
            raise HomonimError('There are fewer reference than source bands.')

        # matching ref_bands indexes for each source band in src_bands
        match_idx = np.array([np.nan] * len(src_bands))

        # match source with reference bands using center wavelengths, if any
        if any(~np.isnan(src_wavelengths)) and any(~np.isnan(ref_wavelengths)):
            # match based on the relative distance between source and reference
            # wavelengths
            abs_dist = np.abs(
                src_wavelengths[:, np.newaxis] - ref_wavelengths[np.newaxis, :]
            )
            rel_dist = abs_dist / src_wavelengths[:, np.newaxis]
            match_idx, match_dist = self._greedy_match(rel_dist)

            # if any of the matched distances are greater than a threshold, raise an
            # informative error
            err_mask = match_dist > self._max_rel_wavelength_diff
            if any(err_mask):
                src_err_names = tuple(src_names[err_mask].tolist())
                ref_err_names = tuple(
                    ref_names[match_idx[err_mask].astype('int')].tolist()
                )
                err_dists = tuple(match_dist[err_mask].round(3).tolist())
                raise HomonimError(
                    f"Source '{src_name}' bands {src_err_names} could not be "
                    f"wavelength matched.  The nearest reference '{ref_name}' bands "
                    f'were {ref_err_names}, with corresponding relative wavelength '
                    f'differences of {err_dists}.'
                )

        # match any remaining bands without center wavelengths
        src_unmatch_mask = np.isnan(match_idx)
        if sum(~src_unmatch_mask) < len(match_idx):
            ref_unmatch_idx = set(range(len(ref_bands))).difference(match_idx)
            ref_unmatch_idx = sorted(ref_unmatch_idx)
            if len(src_bands) == len(ref_bands):
                # assume unmatched source and reference image bands are in matching
                # order
                match_idx[src_unmatch_mask] = ref_unmatch_idx
            else:
                # raise an error when the remaining unmatched source / reference band
                # counts are unequal
                src_unmatch_names = tuple(src_names[src_unmatch_mask].tolist())
                ref_unmatch_names = tuple(ref_names[ref_unmatch_idx].tolist())
                raise HomonimError(
                    f"Could not match source '{src_name}' bands {src_unmatch_names}"
                    f" with reference '{ref_name}' bands {ref_unmatch_names}.  Ensure "
                    f'source and reference band counts match, or that there are center '
                    f"wavelength tags for each band.  Alternatively, set 'force' to "
                    f'True.'
                )

        ref_bands = ref_bands[match_idx.astype('int')]
        return tuple(src_bands.tolist()), tuple(ref_bands.tolist())

    def matched_band_table(self):
        """Return a table string with each row describing a source and matched
        reference band.
        """
        # TODO: can standard RGB wavelengths be included when used?  center
        #  wavelengths would need to become members / properties
        num_bands = min(len(self._src_bands), len(self._ref_bands))
        src_bands = self._src_bands[:num_bands]
        ref_bands = self._ref_bands[:num_bands]

        def get_bands_list(im: DatasetReader, bands: tuple[int, ...]) -> list[dict]:
            """Return a list of band metadata dictionaries."""
            tag_keys = ['name', 'description', 'center_wavelength']
            bands_list = []
            for bi in bands:
                tag_dict = im.tags(bi)
                band_dict = dict(name=im.descriptions[bi - 1] or str(bi))
                band_dict.update(**{k: tag_dict[k] for k in tag_keys if k in tag_dict})
                bands_list.append(band_dict)
            return bands_list

        # get lists of source and reference band metadata
        src_bands_list = get_bands_list(self._src_im, src_bands)
        ref_bands_list = get_bands_list(self._ref_im, ref_bands)

        # zip the lists together, changing tag keys to column headings
        bands_list = []
        heading_map = {
            'name': 'Name',
            'description': 'Description',
            'center_wavelength': 'Wavelen',
        }

        def convert_dict_keys(band_dict: dict, prefix: str) -> dict:
            return {f'{prefix}{heading_map[k]}': v for k, v in band_dict.items()}

        for src_band_dict, ref_band_dict in zip(
            src_bands_list, ref_bands_list, strict=True
        ):
            src_band_dict = convert_dict_keys(src_band_dict, 'Source\n')
            ref_band_dict = convert_dict_keys(ref_band_dict, 'Ref\n')
            bands_list.append(dict(**src_band_dict, **ref_band_dict))

        return tabulate(
            bands_list,
            headers='keys',
            floatfmt='.3f',
            maxcolwidths=20,
            tablefmt='simple',
        )
