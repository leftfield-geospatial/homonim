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
from pathlib import Path
from typing import List, Tuple, Union, Dict
import warnings

import numpy as np
import rasterio as rio
from rasterio.enums import ColorInterp
from tabulate import tabulate
from homonim.raster_pair import RasterPairReader
from homonim.enums import ProcCrs

logger = logging.getLogger(__name__)

class MatchedPairReader(RasterPairReader):
    _max_rel_wavelength_diff = .1   # maximum allowed relative distance between center wavelengths for matching

    def __init__(
        self, src_filename: Union[str, Path], ref_filename: Union[str, Path], proc_crs: ProcCrs = ProcCrs.auto,
        src_bands: Tuple[int, ...] = None, ref_bands: Tuple[int, ...] = None, force: bool = False,
    ):
        """
        Class for reading matching, and optionally overlapping, blocks from a source and reference image pair.

        Reference and source image(s) should be co-located and spectrally similar.  Reference image extents must
        encompass those of the source image.

        The reference image should contain bands that are approximate (wavelength) matches to the source image bands.
        Where source and reference images are RGB, or have ``center_wavelength`` metadata, bands are matched
        automatically based on wavelength.  Where there are the same number of source and reference bands, and no
        ``center_wavelength`` metadata, bands are assumed to be in matching order.  Subsets and ordering of source
        and reference bands can be specified with the ``src_bands`` and ``ref_bands`` parameters.

        .. note::

            Images downloaded with `geedim <https://github.com/dugalh/geedim>`_ have ``center_wavelength`` metadata
            compatible with ``homonim``.

        Parameters
        ----------
        src_filename: str, pathlib.Path
            Path or URL of the source image file.
        ref_filename: str, pathlib.Path
            Path or URL of the reference image file.
        proc_crs: homonim.enums.ProcCrs, optional
            :class:`~homonim.enums.ProcCrs` instance specifying which of the source/reference image CRS and pixel
            grid to use for processing.  For most use cases, it can be left as the default of
            :attr:`~homonim.enums.ProcCrs.auto` i.e. the lowest resolution of the source and reference image CRS's.
        src_bands: list of int, optional.
            Indexes of source spectral bands to be processed (1 based).  If not specified, all bands with the
            ``center_wavelength`` property, or all non-alpha bands, are used.
        ref_bands: list of int, optional.
            Indexes of reference spectral bands to match and fuse with source bands (1 based).  Should contain at
            least as many elements as ``src_bands``, or the number of valid bands in the source image file,
            if ``src_bands`` is not specified.  If ``ref_bands`` is not specified, all reference bands with the
            ``center_wavelength`` property, or all non-alpha bands are used.
        force: bool, optional
            Bypass auto wavelength matching, and any band-matching errors.  Use with caution.
        """
        self._src_bands = src_bands
        self._ref_bands = ref_bands
        self._force = force
        RasterPairReader.__init__(self, src_filename, ref_filename, proc_crs=proc_crs)

    @property
    def src_bands(self) -> Tuple[int, ...]:
        """ Matched source band indices (1-based). """
        return self._src_bands

    @property
    def ref_bands(self) ->Tuple[int, ...]:
        """ Matched reference band indices (1-based). """
        return self._ref_bands

    @staticmethod
    def _get_band_info(
        im: rio.DatasetReader, bands: Tuple[int, ...] = None, name: str = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Return band indices, corresponding names, and center wavelengths (if any). """
        name = name or Path(im.name).name or ''
        non_alpha_bands = np.array([
            i + 1 for i in range(im.count)
            if (
                (im.colorinterp[i] != ColorInterp.alpha) and not
                (im.descriptions[i] and (im.descriptions[i].endswith('_MASK') or im.descriptions[i].endswith('_DIST')))
            )   # if not alpha band or geedim mask band
        ])  # yapf: disable
        refl_bands = np.array([bi for bi in non_alpha_bands if 'center_wavelength' in im.tags(bi)])

        # test bands are valid
        if (bands is not None) and (not set(bands).issubset(range(1, im.count + 1))):
            invalid_bands = list(set(bands).difference(range(1, im.count + 1)))
            raise ValueError(f'User specified {name} bands contain invalid band(s) {invalid_bands}.')
        # test bands do not contain alpha bands
        if (bands is not None) and (not set(bands).issubset(non_alpha_bands)):
            alpha_bands = list(set(bands).difference(non_alpha_bands))
            raise ValueError(f'User specified {name} bands contain alpha band(s) {alpha_bands}.')
        # warn if some but not all bands have center wavelength metadata
        if (bands is not None) and len(refl_bands) and (not set(bands).issubset(refl_bands)):
            non_refl_bands = list(set(bands).difference(refl_bands))
            logger.warning(f'User specified {name} bands contain non-reflectance band index(es) {non_refl_bands}.')

        log_prefix = f'Using {name} bands '
        if (bands is not None) and (len(bands) > 0):
            # use bands if it was specified
            bands = np.array(bands)
        elif len(refl_bands) > 0:
            # else use bands with center wavelengths if any
            log_prefix = f'Using {name} reflectance bands '
            bands = np.array(refl_bands)
        elif len(non_alpha_bands) > 0:
            # else use non-alpha bands
            log_prefix = f'Using {name} non-alpha bands '
            bands = np.array(non_alpha_bands)
        else:
            raise ValueError(f'There are no non-alpha/reflectance {name} bands to use.')

        # center wavelengths for all bands
        center_wavelengths = np.array([
            float(im.tags(bi)['center_wavelength'])
            if 'center_wavelength' in im.tags(bi) else np.nan
            for bi in range(1, im.count + 1)
        ])  # yapf: disables

        # assign standard RGB center wavelengths
        if (len(non_alpha_bands) == 3):
            std_rgb_cws = dict(zip(
                [ColorInterp.red, ColorInterp.green, ColorInterp.blue], [.650, .560, .480]
            ))  # yapf: disable
            log_list = []
            # assign standard RGB center wavelengths according to valid colorinterp's
            for bi in non_alpha_bands:
                if not np.isnan(center_wavelengths[bi - 1]):
                    continue    # don't overwrite existing center wavelengths
                if im.colorinterp[bi - 1] in std_rgb_cws:
                    center_wavelengths[bi - 1] = std_rgb_cws[im.colorinterp[bi - 1]]
                    log_list.append(im.colorinterp[bi - 1].name)
            if len(log_list) > 0:
                logger.warning(
                    f'Assigning standard {", ".join(log_list)} center wavelengths for {name}.'
                )

            # if the image has no valid RGB colorinterp's or center wavelengths
            if sum(np.isnan(center_wavelengths[non_alpha_bands - 1])) == 3:
                # assume RGB and assign center wavelengths in that order
                logger.warning(
                    f'Assuming image is RGB, and assigning standard center wavelengths for {name}.'
                )
                center_wavelengths[non_alpha_bands - 1] = list(std_rgb_cws.values())

        center_wavelengths = center_wavelengths[bands - 1]
        band_names = np.array([im.descriptions[bi - 1] if im.descriptions[bi - 1] else str(bi) for bi in bands])
        logger.debug(f'{log_prefix} {list(band_names)}.')
        return bands, band_names, center_wavelengths

    def _get_pair_band_table(
        self, src_im: rio.DatasetReader, ref_im: rio.DatasetReader, src_bands: Tuple[int, ...] = None,
        ref_bands: Tuple[int, ...] = None
    ):
        """ Return a table of source and reference metadata for the given bands. """
        num_bands = min(len(src_bands), len(ref_bands))
        src_bands = src_bands[:num_bands]
        ref_bands = ref_bands[:num_bands]

        # retrieve src/ref image band metadata as lists of dicts
        src_band_list = []
        ref_band_list = []
        band_keys = {'name': 'Name', 'description': 'Description', 'center_wavelength': 'Wavelen'}
        for band_list, im, bands in zip([src_band_list, ref_band_list], [src_im, ref_im], [src_bands, ref_bands]):
            for band in bands:
                band_dict = {}
                if 'name' not in im.tags(band):
                    band_dict.update(Name=im.descriptions[band-1] or str(band))
                band_tags = im.tags(band)
                band_dict.update(**{bkn: band_tags[bk] for bk, bkn in band_keys.items() if bk in band_tags})
                band_list.append(band_dict)

        # combine src and ref lists of dicts into one
        def prefix_dict_keys(band_dict: Dict, prefix: str):
            return {f'{prefix}\n{k}':v for k, v in band_dict.items()}

        band_list = []
        super_keys = dict()
        for src_band_dict, ref_band_dict in zip(src_band_list, ref_band_list):
            band_dict = {}
            band_dict.update(prefix_dict_keys(src_band_dict, 'Source'))
            band_dict.update(prefix_dict_keys(ref_band_dict, 'Ref'))
            band_list.append(band_dict)
            super_keys.update(dict.fromkeys(band_dict.keys()))

        # update missing keys with empty string vals
        for band_dict in band_list:
            update_keys = [k for k in super_keys if k not in band_dict.keys()]
            update_vals = [''] * len(update_keys)
            band_dict.update(zip(update_keys, update_vals))

        return tabulate(band_list, headers='keys',  maxcolwidths=20, tablefmt='simple')

    def _match_pair_bands(
        self, src_im: rio.DatasetReader, ref_im: rio.DatasetReader
    ) -> Tuple[Tuple[int], Tuple[int]]:  # yapf: disable
        """
        Validate and match source and reference bands. An override of base class RasterPairReader._match_pair_bands().
        """
        # retrieve non-alpha bands
        src_bands, src_band_names, src_wavelengths = MatchedPairReader._get_band_info(src_im, bands=self._src_bands)
        ref_bands, ref_band_names, ref_wavelengths = MatchedPairReader._get_band_info(ref_im, bands=self._ref_bands)
        src_name = Path(src_im.name).name
        ref_name = Path(ref_im.name).name

        if len(src_bands) > len(ref_bands):
            if not self._force:
                raise ValueError(f'{ref_name} has fewer bands than {src_name}.')
            else:
                logger.warning(f'{ref_name} has fewer bands than {src_name}.')

        match_bands = np.array([np.nan] * len(src_bands))
        # match src with ref bands based on center wavelength metadata, bypassing if force==True
        if any(src_wavelengths) and any(ref_wavelengths) and not self._force:
            # TODO: consider using a linear programming type optimisation here,
            #  e.g. https://stackoverflow.com/questions/67368093/find-optimal-unique-neighbour-pairs-based-on-closest-distance

            # absolute & relative distance matrix between src and ref center wavelengths
            abs_dist = np.abs(src_wavelengths[:, np.newaxis] - ref_wavelengths[np.newaxis, :])
            rel_dist = abs_dist / src_wavelengths[:, np.newaxis]
            def greedy_match(dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                """
                Greedy matching of src to ref bands based on the provided center wavelength distance matrix,
                `dist`. src bands must be down the rows, and ref bands along the cols of `dist`.
                Will match one ref band for each src band until either all src or all ref bands
                have been matched.  Works for all cases where len(src_bands) != len(ref_bands).
                """
                # match_idx[i] is the index of the ref band that matches with the ith src band
                match_idx = np.array([np.nan] * dist.shape[0])
                match_dist = np.array([np.nan] * dist.shape[0]) # distances corresponding to the above matches

                # suppress runtime warning on all-Nan slice, as it is expected in normal operation
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    # repeat until all src or ref bands have been matched
                    while not all(np.isnan(np.nanmin(dist, axis=1))) or not all(np.isnan(np.nanmin(dist, axis=0))):
                        # find the row with the smallest distance in it
                        min_dist = np.nanmin(dist, axis=1)
                        min_dist_row_idx = np.nanargmin(min_dist)
                        min_dist_row = dist[min_dist_row_idx, :]
                        # store match idx and distance for this row
                        match_idx[min_dist_row_idx] = np.nanargmin(min_dist_row)
                        match_dist[min_dist_row_idx] = min_dist[min_dist_row_idx]
                        # set the matched row and col to nan, so that it is not used again
                        dist[:, int(match_idx[min_dist_row_idx])] = np.nan
                        dist[min_dist_row_idx, :] = np.nan

                return match_dist, match_idx

            match_dist, match_idx = greedy_match(rel_dist)

            # if any of the matched distances are greater than a threshold, raise an informative error,
            # or log a warning, depending on `self._force`
            if any(match_dist > MatchedPairReader._max_rel_wavelength_diff):
                err_idx = match_dist > MatchedPairReader._max_rel_wavelength_diff
                src_err_band_names = list(src_band_names[err_idx])
                ref_err_band_names = list(ref_band_names[np.int64(match_idx[err_idx])])
                err_dists = list(match_dist[err_idx].round(3))
                raise ValueError(
                    f'{src_name} band(s) {src_err_band_names} could not be auto-matched.  The nearest '
                    f'{ref_name} band(s) were {ref_err_band_names}, at center wavelength difference(s) of '
                    f'{err_dists} (um) respectively.'
                )

            src_matched = ~np.isnan(match_idx)
            match_bands[src_matched] = ref_bands[np.int64(match_idx[src_matched])]
            logger.debug(
                f'Matching {src_name} band(s) {list(src_band_names[src_matched])} with {ref_name} band(s) '
                f'{list(ref_band_names[np.int64(match_idx[src_matched])])}, at center wavelength difference(s) of '
                f'{list(match_dist[src_matched].round(3))} (um) respectively.'
            )

        # match any remaining bands that don't have center wavelength metadata
        if sum(~np.isnan(match_bands)) < min(len(src_bands), len(ref_bands)):
            unmatched = np.isnan(match_bands)
            unmatch_ref_idx = np.array([i for i, bi in enumerate(ref_bands) if bi not in match_bands], dtype=int)
            unmatch_ref_bands = ref_bands[unmatch_ref_idx]
            unmatch_ref_band_names = ref_band_names[unmatch_ref_idx]
            if len(src_bands) == len(ref_bands):
                # assume unmatched src and ref image bands are in matching order
                match_bands[unmatched] = unmatch_ref_bands
                logger.debug(
                    f'Matching {src_name} band(s) {list(src_band_names[unmatched])} in file order with'
                    f' {ref_name} band(s) {list(unmatch_ref_band_names)}.'
                )
            elif self._force:
                # match the remaining N src bands with the first N unmatched ref bands (N=sum(unmatched))
                unmatch_ref_bands = unmatch_ref_bands[:sum(unmatched)]
                unmatch_ref_band_names = unmatch_ref_band_names[:sum(unmatched)]
                # if there are not N unmatched ref bands, truncate unmatched to just match what we can
                unmatched = np.where(unmatched)[0][:len(unmatch_ref_bands)]
                match_bands[unmatched] = unmatch_ref_bands
                logger.warning(
                    f'Matching {src_name} band(s) {list(src_bands[unmatched])} in file order with {ref_name} '
                    f'band(s): {list(unmatch_ref_band_names)}.'
                )
            else:
                # raise an error when remaining unmatched bands counts do not match and `self._force` is False
                raise ValueError(
                    f'Could not match {src_name} band(s) {list(src_bands[unmatched])} with {ref_name} '
                    f'band(s) {list(unmatch_ref_band_names)}.  Ensure {src_name} and {ref_name} non-alpha band '
                    f'counts match, {src_name} and {ref_name} have ``center_wavelength`` tags for each band, '
                    f'or set `force` to True.'
                )
        # Truncate src and ref to reflect the matched bands
        src_matched = ~np.isnan(match_bands)
        src_bands = src_bands[src_matched]
        ref_matched = np.array([list(ref_bands).index(bi) for bi in match_bands[src_matched]])
        ref_bands = ref_bands[ref_matched]
        logger.debug('Matched band(s):\n\n' + self._get_pair_band_table(src_im, ref_im, src_bands, ref_bands) + '\n')
        return tuple(src_bands.tolist()), tuple(ref_bands.tolist())
