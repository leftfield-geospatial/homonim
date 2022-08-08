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
from typing import List, Tuple
from rasterio.enums import ColorInterp
import numpy as np
import rasterio as rio

logger = logging.getLogger(__name__)


class ReflBands():
    _max_wavelength_diff = .150
    def __init__(self, im: rio.DatasetReader, name: str = None, bands: List[int] = None):

        # Note: rasterio / gdal uses 1-based indices, and that ReflBands.bands property sticks to that convention
        # Python/numpy uses 0-based indices of course, so there is (sometimes implied) conversion between these
        # conventions in this class.  I am using the convention of `i` for 0-based python band index, and `bi` for
        # 1-based gdal index

        name = name or im.name or ''
        non_alpha_bands = np.array([i + 1 for i in range(im.count) if im.colorinterp[i] != ColorInterp.alpha])
        refl_bands = np.array([bi for bi in non_alpha_bands if 'center_wavelength' in im.tags(bi)])
        # test bands do not contain alpha bands
        if (bands is not None) and (not set(bands).issubset(non_alpha_bands)):
            alpha_bands = list(set(bands).difference(non_alpha_bands))
            raise ValueError(
                f'User specified {name} bands contain alpha band(s) {alpha_bands}.'
            )
        # warn if some but not all bands have center wavelength metadata
        if (bands is not None) and len(refl_bands) and (not set(bands).issubset(refl_bands)):
            non_refl_bands = list(set(bands).difference(refl_bands))
            logger.warning(
                f'User specified {name} bands contain non-reflectance band(s) {non_refl_bands}.'
            )

        if bands is not None:
            # use bands if it was specified
            bands = np.array(bands)
        elif len(refl_bands) > 0:
            # else use bands with center wavelengths if any
            # TODO: remove these logs, or log band names
            logger.debug(f'Using {name} reflectance bands {list(refl_bands)}.')
            bands = np.array(refl_bands)
        elif len(non_alpha_bands) > 0:
            # else use non-alpha bands
            logger.debug(f'Using {name} non-alpha bands {list(non_alpha_bands)}.')
            bands = np.array(non_alpha_bands)
        else:
            raise ValueError(f'There are no non-alpha/reflectance {name} bands to use.')

        # center wavelengths for all bands
        center_wavelengths = np.array([
            float(im.tags(bi)['center_wavelength'])
            if 'center_wavelength' in im.tags(bi) else np.nan
            for bi in range(1, im.count + 1)
        ])  # yapf: disables

        # if the image appears to an RGB or RGBA image, and it does not have all its center wavelengths,
        # then populate the missing wavelengths with default RGB values
        if (len(non_alpha_bands) == 3) and (sum(np.isnan(center_wavelengths)) > 0):
            for i, rgb_cw in zip(non_alpha_bands - 1, [.660, .520, .450]):
                center_wavelengths[i] = rgb_cw if np.isnan(center_wavelengths[i]) else center_wavelengths[i]
            # TODO: this gets displayed once on init and another time on match for some images
            logger.debug(f'Assuming standard RGB center wavelengths for {name}.')

        self._name = name
        self._im = im
        self._bands = bands
        self._center_wavelengths = center_wavelengths[bands - 1]

        band_names = np.array(im.descriptions)[bands - 1]
        if all(band_names):
            self._band_names = band_names
        else:
            self._band_names = np.array(
                [im.descriptions[bi - 1] if im.descriptions[bi - 1] else f'B{bi}' for bi in bands]
            )

    @property
    def name(self) -> str:
        return self._name

    @property
    def im(self) -> rio.DatasetReader:
        return self._im

    @property
    def bands(self) -> np.ndarray:
        return self._bands

    @property
    def band_names(self) -> np.ndarray:
        return self._band_names

    @property
    def count(self) -> int:
        return len(self.bands)

    @property
    def center_wavelengths(self) -> np.ndarray:
        return self._center_wavelengths

    def select(self, index: np.ndarray) -> 'ReflBands':
        bands = self._bands[index]
        band_names = self._band_names[index]
        center_wavelengths = self._center_wavelengths[index]


    def match(self, other: 'ReflBands', force=False) -> Tuple['ReflBands', 'ReflBands']:
        if other.count < self.count:
            if not force:
                raise ValueError(f'{other.name} has fewer bands than {self.name}.')
            else:
                logger.warning(f'{other.name} has fewer bands than {self.name}.')

        match_bands = np.array([np.nan] * self.count) # TODO: deal with src.count > ref.count
        # match self with other bands based on center wavelength metadata
        if any(self.center_wavelengths) and any(other.center_wavelengths):
            # TODO: can consider using a linear programming type optimisation here,
            #  e.g. https://stackoverflow.com/questions/67368093/find-optimal-unique-neighbour-pairs-based-on-closest-distance

            # absolute distance matrix between self and other center wavelengths
            abs_dist = np.abs(self.center_wavelengths[:, np.newaxis] - other.center_wavelengths[np.newaxis, :])
            def greedy_match(dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                """
                Greedy matching of `self` to `other` bands based on the provided center wavelength distance matix,
                `dist`. `self` bands must be down the rows, and `other` bands along the cols of `dist`.
                Will match one `other` band for each `self` band until either all `self` or all `other` bands
                have been matched.  Works for all cases where self.count != other.count.
                """
                # match_idx[i] is the index of the `other` band that matches with the ith `self` band
                match_idx = np.array([np.nan] * dist.shape[0])
                match_dist = np.array([np.nan] * dist.shape[0]) # distances corresponding to the above matches

                # repeat until all self or other bands have been matched
                while not all(np.isnan(np.nanmin(dist, axis=1))) or not all(np.isnan(np.nanmin(dist, axis=0))):
                    # find the row with the smallest distance in it
                    min_dist = np.nanmin(dist, axis=1)
                    min_dist_row_idx = np.nanargmin(min_dist)
                    min_dist_row = dist[min_dist_row_idx, :]
                    # store match idx and distance for this row
                    match_idx[min_dist_row_idx] = np.nanargmin(min_dist_row)
                    match_dist[min_dist_row_idx] = min_dist_per_row[min_dist_row_idx]
                    # set the matched row and col to nan, so that it is not used again
                    dist[:, int(match_idx[min_dist_row_idx])] = np.nan
                    dist[min_dist_row_idx, :] = np.nan

                return match_dist, match_idx

            match_dist, match_idx = greedy_match(abs_dist)

            # if any of the matched distances are greater than a threshold, raise an informative error,
            # or log a warning, depending on `force`
            # TODO: do a relative (to self wavelength), rather than absolute comparison
            if any(match_dist > self._max_wavelength_diff):
                err_idx = match_dist > self._max_wavelength_diff
                self_err_band_names = list(self.band_names[err_idx])
                other_err_band_names = list(other.band_names[np.int64(match_idx[err_idx])])
                err_dists = list(match_dist[err_idx].round(3))
                if not force:
                    raise ValueError(
                        f'{self.name} band(s) {self_err_band_names} could not be auto-matched.  The nearest '
                        f'{other.name} band(s) were {other_err_band_names}, at center wavelength difference(s) of '
                        f'{err_dists} (um) respectively.'
                    )
                else:
                    logger.warning(
                        f'Force matching {self.name} band(s) {self_err_band_names} with {other.name} band(s) '
                        f'{other_err_band_names}, at center wavelength differences of {err_dists} (um) respectively.'
                    )

            matched = ~np.isnan(match_idx)
            match_bands[matched] = other.bands[np.int64(match_idx[matched])]
            logger.debug(
                f'Matching {self.name} band(s) {list(self.band_names[matched])} with {other.name} band(s) '
                f'{list(other.band_names[np.int64(match_idx[matched])])}, at center wavelength difference(s) of '
                f'{list(match_dist[matched].round(3))} (um) respectively.'
            )

        # match any remaining bands that don't have center wavelength metadata
        if sum(~np.isnan(match_bands)) < min(self.count, other.count):
            unmatched = np.isnan(match_bands)
            unmatch_other_bands = np.array([bi for bi in other.bands if bi not in match_bands], dtype=int)
            unmatch_other_band_names = other.band_names[unmatch_other_bands - 1]
            if self.count == other.count:
                # assume unmatched self and other image bands are in matching order
                match_bands[unmatched] = unmatch_other_bands
                logger.debug(
                    f'Matching {self.name} band(s) {list(self.band_names[unmatched])} in file order with {other.name} '
                    f'band(s) {list(unmatch_other_band_names)}.'
                )
            elif force:
                # match the remaining N self bands with the first N unmatched bands of other (N=sum(unmatched))
                unmatch_other_bands = unmatch_other_bands[:sum(unmatched)]
                unmatch_other_band_names = unmatch_other_band_names[:sum(unmatched)]
                # if there are not N unmatched bands in other, truncate unmatched to just match what we can
                unmatched = unmatched[:len(unmatch_other_bands)]
                match_bands[unmatched] = unmatch_other_bands
                logger.warning(
                    f'Matching {self.name} band(s): {list(self.bands[unmatched])} in file order with {other.name} '
                    f'band(s): {list(unmatch_other_band_names)}.'
                )
            else:
                # raise an error when remaining unmatched bands counts do not match and `force` is False
                raise ValueError(
                    f'Could not match {self.name} band(s) {list(self.bands[unmatched])} with {other.name} '
                    f'band(s) {list(unmatch_other_band_names)}.  Ensure {self.name} and {other.name} non-alpha band '
                    f'counts match, {self.name} and {other.name} have `center_wavelength` tags for each band, or set '
                    f'`force` to True.'
                )
        # Truncate self and other to reflect the matched bands
        matched = ~np.isnan(match_bands)
        self_match = ReflBands(self.im, self.name, bands=self.bands[np.where(matched)[0]])
        other_match = ReflBands(other.im, other.name, bands=np.int64(match_bands[matched]))
        # TODO: log debug table of matching bands including names, wavelengths etc
        logger.info(
            f'Matching {self.name} band(s) {list(self_match.band_names)} with {other.name} band(s) '
            f'{list(other_match.band_names)}.'
        )
        return self_match, other_match
