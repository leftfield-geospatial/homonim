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
            raise ValueError(
                f'User specified {name} bands contains alpha bands: {set(bands).difference(non_alpha_bands)}'
            )
        # test bands have center_wavelength metadata when it exists
        if (bands is not None) and len(refl_bands) and (not set(bands).issubset(refl_bands)):
            logger.warning(
                f'User specified {name} bands contain non-reflectance bands: {set(bands).difference(refl_bands)}'
            )

        if bands is not None:
            # use bands if it was specified
            bands = np.array(bands)
        elif len(refl_bands) > 0:
            # else use bands with center wavelengths if any
            logger.debug(f'Using {name} reflectance bands: {refl_bands}')
            bands = np.array(refl_bands)
        elif len(non_alpha_bands) > 0:
            # else use non-alpha bands
            logger.debug(f'Using {name} non-alpha bands: {non_alpha_bands}')
            bands = np.array(non_alpha_bands)
        else:
            raise ValueError(f'There are no valid non-alpha/reflectance {name} bands to use.')

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
            logger.debug(f'Assuming {name} image is RGB/RGBA.')

        self._name = name
        self._im = im
        self._bands = bands
        self._center_wavelengths = center_wavelengths[bands - 1]
        # TODO: get band string names / descriptors to use in log messages/ exceptions

    @property
    def name(self) -> str:
        return self._name

    @property
    def im(self) -> rio.DatasetReader:
        return self._im

    @property
    def bands(self) -> np.ndarray:
        return self._bands

    # @bands.setter
    # def bands(self, value: Tuple[int, ...]):
    #     if set(value) > set(self._bands):
    #         raise ValueError('`value` must be a subset of `bands`')
    #     self._bands = value
    #     self._center_wavelengths

    @property
    def count(self) -> int:
        return len(self.bands)

    @property
    def center_wavelengths(self) -> np.ndarray:
        return self._center_wavelengths

    def match(self, other: 'ReflBands', force=False) -> Tuple['ReflBands', 'ReflBands']:
        # TODO: maybe force the user to either provide src/ref images with same num bands, or src/ref_bands with same
        #  number of bands?
        if other.count < self.count:
            if not force:
                raise ValueError(f'{other.name} has fewer bands than {self.name}.')
            else:
                logger.warning(f'Using {other.count} of {self.count} {self.name} bands only.')

        match_bands = np.array([np.nan] * self.count) # TODO: deal with src.count > ref.count
        # match self with other bands based on center wavelength metadata
        if any(self.center_wavelengths) and any(other.center_wavelengths):
            # TODO: consider using a linear programming type optimisation here,
            #  e.g. https://stackoverflow.com/questions/67368093/find-optimal-unique-neighbour-pairs-based-on-closest-distance

            # absolute distance matrix between self and other center wavelengths
            abs_dist = np.abs(self.center_wavelengths[:, np.newaxis] - other.center_wavelengths[np.newaxis, :])
            def greedy_match(dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                """ Greedy matching of distances.  Returns match distances and indices. """
                match_idx = np.array([np.nan] * dist.shape[0])
                match_val = np.array([np.nan] * dist.shape[0])
                for rowi, dist_row in enumerate(dist):
                    if not all(np.isnan(dist_row)):
                        match_idx[rowi] = np.nanargmin(dist_row)
                        match_val[rowi] = dist_row[int(match_idx[rowi])]
                        # if force or (val[rowi] < self._max_wavelength_diff):
                        dist[:, int(match_idx[rowi])] = np.nan  # prevent same band being selected twice
                return match_val, match_idx

            match_dist, match_idx = greedy_match(abs_dist)
            # # if there are more matched bands > other.count
            # if sum(~np.isnan(match_bands)) > other.count:
            #     # truncates valid matched bands to the best N unique matches with other, where N = other.count
            #     dist_idx = np.argsort(match_dist)[other.count:]
            #     match_bands[dist_idx] = np.nan
            #     match_dist[dist_idx] = np.nan

            # if any of the matched distances are greater than a threshold, raise an informative error,
            # or log a warning, depending on `force`
            if any(match_dist > self._max_wavelength_diff):
                err_idx = match_dist > self._max_wavelength_diff
                self_err_bands = tuple(self.bands[err_idx])
                other_err_bands = tuple(other.bands[np.int64(match_idx[err_idx])])
                err_dists = tuple(match_dist[err_idx].round(3))
                if not force:
                    raise ValueError(
                        f'{self.name} bands {self_err_bands} could not be auto-matched.  The nearest {other.name} '
                        f'bands were {other_err_bands}, at center wavelength differences of {err_dists} (um) '
                        f'respectively.'
                    )
                else:
                    logger.warning(
                        f'Force matching {self.name} bands {self_err_bands} with {other.name} bands {other_err_bands}'
                        f', at center wavelength differences of {err_dists} (um) respectively.'
                    )

            matched = ~np.isnan(match_idx)
            match_bands[matched] = other.bands[np.int64(match_idx[matched])]
            # log a message about which bands have been mactched
            logger.debug(
                f'Matching {self.name} bands {tuple(self.bands[matched])} with {other.name} bands '
                f'{tuple(other.bands[np.int64(match_idx[matched])])}, at center wavelength differences of '
                f'{tuple(match_dist[matched].round(3))} (um) respectively.'
            )

        # match any remaining bands that don't have center wavelength metadata
        if sum(~np.isnan(match_bands)) < other.count:
            unmatched = np.isnan(match_bands)
            unmatch_other_bands = np.array([bi for bi in other.bands if bi not in match_bands])
            if self.count == other.count:
                # assume unmatched self and other image bands are in matching order
                match_bands[unmatched] = unmatch_other_bands
                logger.debug(
                    f'Matching {self.name} bands: {tuple(self.bands[unmatched])} in file order with {other.name} '
                    f'bands: {tuple(unmatch_other_bands)}.'
                )
            elif force:
                # match the remaining N self bands with the first N unmatched bands of other (N=sum(unmatched))
                unmatch_other_bands = unmatch_other_bands[:sum(unmatched)]
                # if there are not N unmatched bands in other, truncate unmatched to just match what we can
                unmatched = unmatched[:len(unmatch_other_bands)]
                match_bands[unmatched] = unmatch_other_bands
                logger.warning(
                    f'Matching {self.name} bands: {tuple(self.bands[unmatched])} in file order with {other.name} '
                    f'bands: {tuple(unmatch_other_bands)}.'
                )
            else:
                # raise an error when remaining unmatched bands counts do not match and `force` is False
                raise ValueError(
                    f'Could not match {self.name} bands: {tuple(self.bands[unmatch_idx])} with {other.name} '
                    f'bands: {tuple(unmatch_other_bands)}.  \nEnsure {self.name} and {other.name} non-alpha band '
                    f'counts match, {self.name} and {other.name} have `center_wavelength` tags for each band, or '
                    f'set `force` to True.'
                )
        # Truncate self and other to reflect the matched bands
        matched = ~np.isnan(match_bands)
        self_match = ReflBands(self.im, self.name, bands=self.bands[np.where(matched)[0]])
        other_match = ReflBands(self.im, self.name, bands=np.int64(match_bands[matched]))
        return self_match, other_match
