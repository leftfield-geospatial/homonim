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
from typing import List

import numpy as np
import rasterio as rio

logger = logging.getLogger(__name__)


class ReflBands():
    _max_wavelength_diff = 100
    def __init__(self, im: rio.DatasetReader, name: str = None, bands: List[int] = None):

        name = name or im.name or ''
        non_alpha_bands = [bi + 1 for bi in range(im.count) if im.colorinterp[bi] != ColorInterp.alpha]
        refl_bands = [bi for bi in non_alpha_bands if 'center_wavelength' in im.tags(bi)]
        # test bands does not contain alpha bands
        if bands and not set(bands).issubset(non_alpha_bands):
            raise ValueError(f'`{name}_bands` contains alpha bands: {set(bands).difference(non_alpha_bands)}')
        # test bands has center_wavelength metadata when it exists
        if bands and len(refl_bands) and not set(bands).issubset(refl_bands):
            # TODO: change msg (& above) to 'User specified {name} bands contain...'
            logger.warning(f'`{name}_bands` contains non-reflectance bands: {set(bands).difference(refl_bands)}')

        center_wavelengths = np.array([
                im.tags(bi)['center_wavelength']
                if 'center_wavelength' in im.tags(bi) else None
                for bi in range(im.count)
            ]
        )  # yapf: disable


        if bands:
            bands = np.ndarray(bands)
        elif len(refl_bands) > 0:
            logger.debug(f'Using {name} reflectance bands: {refl_bands}')
            bands = np.ndarray(refl_bands)
        elif len(non_alpha_bands) > 0:
            logger.debug(f'Using {name} non-alpha bands: {non_alpha_bands}')
            bands = np.ndarray(non_alpha_bands)
        else:
            raise ValueError(f'There are no valid non-alpha/reflectance {name} bands to use.')

        # center wavelengths for all bands
        center_wavelengths = np.array([
                im.tags(bi)['center_wavelength']
                if 'center_wavelength' in im.tags(bi) else np.nan
                for bi in range(im.count)
            ]
        )  # yapf: disable

        # if the image appears to an RGB or RGBA image, and it does not have all its center wavelengths,
        # then populate the missing wavelengths with default RGB values
        if (len(non_alpha_bands) == 3) and np.count_nonzero(center_wavelengths) < 3:
            for bi, rgb_cw in zip(non_alpha_bands, [660, 520, 450]):
                center_wavelengths[bi] = rgb_cw if np.isnan(center_wavelengths[bi]) else center_wavelengths[bi]
            logger.debug(f'Assuming {name} image is a RGB or RGBA image.')

        self._name = name
        self._im = im
        self._bands = bands
        self._center_wavelengths = center_wavelengths[bands]
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

    def match(self, other: 'ReflBands', force=False):

        if other.count < self.count:
            if not force:
                raise ValueError(f'{other.name} has fewer bands than {self.name}.')
            else:
                logger.warning(f'Using {other.count} of {self.count} {self.name} bands only.')

        match_bands = np.array([np.nan] * self.count) # TODO: deal with src.count > ref.count
        if any(self.center_wavelengths) and any(other.center_wavelengths):
            # match self and other bands with center wavelengths
            abs_dist = np.abs(self.center_wavelengths[:, np.newaxis] - other.center_wavelengths[:, np.newaxis])

            # TODO: we need to deal with the case where mult self bands get matched to the same ref band
            def nanmin(array: np.ndarray) -> np.ndarray:
                """ Return min and argmin along the cols (axis=1), reverting to nan if the whole row is nan. """
                idx = np.array([np.nan] * array.shape[0])
                val = np.array([np.nan] * array.shape[0])
                for rowi, row in enumerate(array):
                    idx[rowi] = np.nanargmin(row) if not all(np.isnan(row)) else np.nan
                    val[rowi] = np.min(row)
                return row, idx

            match_dist, match_bands = nanargmin(abs_dist)

            if sum(~np.isnan(match_bands)) > other.count:
                # truncate match_bands to the best N unique matches with other, where N = other.count
                # match_idx = ~np.isnan(match_bands)
                dist_idx = np.argsort(match_dist)[other.count:]
                match_bands[dist_idx] = np.nan
                match_dist[dist_idx] = np.nan

            if any(match_dist > self._max_wavelength_diff):
                err_idx = match_dist > self._max_wavelength_diff
                self_err_bands = tuple(self.bands[err_idx])
                other_err_bands = tuple(other.bands[match_bands[err_idx]])
                err_dists = tuple(match_dist[err_idx])
                if not force:
                    raise ValueError(
                        f'{self.name} bands: {self_err_bands} could not be auto-matched.  The nearest {other.name} '
                        f'bands were: {other_err_bands}, at center wavelength differences of: {err_dists} (nm) '
                        f'respectively.'
                    )
                else:
                    logger.warning(
                        f'Force matching {self.name} bands: {self_err_bands} with {other.name} bands: {other_err_bands}'
                        f', at center wavelength differences of: {err_dists} (nm) respectively.'
                    )

            match_idx = ~np.isnan(match_bands)
            logger.debug(
                f'Matching {self.name} bands: {tuple(self.bands[match_idx])} with {other.name} bands: '
                f'{tuple(match_bands[match_idx])}, at center wavelength differences of: {tuple(match_dist[match_idx])} '
                f'(nm) respectively.'
            )

        if sum(np.isnan(match_bands)) < other.count:
            # match remaining unmatched bands
            unmatch_idx = np.isnan(match_bands)
            if self.count == other.count:
                # assume remaining bands are in matching order
                unmatch_other_bands = set(other.bands).difference(match_bands)
                match_bands[unmatch_idx] = unmatch_other_bands
                logger.debug(
                    f'Matching {self.name} bands: {tuple(self.bands[unmatch_idx])} in file order with {other.name} '
                    f'bands: {tuple(unmatch_other_bands)}.'
                )
            elif force:
                # take the first sum(unmatch_idx) bands of other, and assumne in matching order with self.
                unmatch_other_bands = list(set(other.bands).difference(match_bands))[:sum(unmatch_idx)]
                # if there are not sum(unmatch_idx) bands in other, just use what is there.
                unmatch_idx = unmatch_idx[:len(unmatch_other_bands)]
                match_bands[unmatch_idx] = unmatch_other_bands
                logger.warning(
                    f'Matching {self.name} bands: {tuple(self.bands[unmatch_idx])} in file order with {other.name} '
                    f'bands: {tuple(unmatch_other_bands)}.'
                )
            else:
                # could not match bands
                unmatch_other_bands = set(other.bands).difference(match_bands)
                raise ValueError(
                    f'Could not match {self.name} bands: {tuple(self.bands[unmatch_idx])} with {other.name} '
                    f'bands: {tuple(unmatch_other_bands)}.  \nEnsure {self.name} and {other.name} non-alpha band '
                    f'counts match, {self.name} and {other.name} have `center_wavelength` tags for each band, or '
                    f'set `force` to True.'
                )

        return ReflBands(other.im, other.name, match_bands)
