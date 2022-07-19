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
from typing import Tuple, List

import numpy as np
import rasterio as rio

logger = logging.getLogger(__name__)


class ReflBands():
    def __init__(self, im: rio.DatasetReader, name: str = '', bands: List[int] = None):

        center_wavelengths = np.array(
            [
                im.tags(bi)['center_wavelength']
                if 'center_wavelength' in im.tags(bi) else None
                for bi in range(im.count)
            ]
        )  # yapf: disable
        non_alpha_bands = [bi + 1 for bi in range(im.count) if im.colorinterp[bi] != ColorInterp.alpha]
        refl_bands = (
            [bi for bi in range(im.count) if center_wavelengths[bi]]
            if any(center_wavelengths) else None
        )  # yapf: disable
        if bands:
            bands_diff = set(bands) - set(non_alpha_bands)
            if len(bands_diff) > 0:
                raise ValueError(f'`{name}_bands` contains invalid bands: {tuple(bands_diff)}')
        if refl_bands:
            bands_diff = set(bands) - set(refl_bands)
            logger.warning(f'`{name}_bands` contains non-reflectance bands: {tuple(bands_diff)}')

        self._name = name
        self._im = im
        self._bands = tuple(bands or refl_bands or non_alpha_bands)

        if len(non_alpha_bands) == 3 and not any(center_wavelengths):
            # assume it is an RGB image, and set wavelengths to typical values
            center_wavelengths[non_alpha_bands] = [660, 520, 450]
        self._center_wavelengths = tuple(center_wavelengths[self._bands])
        # if self._center_wavelengths.count(None) > 0:
        #     none_bands = tuple([bi for bi, cw in zip(self._bands, self._center_wavelengths) if cw is None])
        #     logger.warning(f'There is no center wavelength data for the {self.name} bands: {none_bands}')

    @property
    def name(self) -> str:
        return self._name

    @property
    def im(self) -> rio.DatasetReader:
        return self._im

    @property
    def bands(self) -> Tuple[int, ...]:
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
    def center_wavelengths(self) -> Tuple[float, ...]:
        return self._center_wavelengths

    def match(self, other: 'ReflBands', force=False):
        if not force and other.count < self.count:
            raise ValueError(f'{other.name.capitalize()} has fewer bands than {self.name}')

        if any(self.center_wavelengths) and any(other.center_wavelengths):
            if sum([cw is not None for cw in self.center_wavelengths]) >
                self_wavelengths = np.array(self.center_wavelengths)
            other_wavelengths = np.array(other.center_wavelengths)
            for self_wavelength in self_wavelengths:


        elif self.count == other.count:
            # TODO: warning?
            # assume self and other bands are in same order
            return other
        elif force:
            # assume self/other bands are in matching order, and truncate to match
            trunc_count = min(self.count, other.count)
            trunc_name = self.name if trunc_count == self.count else other.name
            logger.warning(
                f'Assuming {self.name} and {other.name} bands are in matching order.  '
                f'Truncating {trunc_name} to {trunc_count} bands.'
            )
            # TODO: return two instances or something else, instead of modifying self in place?
            self.__init__(self.im, self.name, self.bands[:trunc_count])
            return ReflBands(other.im, other.name, other.bands[:trunc_count])
        else:
            raise ValueError(f'Cannot match {self.name} and {other.name} bands.')
