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

from enum import Enum


class _StrChoiceEnum(str, Enum):
    """String value enumeration class that can be used with a ``click.Choice()``
    parameter type.
    """

    def __repr__(self):
        return self._value_

    def __str__(self):
        return self._value_

    @property
    def name(self):
        # override for click>=8.2.0 Choice options which match passed values to Enum
        # names
        return self._value_


class Model(_StrChoiceEnum):
    """
    Linear model variants for correcting to surface reflectance.

    Roughly speaking, gain compensates for atmospheric absorption and anisotropic
    (BRDF) effects, and offset (when present) compensates for atmospheric reflectance
    and haze.
    """

    gain = 'gain'
    """ 
    Gain-only model, suitable for haze-free and zero offset images (i.e. images where a 
    surface reflectance of zero corresponds to a pixel value of ± zero). 
    """
    gain_blk_offset = 'gain-blk-offset'
    """ 
    Gain-only model applied to offset normalised image blocks.  Suitable for most 
    source - reference combinations.
    """
    gain_offset = 'gain-offset'
    """
    Gain and offset model.  The most accurate model, but sensitive to differences 
    between source and reference, such as shadowing and land cover changes.  Suitable 
    for well-matched source - reference image pairs.  
    """


class ProcCrs(_StrChoiceEnum):
    """
    CRS and pixel grid in which correction parameters are estimated.
    """

    auto = 'auto'
    """Lowest resolution of the source and reference image CRSs (recommended)."""
    src = 'src'
    """Source image CRS."""
    ref = 'ref'
    """Reference image CRS."""


class Driver(_StrChoiceEnum):
    """Raster format drivers."""

    gtiff = 'gtiff'
    """GeoTIFF."""
    cog = 'cog'
    """Cloud Optimised GeoTIFF."""
