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
from enum import Enum


class Model(str, Enum):
    """ Enumeration for the surface reflectance correction model. """
    gain = 'gain'
    """ 
    Gain-only model, suitable for haze-free and zero offset images (i.e. images where a surface reflectance of 
    zero corresponds to a pixel value of ~zero). 
    """
    gain_blk_offset = 'gain-blk-offset'
    """ 
    Gain-only model applied to offset normalised image blocks.  Suitable for most image combinations.
    """
    gain_offset = 'gain-offset'
    """
    Gain and offset model.  The most accurate model, but sensitive to differences between source and reference, 
    such as shadowing and land cover changes.  Suitable for well-matched source / reference image pairs.  
    """


class ProcCrs(str, Enum):
    """ Enumeration for the processing space (image co-ordinate system and resolution in which to perform processing). """
    auto = 'auto'
    src = 'src'
    ref = 'ref'
