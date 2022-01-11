"""
    Homonim: Radiometric homogenisation of aerial and satellite imagery
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

class Method(str, Enum):
    """Enumeration for homogenisation method"""
    gain = 'gain'
    gain_im_offset = 'gain_im_offset'
    gain_offset = 'gain_offset'

class ProcCrs(str, Enum):
    """Enumeration for processing CRS (co-oridinate system)"""
    auto = 'auto'
    src = 'src'
    ref = 'ref'
