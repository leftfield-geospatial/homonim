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
import os
import pathlib
import logging

from homonim.compare import RasterCompare
from homonim.enums import Model, ProcCrs
from homonim.fuse import RasterFuse
from homonim.kernel_model import KernelModel
from homonim.stats import ParamStats

# Add a NullHandler to the package logger to hide logs by default.  Applications can then add
# their own handler(s).
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path(os.getcwd())
