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


class HomonimError(Exception):
    """ Root exception class. """


class UnsupportedImageError(HomonimError):
    """ Raised when reading a 12bit jpeg compressed image. """


class ImageContentError(HomonimError):
    """ Raised when an image has insufficient coverage or bands. """


class BlockSizeError(HomonimError):
    """ Raised when the image block size is invalid. """


class ImageProfileError(HomonimError):
    """ Raised when an image profile is invalid. """


class ImageFormatError(HomonimError):
    """ Raised when an image format is invalid. """


class IoError(HomonimError):
    """ Raised when accessing unopened file(s). """


class HomonimWarning(RuntimeWarning):
    """ Homonim runtime warning. """


class BandMatchWarning(HomonimWarning):
    """ Warn about band matching issues. """


class ImageFormatWarning(HomonimWarning):
    """ Warn about image format issues. """


class ConfigWarning(HomonimWarning):
    """ Warn about configuration issues. """
