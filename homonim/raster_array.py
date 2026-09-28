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

import logging
import multiprocessing
from os import PathLike
from typing import Any

import numpy as np
import rasterio as rio
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.dtypes import can_cast_dtype
from rasterio.enums import MaskFlags
from rasterio.io import DatasetReader, DatasetWriter
from rasterio.transform import TransformMethodsMixin, array_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, WindowMethodsMixin

from homonim import utils
from homonim.enums import Driver
from homonim.errors import HomonimError, ImageFormatError

logger = logging.getLogger(__name__)


class RasterArray(TransformMethodsMixin, WindowMethodsMixin):
    # TODO: rename with _
    default_nodata = float('nan')  # default internal nodata value
    default_dtype = 'float32'  # default internal data type

    def __init__(
        self,
        array: np.ndarray,
        crs: CRS,
        transform: Affine,
        nodata: float | None = default_nodata,
    ):
        """
        Class for reading, writing and reprojecting a geo-referenced NumPy array.

        :param array:
            Array of image data.  2D if there is one band, or 3D with bands along the
            first dimension otherwise.
        :param crs:
            Coordinate reference system of ``array``.
        :param transform:
            Geo-referencing transform.
        :param nodata:
            Value of nodata (invalid) pixels in ``array``.  Can be ``None`` if there
            are no invalid pixels.
        """
        if (array.ndim < 2) or (array.ndim > 3):
            raise ValueError(
                "'array' should be have 2 or 3 dimensions with bands along the first "
                'dimension.'
            )
        self._array = array

        if isinstance(crs, CRS):
            self._crs = crs
        else:
            raise TypeError(
                f"'crs' must be a RasterIO 'CRS' instance, not {type(crs)}."
            )

        if not isinstance(transform, Affine):
            raise TypeError(
                f"'transform' must be a RasterIO 'Affine' instance, not "
                f"'{type(transform)}."
            )

        self._transform = transform
        self._nodata = nodata
        self._mask = None

    @classmethod
    def from_rio_dataset(
        cls,
        dataset: DatasetReader,
        indexes: int | list[int] | None = None,
        window: Window | None = None,
        dtype: str = default_dtype,
        nodata: float | None = None,
        **kwargs,
    ) -> 'RasterArray':
        """
        Create a RasterArray from an open RasterIO dataset.

        :param dataset:
            Dataset to read from.
        :param indexes:
            Band index(es) to read (1 based).  If ``None``, all non-alpha bands are
            read.
        :param window:
            Boundless region of the dataset to read.  If ``None``, the full
            dataset extent is read.
        :param dtype:
            RasterArray data type.
        :param nodata:
            RasterArray nodata value.  If ``None``, it defaults to the dataset's nodata
            value when it has one, otherwise ``nan``.
        :param kwargs:
            Additional keyword arguments to pass to
            :meth:`~rasterio.io.DatasetReader.read`.

        :return:
            RasterArray.
        """
        if indexes is None:
            indexes = utils.get_nonalpha_bands(dataset)
            indexes = indexes if len(indexes) > 1 else indexes[0]

        if window is None:
            # window of the full dataset extent
            window = Window(
                col_off=0, row_off=0, width=dataset.width, height=dataset.height
            )

        # determine if bands have masks (i.e. internal/side-car mask or alpha channel)
        is_masked = any(
            MaskFlags.per_dataset in dataset.mask_flag_enums[bi - 1]
            for bi in ([indexes] if np.ndim(indexes) == 0 else indexes)
        )

        nodata_changed = False
        if nodata is None:
            # use the dataset's nodata when it has one and is not masked, otherwise
            # use default_nodata
            nodata = (
                cls.default_nodata
                if (is_masked or dataset.nodata is None)
                else dataset.nodata
            )
        elif not utils.nan_equals(nodata, dataset.nodata):
            nodata_changed = True
            with np.errstate(invalid='ignore'):
                if not can_cast_dtype(nodata, dtype):
                    raise HomonimError(
                        f"'nodata' value: {nodata} cannot be safely cast to 'dtype': "
                        f'{dtype}.'
                    )

        # crop the boundless window to the dataset bounds
        bounded_window = window.crop(dataset.height, dataset.width)

        # create slices to crop an array corresponding to the boundless window into an
        # array corresponding to bounded_window
        bounded_slices = [
            slice(s.start - off, s.stop - off)
            for s, off in zip(
                bounded_window.toslices(),
                (window.row_off, window.col_off),
                strict=True,
            )
        ]
        bounded_slices = tuple(bounded_slices)

        # create an array of nodata matching the boundless window dimensions
        shape = (window.height, window.width)
        if np.ndim(indexes) > 0:
            shape = (len(indexes), *shape)
            bounded_slices = (slice(shape[0]), *bounded_slices)
        array = np.full(shape, fill_value=nodata, dtype=dtype)

        # read into the bounded region of the array (this is a lot faster than using
        # read(boundless=True))
        bounded_array = array[bounded_slices]
        dataset.read(
            out=bounded_array,
            indexes=indexes,
            window=bounded_window,
            out_dtype=dtype,
            **kwargs,
        )

        if is_masked:
            # read the bounded region of the mask and apply it to the array
            bounded_mask = dataset.dataset_mask(window=bounded_window).view('bool')
            bounded_array[..., ~bounded_mask] = nodata
        elif nodata_changed:
            # change the dataset nodata value to the user value
            bounded_mask = utils.nan_equals(bounded_array, dataset.nodata)
            bounded_array[..., bounded_mask] = nodata

        return cls(array, dataset.crs, dataset.window_transform(window), nodata=nodata)

    @property
    def array(self) -> np.ndarray:
        """Image data array.  2D if there is one band, or 3D with bands along the
        first dimension otherwise.
        """
        return self._array

    @array.setter
    def array(self, value: np.ndarray):
        if np.all(value.shape[-2:] == self._array.shape[-2:]):
            self._array = value
            self._mask = None
        else:
            raise ValueError(
                "The 'array' property can only be set to another array with the same "
                '(row, col) dimensions.'
            )

    @property
    def crs(self) -> CRS:
        """Coordinate reference system."""
        return self._crs

    @property
    def width(self) -> int:
        """Array width in pixels."""
        return self.shape[-1]

    @property
    def height(self) -> int:
        """Array height in pixels."""
        return self.shape[-2]

    @property
    def shape(self) -> tuple[int, int]:
        """Array (row, col) dimensions in pixels."""
        return self._array.shape[-2:]

    @property
    def count(self) -> int:
        """Number of array bands."""
        return self._array.shape[0] if self.array.ndim == 3 else 1

    @property
    def dtype(self) -> str:
        """Array data type."""
        return self._array.dtype.name

    @property
    def transform(self) -> Affine:
        """Array geo-referencing transform."""
        return self._transform

    @property
    def res(self) -> tuple[float, float]:
        """Array (col, row) resolution in units of the :attr:`crs`."""
        return abs(self._transform.a), abs(self._transform.e)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) coordinates of the array extents."""
        return array_bounds(*self.shape, self._transform)

    @property
    def profile(self) -> dict[str, Any]:
        """RasterIO profile of the array."""
        return dict(
            crs=self._crs,
            transform=self._transform,
            nodata=self._nodata,
            count=self.count,
            width=self.width,
            height=self.height,
            dtype=self.dtype,
        )

    @property
    def proj_profile(self) -> dict[str, Any]:
        """The ``crs``, ``transform`` and ``shape`` items of the :attr:`profile` for
        passing as keyword arguments to :meth:`reproject`.
        """
        # TODO: remove if possible and replace with reproject_like?
        return dict(crs=self._crs, transform=self._transform, shape=self.shape)

    @property
    def nodata(self) -> float | None:
        """Value of nodata (invalid) pixels."""
        return self._nodata

    @nodata.setter
    def nodata(self, value: float | None):
        if value is None or self._nodata is None:
            self._nodata = value
        elif not (utils.nan_equals(value, self._nodata)):
            # if the new nodata value is different to the current nodata,
            # set the mask area in array to the new nodata value
            self._array[self._array == self._nodata] = value
            self._nodata = value

    def _crop_to_window(self, window: Window) -> 'RasterArray':
        """Return a view into the RasterArray, cropped to the given window bounds."""
        ranges = np.array(window.toranges()).T
        if any(ranges[0] < 0) or any(ranges[1] > self.shape):
            raise ValueError(
                "'window' bounds lie outside the bounds of the RasterArray."
            )
        return RasterArray(
            self._array[(..., *window.toslices())],
            self._crs,
            self.window_transform(window),
            nodata=self._nodata,
        )

    def _convert_array_dtype(self, dtype: str, nodata: float | None = None) -> np.array:
        """Return the image array converted to dtype, rounding and clipping when
        dtype is integer.  Passing nodata will set invalid areas in the returned array
        to this value.
        """
        with np.errstate(invalid='ignore'):
            if nodata is not None and not can_cast_dtype(nodata, dtype):
                raise HomonimError(
                    f"'nodata': {nodata} cannot be safely cast to 'dtype': {dtype}."
                )

        # return an array converted to dtype if that is safe and nodata remains
        # the same
        safe_cast = np.can_cast(self.dtype, dtype, casting='safe')
        nodata_unchanged = (
            nodata is None
            or self._nodata is None
            or utils.nan_equals(nodata, self.nodata)
        )
        if safe_cast and nodata_unchanged:
            return self._array.astype(dtype, copy=False)

        # create a copy of the array with promoted dtype to allow nodata conversion
        # and clipping
        array = self._array.astype(np.promote_types(self.dtype, dtype), copy=True)

        # round if converting from float to integer dtype
        rounded = False
        if np.issubdtype(self.dtype, np.floating) and np.issubdtype(dtype, np.integer):
            np.round(array, out=array)
            rounded = True

        # clip if converting to integer dtype with smaller range than current dtype
        if np.issubdtype(dtype, np.integer):
            src_info = (
                np.iinfo(self.dtype)
                if np.issubdtype(self.dtype, np.integer)
                else np.finfo(self.dtype)
            )
            dst_info = np.iinfo(dtype)
            if src_info.min < dst_info.min or src_info.max > dst_info.max:
                np.clip(array, dst_info.min, dst_info.max, out=array)

        # convert dtype (ignoring numpy warnings for float overflow or cast of nan to
        # integer)
        with np.errstate(invalid='ignore', over='ignore'):
            array = array.astype(dtype, copy=False, casting='unsafe')

        # set the nodata value if it has changed, or may be invalid after rounding
        if not nodata_unchanged or (
            nodata is not None and not self._nodata and rounded
        ):
            nodata_mask = utils.nan_equals(self._array, self._nodata)
            array[nodata_mask] = nodata

        return array

    def copy(self) -> 'RasterArray':
        """Return a deep copy of the RasterArray."""
        return RasterArray(
            self._array.copy(), self._crs, self._transform, nodata=self._nodata
        )

    def mask(self) -> np.ndarray[bool]:
        """Return the 2D mask of valid array pixels, as the OR of the individual band
        masks.
        """
        if self._nodata is None:
            mask = np.full(self._array.shape[-2:], True)
        else:
            mask = ~utils.nan_equals(self._array, self._nodata)
            if mask.ndim > 2:
                mask = np.any(mask, axis=0)
        return mask

    def to_rio_dataset(
        self,
        dataset: DatasetWriter,
        indexes: int | list[int] | None = None,
        window: Window | None = None,
        **kwargs,
    ) -> None:
        """
        Write the RasterArray into an open RasterIO dataset.

        The :attr:`mask` is written as an internal mask band when ``dataset.nodata``
        is ``None``, otherwise nodata pixels are converted from the RasterArray to
        dataset value before writing.

        :param dataset:
            Dataset to write into.
        :param indexes:
            Dataset band index(es) to write (1 based).  Should contain :attr:`count`
            elements. If ``None``, it defaults to the dataset non-alpha bands.
        :param window:
            Boundless region of the dataset to write into.  If ``None``, the full
            RasterArray extent is written into the corresponding dataset region.
        :param kwargs:
            Additional keyword arguments to pass to
            :meth:`~rasterio.io.DatasetWriter.write`.
        """
        # check that the RasterArray lies on the same pixel grid as the dataset
        ji_offset = ~dataset.transform * (self._transform.xoff, self._transform.yoff)
        is_int_offset = np.allclose(np.round(ji_offset), ji_offset)
        if not self.res == dataset.res or not is_int_offset:
            raise ImageFormatError(
                "'dataset' should lie on the same pixel grid as the RasterArray."
            )
        # TODO: this comparison is time-consuming to do for every write - benchmark
        #  removing it
        if self._crs != dataset.crs:
            raise ImageFormatError(
                "'dataset' should have the same CRS as the RasterArray."
            )

        if indexes is None:
            indexes = utils.get_nonalpha_bands(dataset)
            indexes = indexes if len(indexes) > 1 else indexes[0]

        if np.ndim(indexes) == 1 and len(indexes) != self.count:
            raise ValueError(
                "'indexes' should contain the same number of elements as the number "
                'of RasterArray bands.'
            )

        if window is None:
            # region in the dataset corresponding to the RasterArray extents
            window = utils.round_window_to_grid(dataset.window(*self.bounds))

        # crop the boundless window to the dataset bounds
        window = window.crop(dataset.height, dataset.width)

        # create a view into the RasterArray, cropped to the bounds of window
        ra_window = self.window(*dataset.window_bounds(window))
        ra_window = utils.round_window_to_grid(ra_window)
        crop_ra = self._crop_to_window(ra_window)

        # convert data type and write to dataset
        # TODO: clip to nbits when it is set in the dataset profile
        array = crop_ra._convert_array_dtype(dataset.dtypes[0], nodata=dataset.nodata)
        dataset.write(array, window=window, indexes=indexes, **kwargs)

        if dataset.nodata is None and (1 in np.array(indexes)):
            # TODO: this doesn't work if the dataset is written band-by-band,
            #  and bands don't have the same masks
            # write an internal mask (once for the first band, if the dataset is
            # written band-by-band)
            dataset.write_mask(crop_ra.mask(), window=window)

    def to_file(
        self,
        filename: str | PathLike,
        driver: Driver | str = Driver.gtiff,
        **creation_options,
    ) -> None:
        """
        Write the RasterArray to an image file.

        :param filename:
            Path or URI of the image file.
        :param driver:
            Image driver.
        :param creation_options:
            Driver specific creation options as a dictionary of ``name: value``
            pairs.  See the GDAL `GTiff
            <https://gdal.org/en/latest/drivers/raster/gtiff.html#creation
            -options>`__ and `COG <https://gdal.org/en/latest/drivers/raster/cog.html
            #creation -options>`__ documentation for details on the options for those
            drivers.
        """
        driver = Driver(driver.lower())
        with rio.Env(
            GDAL_NUM_THREADS='ALL_CPUs',
            GTIFF_FORCE_RGBA=False,
            CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=True,
        ):
            with rio.open(
                filename, 'w', driver=driver, **self.profile, **creation_options
            ) as ds:
                ds.write(self._array)

    def reproject(
        self,
        crs: CRS | None = None,
        transform: Affine | None = None,
        shape: tuple[int, int] | None = None,
        nodata: float | None = default_nodata,
        dtype: str = default_dtype,
        resampling: str | Resampling = Resampling.lanczos,
        **kwargs,
    ) -> 'RasterArray':
        """
        Reproject the RasterArray.

        :param crs:
            Destination CRS.  If ``None``, use the RasterArray :attr:`crs`.
        :param transform:
            Destination geo-referencing transform.  If supplied, ``shape`` is also
            required.  If ``None``, use the RasterArray :attr:`transform`.
        :param shape:
            Destination (row, col) shape.  If ``None``, use the RasterArray
            :attr:`shape`.
        :param nodata:
            Destination nodata value.
        :param dtype:
            Destination data type.
        :param resampling:
            Resampling method to use.
        :param kwargs:
            Additional keyword arguments to pass to :meth:`~rasterio.warp.reproject`.

        :return:
            Reprojected RasterArray.
        """
        if transform is not None and shape is None:
            raise ValueError("If 'transform' is supplied, 'shape' is also required.")
        if isinstance(resampling, str):
            resampling = Resampling[resampling.lower()]

        crs = crs or self._crs
        shape = shape or self.shape
        dtype = dtype or self.dtype

        fill_value = nodata if nodata is not None else 0
        if self.count > 1:
            shape = (self.count, *shape)
        dst_array = np.full(shape, fill_value=fill_value, dtype=dtype)

        _, dst_transform = reproject(
            self._array,
            destination=dst_array,
            src_crs=self._crs,
            src_transform=self._transform,
            src_nodata=self._nodata,
            dst_crs=crs,
            dst_transform=transform,
            dst_nodata=nodata,
            num_threads=multiprocessing.cpu_count(),
            resampling=resampling,
            **kwargs,
        )
        return RasterArray(dst_array, crs=crs, transform=dst_transform, nodata=nodata)
