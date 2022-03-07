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

import logging
import multiprocessing
from typing import Tuple

import numpy
import numpy as np
import rasterio as rio
import rasterio.windows
from homonim import utils
from homonim.errors import ImageProfileError, ImageFormatError
from rasterio import Affine
from rasterio import transform
from rasterio import windows
from rasterio.crs import CRS
from rasterio.enums import MaskFlags
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window

logger = logging.getLogger(__name__)


class RasterArray(transform.TransformMethodsMixin, windows.WindowMethodsMixin):
    """
    A class for encapsulating a masked, geo-referenced numpy array.
    Provides methods for re-projection, and reading/writing from/to rasterio datasets.
    """
    default_nodata = float('nan')  # default internal nodata value
    default_dtype = 'float32'  # default internal data type

    def __init__(self, array, crs, transform, nodata=default_nodata, window=None):
        """
        Construct a RasterArray.

        Parameters
        ----------
        array: numpy.ndarray
               A 2 or 3D array of image data, if 3D, bands are along the first dimension.
        crs: rasterio.crs.CRS
             The array CRS.
        transform: rasterio.transform.Affine
                   The array geo-transform.
        nodata: optional
                A number or nan, specifying the nodata value to use for masking the array.
        window: rasterio.windows.Window, optional
                An optional window into the transform specifying the array region.
        """

        if (array.ndim < 2) or (array.ndim > 3):
            raise ValueError("'array' must be have 2 or 3 dimensions with bands along the first dimension")
        self._array = array

        if window is not None and (window.height, window.width) != array.shape[-2:]:
            raise ValueError("'window' and 'array' width and height must match")

        if isinstance(crs, CRS):
            self._crs = crs
        else:
            raise TypeError("'crs' must be an instance of rasterio.crs.CRS")

        if isinstance(transform, Affine):
            if window is not None:
                self._transform = windows.transform(window, transform)
            else:
                self._transform = transform
        else:
            raise TypeError("'transform' must be an instance of rasterio.transform.Affine")

        self._nodata = nodata
        self._nodata_mask = None

    @classmethod
    def from_profile(cls, array, profile, window=None):
        """
        Construct a RasterArray from an array of image data and a profile dictionary.

        Parameters
        ----------
        array: numpy.ndarray, None
               A 2 or 3D array of image data, if 3D, bands are along the first dimension.
               Can be None, in which case a nodata array is created with the 'width', 'height', 'count', 'dtype' and
               'nodata' fields in profile.
        profile: dict
                 A configuration dictionary with items specifying the 'crs', 'transform' and 'nodata' values
                 (as used by rasterio datasets).  If 'array' is None, this dict should contain the additional fields
                 to create the array.
        window: rasterio.windows.Window, optional
                An optional window into profile['transform'], specifying the array region.

        Returns
        -------
        ra: RasterArray
            The constructed RasterArray.
        """
        if not {'crs', 'transform', 'nodata'} <= set(profile):
            raise ImageProfileError("'profile' should include 'crs', 'transform' and 'nodata' keys")

        if array is None:  # create array filled with nodata
            if not {'width', 'height', 'count', 'dtype'} <= set(profile):
                raise ImageProfileError("'profile' should include 'width', 'height', 'count' and 'dtype' keys")
            array_shape = (profile['count'], profile['height'], profile['width'])
            array = np.full(array_shape, fill_value=profile['nodata'], dtype=profile['dtype'])

        return cls(array, profile['crs'], profile['transform'], nodata=profile['nodata'], window=window)

    @classmethod
    def from_rio_dataset(cls, rio_dataset, indexes=None, window=None, **kwargs):
        """
        Construct a RasterArray by reading from a rasterio dataset.

        Implements 'boundless' reads internally which is faster than using rasterio's boundless=True option.

        Parameters
        ----------
        rio_dataset: rasterio.DatasetReader
                     The rasterio dataset to be read from.
        indexes: int, list[int], optional
                 The 1-based index or list of indexes of the bands to be read from 'rio_dataset'.
                 The default is to read all the 'rio_dataset' bands.
        window: rasterio.windows.Window, optional
                An optional window into 'rio_dataset' to be read from.
                This can be a 'boundless' window i.e. a window that extends beyond the bounds of 'rio_dataset',
                in which case the RasterArray.array will be filled with nodata outside of the the 'rio_dataset' bounds.
        kwargs: dict, optional
                Additional arguments to be passed to the dataset's read() method.

        Returns
        -------
        ra: RasterArray
            The constructed RasterArray.
        """
        # form a list of indexes
        if indexes is None:
            index_list = utils.get_nonalpha_bands(rio_dataset)
        else:
            index_list = [indexes] if np.isscalar(indexes) else indexes

        if window is None:
            # window of the full dataset extent
            window = Window(col_off=0, row_off=0, width=rio_dataset.width, height=rio_dataset.height)

        # check bands if bands have masks (i.e. internal/side-car mask or alpha channel), as opposed to nodata value
        is_masked = any([MaskFlags.per_dataset in rio_dataset.mask_flag_enums[bi - 1] for bi in index_list])

        # use the dataset's nodata value if it 'unmasked', and has one, otherwise revert to default
        nodata = cls.default_nodata if (is_masked or rio_dataset.nodata is None) else rio_dataset.nodata

        # construct an array of nodata matching the (possibly boundless) window dimension
        bounded_window, bounded_slices = cls.bounded_window_slices(rio_dataset, window)
        if len(index_list) > 1:
            array = np.full((len(index_list), window.height, window.width), fill_value=nodata, dtype=cls.default_dtype)
            bounded_array = array[(slice(array.shape[0]), *bounded_slices)]  # a bounded view into array
            rio_dataset.read(out=bounded_array, indexes=index_list, window=bounded_window,
                             out_dtype=cls.default_dtype, **kwargs)
        else:
            array = np.full((window.height, window.width), fill_value=nodata, dtype=cls.default_dtype)
            bounded_array = array[bounded_slices]  # a bounded view into array
            rio_dataset.read(out=bounded_array, indexes=index_list[0], window=bounded_window,
                             out_dtype=cls.default_dtype, **kwargs)

        # read into the bounded section of the array

        if is_masked:
            # read the mask from dataset and apply it to the array
            bounded_mask = rio_dataset.dataset_mask(window=bounded_window).astype('bool', copy=False)
            if bounded_array.ndim == 2:
                bounded_array[~bounded_mask] = nodata
            else:
                bounded_array[:, ~bounded_mask] = nodata

        return cls(array, rio_dataset.crs, rio_dataset.transform, nodata=nodata, window=window)

    @staticmethod
    def bounded_window_slices(rio_dataset, window):
        """ Bounded array slices and dataset window from dataset and boundless window """

        # find window UL and BR corners and crop to rio_dataset bounds
        win_ul = np.array((window.row_off, window.col_off))
        win_br = win_ul + np.array((window.height, window.width))
        bounded_ul = np.fmax(win_ul, (0, 0))
        bounded_br = np.fmin(win_br, rio_dataset.shape)

        # create bounded window and slices from bounded corners
        bounded_window = Window.from_slices((bounded_ul[0], bounded_br[0]), (bounded_ul[1], bounded_br[1]))
        bounded_start = bounded_ul - win_ul
        bounded_stop = bounded_start + (bounded_br - bounded_ul)
        bounded_slices = (slice(bounded_start[0], bounded_stop[0], None),
                          slice(bounded_start[1], bounded_stop[1], None))
        return bounded_window, bounded_slices

    @property
    def array(self) -> numpy.ndarray:
        """A 2 or 3D array of image data, if 3D, bands are along the first dimension."""
        return self._array

    @array.setter
    def array(self, value: numpy.ndarray):
        if np.all(value.shape[-2:] == self._array.shape[-2:]):
            self._array = value
        else:
            raise ValueError("'value' and 'array' shapes must match")

    @property
    def crs(self) -> rasterio.crs.CRS:
        """The coordinate reference system."""
        return self._crs

    @property
    def width(self) -> int:
        """The array width in pixels."""
        return self.shape[-1]

    @property
    def height(self) -> int:
        """The array height in pixels."""
        return self.shape[-2]

    @property
    def shape(self) -> Tuple[int, int]:
        """The array shape (height, width) in pixels."""
        return tuple(self._array.shape[-2:])

    @property
    def count(self) -> int:
        """The number of bands."""
        return self._array.shape[0] if self.array.ndim == 3 else 1

    @property
    def dtype(self) -> str:
        """The internal data type of the image data."""
        return self._array.dtype.name

    @property
    def transform(self) -> rasterio.transform.Affine:
        """An affine geo-transform describing the location and orientation of the array in the CRS."""
        return self._transform

    @property
    def res(self) -> Tuple[float, float]:
        """Array (x, y) resolution (m)."""
        return (self._transform.a, -self._transform.e)

    @property
    def bounds(self) -> Tuple[float,]:
        """The (left, bottom, right, top) co-ordinates of the array extent."""
        return windows.bounds(windows.Window(0, 0, self.width, self.height), self._transform)

    @property
    def profile(self) -> dict:
        """The RasterArray properties formatted as a dictionary, compatible with rasterio."""
        return dict(crs=self._crs, transform=self._transform, nodata=self._nodata, count=self.count,
                    width=self.width, height=self.height, dtype=self.dtype)

    @property
    def proj_profile(self) -> dict:
        """
        RasterArray properties relevant to re-projection (i.e. 'crs', 'transform' and 'shape') formatted as a
        dictionary.
        Useful for expanding to keyword arguments to reproject().
        """
        return dict(crs=self._crs, transform=self._transform, shape=self.shape)

    @property
    def mask(self) -> numpy.ndarray:
        """A 2D boolean mask corresponding to valid pixels in the array."""
        if self._nodata is None:
            return np.full(self._array.shape[-2:], True)
        mask = ~utils.nan_equals(self._array, self._nodata)
        if self._array.ndim > 2:
            mask = np.any(mask, axis=0)
        return mask

    @mask.setter
    def mask(self, value: numpy.ndarray):
        if self._array.ndim == 2:
            self._array[~value] = self._nodata
        else:
            self._array[:, ~value] = self._nodata

    @property
    def mask_ra(self) -> 'RasterArray':
        """
        A RasterArray containing the 2D mask as 'uint8' data type, and with nodata=None.  Useful for re-projecting
        the mask.
        """
        mask = self.mask.astype('uint8', copy=False)
        return RasterArray(mask, crs=self._crs, transform=self._transform, nodata=None)

    @property
    def nodata(self) -> float:
        """The nodata value."""
        return self._nodata

    @nodata.setter
    def nodata(self, value: float):
        if value is None or self._nodata is None:
            # if new nodata value is None, remove the current mask
            # if current nodata is None, there is no mask, so just set the new nodata value and return
            self._nodata = value
        elif not (utils.nan_equals(value, self._nodata)):
            # if the new nodata value is different to the current nodata,
            # set the mask area in array to the new nodata value and return
            nodata_mask = ~self.mask
            if self._array.ndim == 3:
                self._array[:, nodata_mask] = value
            else:
                self._array[nodata_mask] = value
            self._nodata = value

    def copy(self):
        """Create a deep copy of the RasterArray."""
        return RasterArray.from_profile(self._array.copy(), self.profile)

    def slice_to_bounds(self, *bounds):
        """
        Create a new RasterArray representing a rectangular sub-region of this RasterArray.
        Note that the created RasterArray is a view into the current array, not a copy.

        Parameters
        ----------
        bounds: Tuple
                The co-ordinate bounds to slice the new array to (left, bottom, right, top), in the current CRS.
        Returns
        -------
        ra: RasterArray
            The sliced RasterArray.
        """
        window = self.window(*bounds)
        window = utils.round_window_to_grid(window)
        ul = np.array((window.row_off, window.col_off))
        shape = np.array((window.height, window.width))
        if np.any(ul < 0) or np.any(shape > self._array.shape[-2:]):
            raise ValueError(f'The provided bounds ({bounds}) lie outside the extent of the RasterArray '
                             f'({self.bounds})')

        if self._array.ndim == 2:
            array = self._array[window.toslices()]
        else:
            array = self._array[(slice(self._array.shape[0]), *window.toslices())]

        return RasterArray(array, self._crs, self.window_transform(window), nodata=self._nodata)

    def to_rio_dataset(self, rio_dataset, indexes=None, window=None, **kwargs):
        """
        Write the RasterArray into a rasterio dataset.

        Note that typically, the dataset bounds would encompass the RasterArray bounds.

        Parameters
        ----------
        rio_dataset: rasterio.io.DatasetWriter
                     An open rasterio dataset into which to write the RasterArray.
                     The dataset must CRS must match that of the RasterArray.
        indexes: int, list[int], optional
                 The 1-based index or list of indexes of the bands to be written in 'rio_dataset'.
                 It should contain the same number of items as there are RasterArray bands.
                 [default: write into the first 'count' non-alpha bands of the dataset, where 'count' is the number
                 of RasterArray bands.]
        window: rasterio.windows.Window, optional
                A window defining the region in the dataset to write the RasterArray to, and how to crop the
                RasterArray, if necessary.  If it is a 'boundless' window i.e. extended beyond the bounds of the
                dataset, it is cropped to fit the bounds of the dataset. The RasterArray is cropped to fit the bounds
                of the window in the dataset.
                [default: write the full extent of RasterArray into the corresponding region in the dataset.]
        kwargs: optional
                Arguments to passed through the dataset's write() method.
        """
        if not np.all(self.res == rio_dataset.res):
            raise ImageFormatError(f"The dataset resolution does not match that of the RasterArray. "
                                   f"Dataset res: {rio_dataset.res}, RasterArray res: {self.res}")
        # TODO: raise an issue with rasterio about the speed of crs comparison.  comparing the _crs attr is faster.
        if self.crs._crs != rio_dataset.crs._crs:
            raise ImageFormatError(f"The dataset CRS does not match that of the RasterArray. "
                                   f"Dataset CRS: {rio_dataset.crs.to_proj4()}, RasterArray CRS: {self.crs.to_proj4()}")

        if indexes is None:
            indexes = utils.get_nonalpha_bands(rio_dataset)
            indexes = indexes if len(indexes) > 1 else indexes[0]

        if np.any((np.array(indexes) < 1) | (np.array(indexes) > rio_dataset.count)):
            error_indexes = np.array(indexes)[np.array(indexes) > rio_dataset.count]
            raise ValueError(f'Band index(es) {error_indexes} are out of the valid range (1..{rio_dataset.count})')

        if (not np.isscalar(indexes)) and (len(indexes) > self.count):
            raise ValueError(f'The length of indexes ({len(indexes)}) exceeds the number of bands in the '
                             f'RasterArray ({self.count})')

        if window is None:
            # a window defining the region in the dataset corresponding to the RasterArray extents
            window = rio_dataset.window(*self.bounds)

        # crop the window to dataset bounds
        window, _ = self.bounded_window_slices(rio_dataset, window)
        # crop the RasterArray to match the bounds of the the dataset window
        bounded_ra = self.slice_to_bounds(*rio_dataset.window_bounds(window))

        if np.any(bounded_ra.shape != np.array((window.height, window.width))):
            raise ValueError(
                f'The bounds of the dataset / window ({rio_dataset.window_bounds(window)}) lie outside the '
                f'bounds of the RasterArray ({bounded_ra.bounds})')

        rio_dataset.write(bounded_ra.array, window=window, indexes=indexes, **kwargs)

    def to_file(self, filename, driver='GTiff', **kwargs):
        """
        Write the RasterArray to an image file.

        Parameters
        ----------
        filename: str, pathlib.Path
                  The name of the file to create.
        driver: str, optional
                A valid rasterio short format driver name - see http://www.gdal.org/formats_list.html.
        kwargs: dict, optional
                Driver specific creation options e.g. compression='deflate' for a geotiff.
                See the rasterio documentation for more details:
        """
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            with rio.open(filename, 'w', driver=driver, **self.profile, **kwargs) as out_im:
                out_im.write(self._array, indexes=range(1, self.count + 1) if self.count > 1 else 1)

    def reproject(self, crs=None, transform=None, shape=None, nodata=default_nodata, dtype=default_dtype,
                  resampling=Resampling.lanczos, **kwargs):
        """
        Re-project the RasterArray.

        Parameters
        ----------
        crs: rasterio.crs.CRS, optional
             The CRS to project into.  [default: use the CRS of the this RasterArray]
        transform: rasterio.transform.Affine, optional
                    The geo-transform to project into.  If 'transform' is specified, 'shape' is also required.
                    [default: use the transform of this RasterArray]
        shape: tuple, optional
               The (rows, columns) size of the destination array. [default: use the shape of this RasterArray]
        nodata: float, int, optional
                The nodata value of the destination array.  [default: use the nodata value of this RasterArray]
        dtype: type, str, optional
                The data type of destination array.  [default: use the data type of this RasterArray]
        resampling: rasterio.enums.Resampling, optional
                    Resampling method to use.  [default: Resampling.lanczos]
        kwargs: dict, optional
                Arguments to passed through the rasterio's reproject() function.

        Returns
        -------
        ra: RasterArray
            The reprojected RasterArray.
        """

        if transform is not None and shape is None:
            raise ValueError('If "transform" is specified, "shape" must also be specified')

        if isinstance(resampling, str):
            resampling = Resampling[resampling]

        crs = crs or self.crs
        shape = shape or self.shape
        dtype = dtype or self.dtype

        if self.array.ndim > 2:
            _dst_array = np.zeros((self._array.shape[0], *shape), dtype=dtype)
        else:
            _dst_array = np.zeros(shape, dtype=dtype)

        _, _dst_transform = reproject(
            self._array,
            destination=_dst_array,
            src_crs=self._crs,
            src_transform=self._transform,
            src_nodata=self._nodata,
            dst_crs=crs,
            dst_transform=transform,
            dst_nodata=nodata,
            num_threads=multiprocessing.cpu_count(),
            resampling=resampling,
            **kwargs
        )
        return RasterArray(_dst_array, crs=crs, transform=_dst_transform, nodata=nodata)

##
