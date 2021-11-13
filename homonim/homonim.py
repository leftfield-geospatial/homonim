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
import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import rasterio as rio
# from rasterio import transform
from rasterio.warp import reproject, Resampling, transform_geom, transform_bounds, calculate_default_transform
from rasterio.fill import fillnodata
from rasterio.features import dataset_features
from enum import Enum
import pathlib
from shapely.geometry import box, shape
from homonim import get_logger
import multiprocessing
import cv2 as cv
from scipy.optimize import lsq_linear
from collections import namedtuple
import datetime

import concurrent.futures
import multiprocessing
import threading
import click
from tqdm import tqdm

logger = get_logger(__name__)

class Model(Enum):
    GAIN_ONLY = 1
    OFFSET_ONLY = 2
    GAIN_AND_OFFSET = 3
    GAIN_AND_IMAGE_OFFSET = 4
    OFFSET_AND_IMAGE_GAIN = 5
    GAIN_ZERO_MEAN_OFFSET = 6


def expand_window_to_grid(win):
    """
    Expands float window extents to be integers that include the original extents

    Parameters
    ----------
    win : rasterio.windows.Window
        the window to expand

    Returns
    -------
    exp_win: rasterio.windows.Window
        the expanded window
    """
    col_off, col_frac = np.divmod(win.col_off, 1)
    row_off, row_frac = np.divmod(win.row_off, 1)
    width = np.ceil(win.width + col_frac)
    height = np.ceil(win.height + row_frac)
    exp_win = rio.windows.Window(col_off, row_off, width, height)
    return exp_win

"""Container for raster properties relevant to re-projection"""
RasterProps = namedtuple('RasterProps', ['crs', 'transform', 'shape', 'res', 'bounds', 'nodata', 'count', 'profile'])

"""Internal nodata value"""
int_nodata = 0

"""Internal type for homogenisation data"""
int_dtype = np.float32


class HomonImBase:
    def __init__(self, src_filename, ref_filename, win_size=[3, 3], homo_config=None, out_config=None):
        """
        Class for homogenising images, model found in reference space as per original method

        Parameters
        ----------
        src_filename : str, pathlib.Path
            Source image filename
        ref_filename: str, pathlib.Path
            Reference image filename
        homo_config: dict
            Dictionary for advanced homogenisation configuration
        out_config: dict
            Dictionary for configuring output file format
        """
        # TODO: refactor which parameters get passed here, and which to homogenise()
        self._src_filename = pathlib.Path(src_filename)
        self._ref_filename = pathlib.Path(ref_filename)
        self._check_rasters()
        if not np.all(np.mod(win_size, 2) == 1):
            raise Exception('win_size must be odd in both dimensions')
        self.win_size = win_size

        self._src_props = None
        self._ref_props = None
        self._ref_array = None

        if homo_config is None:
            self._homo_config = {
                'src2ref_interp': 'cubic_spline',
                'ref2src_interp': 'average',
                'debug': False,
                'mask_partial_pixel': True,
                'mask_partial_window': False,
                'mask_partial_interp': False,
                'multithread': True,
            }
        else:
            self._homo_config = homo_config

        if out_config is None:
            self._out_config = {
                'driver': 'GTiff',
                'dtype': 'float32',
                'tile_size': [512, 512],
                'compress': 'deflate',
                'interleave': 'band',
                'photometric': None,
                'nodata': 0
            }
        else:
            self._out_config = out_config


    @property
    def ref_array(self):
        if self._ref_array is None:
            self._ref_array = self._read_ref()
        return self._ref_array

    @property
    def src_props(self):
        if self._src_props is None:
            self._read_ref()
        return self._src_props

    @property
    def ref_props(self):
        if self._ref_props is None:
            self._read_ref()
        return self._ref_props


    def _check_rasters(self):
        """Check bounds, band count, and compression type of source and reference images"""
        if not self._src_filename.exists():
            raise Exception(f'Source file {self._src_filename.stem} does not exist' )
        if not self._ref_filename.exists():
            raise Exception(f'Reference file {self._ref_filename.stem} does not exist' )

        # check we can read the images
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            for filename in [self._src_filename, self._ref_filename]:
                with rio.open(filename, 'r') as im:
                    try:
                        tmp_array = im.read(1, window=im.block_window(1, 0, 0))
                    except Exception as ex:
                        if im.profile['compress'] == 'jpeg':  # assume it is a 12bit JPEG
                            raise Exception(f'Could not read {filename.stem}\n'
                                            f'    This GDAL package does not support JPEG compression with NBITS==12\n'
                                            f'    you probably need to recompress this file.\n'
                                            f'    See the README for details.')
                        else:
                            raise ex

            with rio.open(self._src_filename, 'r') as src_im, rio.open(self._ref_filename, 'r') as ref_im:
                # check reference covers source
                src_box = box(*src_im.bounds)
                ref_box = box(*ref_im.bounds)

                # reproject the reference bounds if necessary
                if src_im.crs.to_proj4() != ref_im.crs.to_proj4():    # CRS's don't match
                    ref_box = shape(transform_geom(ref_im.crs, src_im.crs, ref_box))

                if not ref_box.covers(src_box):
                    raise Exception(f'Reference image {self._ref_filename.stem} does not cover source image '
                                    f'{self._src_filename.stem}.')

                # check reference has enough bands for the source
                if src_im.count > ref_im.count:
                    raise Exception(f'Reference image {self._ref_filename.stem} has fewer bands than source image '
                                    f'{self._src_filename.stem}.')

                # if the band counts don't match assume the first src_im.count bands of ref_im match those of src_im
                if src_im.count != ref_im.count:
                    logger.warning('Reference bands do not match source bands.  \n'
                                   f'Using the first {src_im.count} bands of reference image  {self._ref_filename.stem}.')

                for fn, nodata in zip([self._src_filename, self._ref_filename], [src_im.nodata, ref_im.nodata]):
                    if nodata is None:
                        logger.warning(f'{fn} has no nodata value, defaulting to {int_nodata}.\n'
                                       'Any invalid pixels in this image should be first be masked with nodata.')
                # if src_im.crs != ref_im.crs:    # CRS's don't match
                #     logger.warning('The reference will be re-projected to the source CRS.  \n'
                #                    'To avoid this step, provide a reference image in the source CRS')

    def _read_ref(self):
        """
        Read the source region from the reference image in the source CRS
        """
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._src_filename, 'r') as src_im:
            with rio.open(self._ref_filename, 'r') as ref_im:
                # find source source bounds in ref CRS  (possibly same CRS)
                ref_src_bounds = transform_bounds(src_im.crs, ref_im.crs, *src_im.bounds)

                # find ref window that covers source and is aligned with ref grid
                ref_src_win = rio.windows.from_bounds(*ref_src_bounds, transform=ref_im.transform)
                ref_src_win = expand_window_to_grid(ref_src_win)

                # transform and bounds of ref_src_win in ref CRS
                ref_src_transform = rio.windows.transform(ref_src_win, ref_im.transform)
                ref_src_bounds = rio.windows.bounds(ref_src_win, ref_im.transform)

                # TODO: consider reading in tiles for large ref ims
                # TODO: work in a particular dtype, or transform to a particular dtype here/somewhere?
                # TODO: how to deal with pixel alignment i.e. if ref_win is float below,
                #   including resampling below with a non-NN option, does resample to the float window offsets
                #   but should we not rather resample the source to the ref grid than sort of the other way around

                ref_bands = range(1, src_im.count + 1)
                _ref_array = ref_im.read(ref_bands, window=ref_src_win).astype(int_dtype)

                # check for for valid pixel values
                ref_mask = ref_im.read_masks(ref_bands, window=ref_src_win).astype('bool')
                if np.any(_ref_array[ref_mask] <= 0):
                    raise Exception(f'Reference image {self._ref_filename.name} contains pixel values <=0.')

                if src_im.crs.to_proj4() != ref_im.crs.to_proj4():    # re-project the reference image to source CRS
                    # TODO: here we reproject from transform on the ref grid and that include source bounds
                    #   we could just project into src_im transform though too.
                    logger.warning('Reprojecting reference image to the source CRS. '
                                   'To avoid this step, provide reference and source images in the same CRS')

                    # transform and dimensions of reprojected ref ROI
                    ref_transform, width, height = calculate_default_transform(ref_im.crs, src_im.crs, ref_src_win.width,
                                                                           ref_src_win.height, *ref_src_bounds)
                    ref_array = np.zeros((src_im.count, height, width), dtype=int_dtype)
                    # TODO: another thing to think of is that we ideally shouldn't reproject the reference unnecessarily to avoid damaging radiometric info
                    #  this could mean we reproject the source to/from ref CRS rather than just resample it
                    _, xform = reproject(_ref_array, ref_array, src_transform=ref_src_transform, src_crs=ref_im.crs,
                        dst_transform=ref_transform, dst_crs=src_im.crs, resampling=Resampling.bilinear,
                              num_threads=multiprocessing.cpu_count(), dst_nodata=int_nodata)
                else:
                    # TODO: can we allow reference nodata covering the source area or should we throw an exception if this is the case
                    # replace reference nodata with internal nodata value
                    if (ref_im.nodata is not None) and (ref_im.nodata != int_nodata):
                        _ref_array[_ref_array == ref_im.nodata] = int_nodata
                    ref_transform = ref_src_transform
                    ref_array = _ref_array
                # create a profile dict for ref_array
                ref_profile = ref_im.profile.copy()
                ref_profile['transform'] = ref_transform
                ref_profile['crs'] = src_im.crs
                ref_profile['width'] = ref_array.shape[2]
                ref_profile['height'] = ref_array.shape[1]
                ref_nodata = int_nodata if ref_im.nodata is None else ref_im.nodata
                src_nodata = int_nodata if src_im.nodata is None else src_im.nodata

                self._ref_props = RasterProps(crs=src_im.crs, transform=ref_transform, shape=ref_array.shape[1:],
                                              res=list(ref_im.res), bounds=ref_src_bounds, nodata=ref_nodata,
                                              count=ref_im.count, profile=ref_profile)
                self._src_props = RasterProps(crs=src_im.crs, transform=src_im.transform, shape=src_im.shape,
                                              res=list(src_im.res), bounds=src_im.bounds, nodata=src_nodata,
                                              count=src_im.count, profile=src_im.profile)

        return ref_array

    def _project_src_to_ref(self, src_array, src_nodata=int_nodata, dst_dtype=int_dtype, dst_nodata=int_nodata, resampling=None):
        """
        Re-project an array from source to reference CRS

        Parameters
        ----------
        src_array: numpy.array_like
                   Source raster array
        src_nodata: int, float, optional
                    Nodata value for src_array (Default: 0)
        dst_dtype: str, type, optional
                    Data type for re-projected array. (Default: float32)
        dst_nodata: int, float, optional
                    Nodata value for re-projected array (Default: 0)

        Returns
        -------
        : numpy.array_like
          Re-projected array
        """
        if src_array.ndim > 2:
            ref_array = np.zeros((src_array.shape[0], *self.ref_props.shape), dtype=dst_dtype)
        else:
            ref_array = np.zeros(self.ref_props.shape, dtype=dst_dtype)

        if resampling is None:
            resampling = Resampling[self._homo_config['src2ref_interp']]

        # TODO: checks on nodata vals, that a dest_nodata==0 will not conflict with pixel values
        _, _ = reproject(
            src_array,
            destination=ref_array,
            src_transform=self.src_props.transform,
            src_crs=self.src_props.crs,
            dst_transform=self.ref_props.transform,
            dst_crs=self.src_props.crs,
            resampling=resampling,
            num_threads=multiprocessing.cpu_count(),
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
        )
        return ref_array

    def _project_ref_to_src(self, ref_array, ref_nodata=int_nodata, dst_dtype=int_dtype, dst_nodata=int_nodata, resampling=None):
        """
        Re-project an array from reference to source CRS

        Parameters
        ----------
        ref_array: numpy.array_like
                   Reference CRS raster array to re-project
        ref_nodata: int, float, optional
                    Nodata value for ref_array (Default: 0)
        dst_dtype: str, type, optional
                    Data type for re-projected array. (Default: float32)
        dst_nodata: int, float, optional
                    Nodata value for re-projected array (Default: 0)

        Returns
        -------
        : numpy.array_like
          Re-projected array
        """
        if ref_array.ndim > 2:
            src_array = np.zeros((ref_array.shape[0], *self.src_props.shape), dtype=dst_dtype)
        else:
            src_array = np.zeros(self.src_props.shape, dtype=dst_dtype)

        if resampling is None:
            resampling = Resampling[self._homo_config['ref2src_interp']]

        # TODO: checks on nodata vals, that a dest_nodata==0 will not conflict with pixel values
        _, _ = reproject(
            ref_array,
            destination=src_array,
            src_transform=self.ref_props.transform,
            src_crs=self.src_props.crs,
            dst_transform=self.src_props.transform,
            dst_crs=self.src_props.crs,
            resampling=resampling,
            num_threads=multiprocessing.cpu_count(),
            src_nodata=ref_nodata,
            dst_nodata=dst_nodata,
        )
        return src_array


    def _sliding_window_view(self, x):
        """
        Return a 3D strided view of 2D array to allow fast sliding window operations.
        Rolling windows are stacked along the third dimension.  No data copying is involved.

        Parameters
        ----------
        x : numpy.array_like
            array to return view of

        Returns
        -------
        3D rolling window view of x
        """
        xstep = 1
        shape = x.shape[:-1] + (self.win_size[0], int(1 + (x.shape[-1] - self.win_size[0]) / xstep))
        strides = x.strides[:-1] + (x.strides[-1], xstep * x.strides[-1])
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)


    def _find_gains_int_arr(self, ref_array, src_array, win_size=[3,3], param_nodata=np.nan):
        """
        Find sliding window gains for a band using integral arrays

        Parameters
        ----------
        ref_array : numpy.array_like
            a reference band in an MxN array
        src_array : numpy.array_like
            a source band, collocated with ref_array and of the same MxN shape, and with nodata==0
        win_size : numpy.array_like
            sliding window [width, height] in pixels

        Returns
        -------
        param_array : numpy.array_like
            an M x N array of gains
        """

        # find ratios avoiding divide by nodata=0
        src_nodata = int_nodata
        src_mask = (src_array != src_nodata).astype(np.int32)
        ratio_array = np.zeros_like(src_array, dtype=int_dtype)
        _ = np.divide(ref_array, src_array, out=ratio_array, where=src_mask.astype('bool', copy=False))

        # find the integral arrays for ratios and mask (prepend a row and column of zeros)
        int_ratio_array = np.zeros(np.array(ratio_array.shape)+1, dtype=int_dtype)
        int_ratio_array[1:, 1:] = ratio_array
        int_mask_array = np.zeros(np.array(ratio_array.shape)+1, dtype=np.int32)
        int_mask_array[1:, 1:] = src_mask
        for i in range(2):
            int_ratio_array = int_ratio_array.cumsum(axis=i)
            int_mask_array = int_mask_array.cumsum(axis=i)

        # initialise the sliding window operation
        win_offset = np.floor(np.array(win_size)/2).astype(np.int32)      # the window center
        param_array = np.empty_like(ref_array, dtype=int_dtype)
        param_array[:] = param_nodata

        # TODO: implement this in cython
        # TODO: compare speed to a similar algorithm with cv.filter2d (i.e. filter2d on mask and ratio)
        # find sliding window gains using integral arrays
        for i in range(0, int_ratio_array.shape[0] - win_size[0]):
            i_bottom = i + win_size[0]
            for j in range(0, int_ratio_array.shape[1] - win_size[1]):
                j_right = j + win_size[1]
                sum_ratio = int_ratio_array[i_bottom, j_right] - int_ratio_array[i, j_right] \
                            - int_ratio_array[i_bottom, j] + int_ratio_array[i, j]
                sum_mask = int_mask_array[i_bottom, j_right] - int_mask_array[i, j_right] \
                            - int_mask_array[i_bottom, j] + int_mask_array[i, j]
                param_array[i + win_offset[0], j + win_offset[1]] = sum_ratio/sum_mask

        # param_array[np.isnan(src_array)] = param_nodata
        return param_array


    def _normalise_src(self, ref_array, src_array):
        """Normalise source in place"""
        src_mask = src_array != int_nodata
        # TODO: can we avoid these masked copies?
        _ref_array = ref_array[src_mask]
        _src_array = src_array[src_mask]
        norm_model = np.zeros(2)
        norm_model[0] = _ref_array.std() / _src_array.std()
        norm_model[1] = np.percentile(_ref_array, 1) - np.percentile(_src_array, 1) * norm_model[0]

        logger.info(f'Image normalisation gain / offset: {norm_model[0]:.4f} / {norm_model[1]:.4f}')
        src_array[src_mask] = norm_model[0]*_src_array + norm_model[1]
        return norm_model


    def _find_gains_cv(self, ref_array, src_array, win_size=[3,3]):
        """
        Find sliding window gains for a band using opencv filter2D

        Parameters
        ----------
        ref_array : numpy.array_like
            a reference band in an MxN array, with nodata==0
        src_array : numpy.array_like
            a source band, collocated with ref_array and of the same MxN shape, and with nodata==0
        win_size : numpy.array_like
            sliding window [width, height] in pixels
        normalise : bool
            Pre-process by applying an image wide linear model to remove any offset

        Returns
        -------
        param_array : numpy.array_like
            an M x N array of gains
        """

        mask = src_array != int_nodata    #.astype(np.int32)   # use int32 as it needs to accumulate sums below
        # find ratios avoiding divide by nodata=0
        ratio_array = np.full_like(src_array, fill_value=int_nodata, dtype=int_dtype)
        _ = np.divide(ref_array, src_array, out=ratio_array, where=mask.astype('bool', copy=False))

        # sum the ratio and mask over sliding windows (uses DFT for large kernels)
        kernel = np.ones(win_size, dtype=int_dtype)
        ratio_sum = cv.filter2D(ratio_array, -1, kernel, borderType=cv.BORDER_CONSTANT)
        mask_sum = cv.filter2D(mask.astype(int_dtype), -1, kernel, borderType=cv.BORDER_CONSTANT)

        # mask out parameter pixels that were not completely covered by a window of valid data
        # TODO: this interacts with mask_extrap, so rather do it in homogenise after --ref-space US?
        #   this is sort of the same question as, is it better to do pure extrapolation, or interpolation with semi-covered data?
        if self._homo_config['mask_partial_window']:
            mask &= (mask_sum >= kernel.size)

        # calculate gains, masking out invalid etc pixels
        param_array = np.full((1, *src_array.shape), fill_value=int_nodata, dtype=int_dtype)
        _ = np.divide(ratio_sum, mask_sum, out=param_array[0, :, :], where=mask.astype('bool', copy=False))

        # param_array[np.isnan(src_array)] = param_nodata
        return param_array


    def _find_gain_and_offset_cv_old(self, ref_array, src_array, win_size=[15, 15]):
        """
        Find the sliding window calibration parameters for a band.  OpenCV faster version.

        Parameters
        ----------
        ref_array : numpy.array_like
            a reference band in an MxN array
        src_array : numpy.array_like
            a source band, collocated with ref_array and of the same MxN shape
        win_size : numpy.array_like
            sliding window [width, height] in pixels

        Returns
        -------
        param_array : numpy.array_like
            an M x N  array of gains with nodata = nan
        """
        src_mask = src_array != int_nodata    #.astype(np.int32)   # use int32 as it needs to accumulate sums below
        ref_array[~src_mask] = int_nodata

        # sum the ratio and mask over sliding windows (uses DFT for large kernels)
        kernel = np.ones(win_size, dtype=int_dtype)
        src_sum = cv.filter2D(src_array, -1, kernel, borderType=cv.BORDER_CONSTANT)
        ref_sum = cv.filter2D(ref_array, -1, kernel, borderType=cv.BORDER_CONSTANT)
        mask_sum = cv.filter2D(src_mask.astype('uint8', copy=False), cv.CV_32F, kernel, borderType=cv.BORDER_CONSTANT)

        # TODO: do we need the where= clauses below?
        src_mean = np.zeros_like(src_array, dtype=int_dtype)
        _ = np.divide(src_sum, mask_sum, out=src_mean, where=src_mask.astype('bool', copy=False))
        ref_mean = np.zeros_like(ref_array, dtype=int_dtype)
        _ = np.divide(ref_sum, mask_sum, out=ref_mean, where=src_mask.astype('bool', copy=False))

        src2_sum = cv.filter2D(src_array**2, -1, kernel, borderType=cv.BORDER_CONSTANT)
        den_array = src2_sum - (2 * (src_mean) * (src_sum)) + (mask_sum * (src_mean**2))
        del(src2_sum)

        src_ref_sum = cv.filter2D(src_array*ref_array, -1, kernel, borderType=cv.BORDER_CONSTANT)
        num_array = src_ref_sum - (src_mean * ref_sum) - (ref_mean * src_sum) + (mask_sum * src_mean * ref_mean)
        del(src_ref_sum, src_sum, ref_sum, mask_sum)

        param_array = np.full((2, *src_array.shape), fill_value=int_nodata, dtype=int_dtype)
        _ = np.divide(num_array, den_array, out=param_array[0, :, :], where=src_mask.astype('bool', copy=False))
        _ = np.subtract(ref_mean, param_array[0, :, :] * src_mean, out=param_array[1, :, :], where=src_mask.astype('bool', copy=False))
        return param_array

    def _find_gain_and_offset_cv(self, ref_array, src_array, win_size=[15, 15]):
        """
        Find the sliding window calibration parameters for a band.  OpenCV faster version.

        Parameters
        ----------
        ref_array : numpy.array_like
            a reference band in an MxN array
        src_array : numpy.array_like
            a source band, collocated with ref_array and of the same MxN shape
        win_size : numpy.array_like
            sliding window [width, height] in pixels

        Returns
        -------
        param_array : numpy.array_like
            an M x N  array of gains with nodata = nan
        """
        mask = src_array != int_nodata   # use int32 as it needs to accumulate sums below
        ref_array[~mask] = int_nodata

        # sum the ratio and mask over sliding windows (uses DFT for large kernels)
        kernel = np.ones(win_size, dtype=int_dtype)
        src_sum = cv.filter2D(src_array, -1, kernel, borderType=cv.BORDER_CONSTANT)
        ref_sum = cv.filter2D(ref_array, -1, kernel, borderType=cv.BORDER_CONSTANT)
        src_ref_sum = cv.filter2D(src_array*ref_array, -1, kernel, borderType=cv.BORDER_CONSTANT)
        mask_sum = cv.filter2D(mask.astype(int_dtype), -1, kernel, borderType=cv.BORDER_CONSTANT)

        m_num_array = (mask_sum * src_ref_sum) - (src_sum * ref_sum)
        del(src_ref_sum)
        src2_sum = cv.filter2D(src_array**2, -1, kernel, borderType=cv.BORDER_CONSTANT)
        m_den_array = (mask_sum * src2_sum) - (src_sum ** 2)
        del(src2_sum)

        # mask out parameter pixels that were not completely covered by a window of valid data
        # TODO: this interacts with mask_extrap, so rather do it in homogenise after --ref-space US?
        #   this is sort of the same question as, is it better to do pure extrapolation, or interpolation with semi-covered data?
        if self._homo_config['mask_partial_window']:
            mask &= (mask_sum >= kernel.size)

        param_array = np.full((2, *src_array.shape), fill_value=int_nodata ,dtype=int_dtype)
        _ = np.divide(m_num_array, m_den_array, out=param_array[0, :, :], where=mask.astype('bool', copy=False))
        _ = np.divide(ref_sum - (param_array[0, :, :] * src_sum), mask_sum, out=param_array[1, :, :], where=mask.astype('bool', copy=False))

        return param_array


    def _find_gain_and_offset_winview(self, ref_array, src_array, win_size=[15, 15]):
        """
        Find the sliding window calibration parameters for a band.  Brute force.

        Parameters
        ----------
        ref_array : numpy.array_like
            a reference band in an MxN array
        src_array : numpy.array_like
            a source band, collocated with ref_array and of the same MxN shape
        win_size : numpy.array_like
            sliding window [width, height] in pixels

        Returns
        -------
        param_array : numpy.array_like
            an M x N  array of gains with nodata = nan
        """
        src_mask = (src_array != int_nodata)
        src_winview = sliding_window_view(src_array, win_size)
        ref_winview = sliding_window_view(ref_array, win_size)
        src_mask_winview = sliding_window_view(src_mask, win_size)

        param_array = np.zeros((2, *ref_array.shape), dtype=int_dtype)
        win_offset = np.floor(np.array(win_size) / 2).astype(np.int32)  # the window center
        a_const = np.ones((np.prod(win_size), 1))
        for win_i in np.ndindex(src_winview.shape[0]):
            for win_j in np.ndindex(src_winview.shape[1]):
                src_win = src_winview[win_i, win_j, :, :]
                ref_win = ref_winview[win_i, win_j, :, :]
                src_win_mask = src_mask_winview[win_i, win_j, :, :]
                src_win_v = src_win[src_win_mask].reshape(-1, 1)
                ref_win_v = ref_win[src_win_mask].reshape(-1, 1)
                src_const = np.ones(src_win_v.shape)

                src_a = np.hstack((src_win_v.reshape(-1, 1), src_const.reshape(-1, 1)))
                if True:
                    soln = np.linalg.lstsq(src_a, ref_win_v, rcond=None)
                    params = soln[0]
                    # r2 = 1 - (soln[1] / (ref_win_v.size * ref_win_v.var()))
                    # if poor model fit, find image-wide normlised gain-only model
                    if False and ((r2 < 0.01) or params[0] < 0):
                        soln_n = np.linalg.lstsq(np.dot(src_a, norm_model).reshape(-1, 1), ref_win_v, rcond=None)
                        params = (soln_n[0]*norm_model).T
                        if False:
                            from matplotlib import pyplot
                            pyplot.plot(src_win_v, ref_win_v, 'o')
                            x = np.arange(np.stack((src_win_v, ref_win_v)).min(), np.stack((src_win_v, ref_win_v)).max())
                            y_n = params[0] * x + params[1]
                            y = soln[0][0] * x + soln[0][1]
                            pyplot.plot(x, y, label='gain offset')
                            pyplot.plot(x, y_n, label='norm')
                            pyplot.legend()


                elif False:
                    soln = lsq_linear(src_a, ref_win_v.ravel(), bounds=[[0, -np.inf], [np.inf, np.inf]])
                    params = soln.x
                    r2 = 1 - (sum(soln.fun**2) / (ref_win_v.size * ref_win_v.var()))
                elif False:
                    # from https://towardsdatascience.com/linear-regression-using-least-squares-a4c3456e8570
                    params = np.zeros(2)
                    src_bar = src_win_v.mean()
                    delta_src_bar = src_win_v - src_bar
                    ref_bar = ref_win_v.mean()
                    params[0] = np.sum(delta_src_bar * (ref_win_v - ref_bar)) / np.sum(delta_src_bar**2)
                    params[1] = ref_bar - params[0] * src_bar

                    # r2 = 1 - (soln[1] / (ref_win_v.size * ref_win_v.var()))

                param_array[:, win_i + win_offset[0], win_j + win_offset[1]] = params.reshape(-1, 1)

        if False:   # inpaint negative gain areas
            gain_mask = param_array[0, :, :] >= 0
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            gain_mask = cv2.erode(gain_mask.astype('uint8'), se)
            valid_mask = src_mask & (param_array[0, :, :] != 0)
            gain_mask = gain_mask & valid_mask
            for bi in range(param_array.shape[0]):
                param_array[bi, :, :] = fillnodata(param_array[bi, :, :], gain_mask)

            param_array[:, np.logical_not(valid_mask)] = int_nodata
        param_array[:, np.logical_not(src_mask)] = int_nodata

        return param_array



    def _create_out_profile(self, init_profile):
        """Create a rasterio profile for the output raster based on a starting profile and configuration"""
        out_profile = init_profile.copy()
        for key, value in self._out_config.items():
            if value is not None:
                out_profile.update(**{key: value})
        out_profile.update(tiled=True)
        return out_profile

    def _create_param_profile(self, init_profile):
        """Create a rasterio profile for the debug parameter raster based on a starting profile and configuration"""
        param_profile = init_profile.copy()
        for key, value in self._out_config.items():
            if value is not None:
                param_profile.update(**{key: value})
        param_profile.update(dtype=int_dtype, count=self.src_props.count*2, nodata=int_nodata, tiled=True)
        return param_profile

    def _create_param_filename(self, filename):
        filename = pathlib.Path(filename)
        return filename.parent.joinpath(f'{filename.stem}_PARAMS.{filename.suffix}')

    def build_overviews(self, filename):
        """
        Builds internal overviews for an existing image.

        Parameters
        ----------
        filename: str, pathlib.Path
                  Path to the raster file to build overviews for
        """
        filename = pathlib.Path(filename)

        if not filename.exists():
            raise Exception(f'{filename} does not exist')
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'r+') as homo_im:
            homo_im.build_overviews([2, 4, 8, 16, 32], Resampling.average)


    def set_metadata(self, filename, **kwargs):
        """
        Copy geedim reference band metadata to a homogenised raster file

        Parameters
        ----------
        filename: str, pathlib.Path
                  Path to the raster file to copy metadata to
        """
        filename = pathlib.Path(filename)

        if not filename.exists():
            raise Exception(f'{filename} does not exist')

        with rio.open(self._ref_filename, 'r') as ref_im,rio.open(filename, 'r+') as homo_im:
            homo_im.update_tags(**kwargs)
            for bi in range(1, homo_im.count+1):
                ref_meta_dict = ref_im.tags(bi)
                homo_meta_dict = {k: v for k, v in ref_meta_dict.items() if k in ['ABBREV', 'ID', 'NAME']}
                homo_im.update_tags(bi, **homo_meta_dict)

    def _homogenise_array(self, ref_array, src_array, method='gain_only', normalise=False):
        raise NotImplementedError()

    def homogenise(self, out_filename, method='gain_only', normalise=False):
        """
        Perform homogenisation.

        Parameters
        ----------
        out_filename: str, pathlib.Path
                       Name of the raster file to save the homogenised image to
        method: str, optional
                Specify the homogenisation method: ['gain_only'|'gain_offset'].  (Default: 'gain_only')
        normalise: bool, optional
                   Perform image-wide normalisation prior to homogenisation.  (Default: False)
        """
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._src_filename, 'r') as src_im:
            # TODO nodata conversion to/from 0 to/from configured output format
            if self._homo_config['debug']:
                param_profile = self._create_param_profile(src_im.profile.copy())
                param_out_file_name = self._create_param_filename(out_filename)
                param_im = rio.open(param_out_file_name, 'w', **param_profile)

            out_profile = self._create_out_profile(src_im.profile)
            out_im = rio.open(out_filename, 'w', **out_profile)
            # process by band to limit memory usage
            bands = list(range(1, src_im.count + 1))
            bar = tqdm(total=len(bands)+1)
            bar.update(0)
            read_lock = threading.Lock()
            write_lock = threading.Lock()
            param_lock = threading.Lock()
            try:
                def process_band(bi):
                    with read_lock:
                        src_array = src_im.read(bi, out_dtype=int_dtype)  # NB bands along first dim

                    out_array, param_array = self._homogenise_array(self.ref_array[bi - 1, :, :], src_array, method=method, normalise=normalise)

                    # convert nodata if necessary
                    if out_im.nodata != int_nodata:
                        out_array[out_array==int_nodata] = out_im.nodata

                    if self._homo_config['debug']:
                        with param_lock:
                            param_im.write(param_array[0, :, :].astype(param_im.dtypes[bi - 1]), indexes=bi)
                            if param_array.shape[0] > 1:
                                _bi = bi + src_im.count
                                param_im.write(param_array[1, :, :].astype(param_im.dtypes[_bi - 1]), indexes=_bi)

                    with write_lock:
                        out_im.write(out_array.astype(out_im.dtypes[bi-1]), indexes=bi)
                        bar.update(1)  # update with progress increment

                if self._homo_config['multithread']:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                        executor.map(process_band, bands)
                else:
                    for bi in bands:
                        process_band(bi)
            finally:
                out_im.close()
                if self._homo_config['debug']:
                    param_im.close()
                bar.update(1)  # update with progress increment
                bar.close()

class HomonimRefSpace(HomonImBase):

    def _homogenise_array(self, ref_array, src_array, method='gain_only', normalise=False):
        src_ds_array = self._project_src_to_ref(src_array, src_nodata=self.src_props.nodata)

        # set partially covered pixels to nodata
        # TODO is this faster, or setting param_array[src_array == 0] = 0 below
        if self._homo_config['mask_partial_pixel']:
            mask = (src_array != self.src_props.nodata).astype('uint8')
            mask_ds_array = self._project_src_to_ref(mask, src_nodata=None)
            src_ds_array[np.logical_not(mask_ds_array == 1)] = int_nodata

        # find the calibration parameters for this band
        if normalise:
            norm_model = self._normalise_src(ref_array, src_ds_array)

        if method.lower() == 'gain_only':
            param_ds_array = self._find_gains_cv(ref_array, src_ds_array, win_size=self.win_size)
        else:
            param_ds_array = self._find_gain_and_offset_cv(ref_array, src_ds_array, win_size=self.win_size)

        # upsample the parameter array
        param_array = self._project_ref_to_src(param_ds_array)

        if self._homo_config['mask_partial_interp']:
            param_mask = (param_array[0, :, :] == int_nodata)
            kernel_size = np.ceil(np.divide(self.ref_props.res, self.src_props.res)).astype('int32')  # /2
            se = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(kernel_size))
            param_mask = cv2.dilate(param_mask.astype('uint8', copy=False), se).astype('bool', copy=False)
            param_array[:, param_mask] = int_nodata

        # apply the normalisation
        if normalise:
            out_array = param_array[0, :, :] * norm_model[0] * src_array
            out_array += param_array[0, :, :] * norm_model[1]
        else:
            out_array = param_array[0, :, :] * src_array

        if param_array.shape[0] > 1:
            out_array += param_array[1, :, :]

        return out_array, param_ds_array



##

class HomonimSrcSpace(HomonImBase):

    def _homogenise_array(self, ref_array, src_array, method='gain_only', normalise=False):
        if self.src_props.nodata != int_nodata:
            src_array[src_array == self.src_props.nodata] = int_nodata

        # upsample ref to source CRS and grid
        ref_us_array = self._project_ref_to_src(ref_array, ref_nodata=self.ref_props.nodata)

        if normalise:  # normalise source in place
            self._normalise_src(ref_us_array, src_array)
        else:
            logger.info(f'Processing band...')

        # find the calibration parameters for this band
        win_size = self.win_size * np.round(
            np.array(self.ref_props.res) / np.array(self.src_props.res)).astype(int)

        # TODO parallelise this, use opencv and or gpu, and or use integral image!
        if method.lower() == 'gain_only':
            param_array = self._find_gains_cv(ref_us_array, src_array, win_size=win_size)
        else:
            param_array = self._find_gain_and_offset_cv(ref_us_array, src_array, win_size=win_size)

        out_array = param_array[0, :, :] * src_array
        if param_array.shape[0] > 1:
            out_array += param_array[1, :, :]

        return out_array, param_array


##
