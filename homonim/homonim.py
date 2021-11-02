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


class HomonImBase:
    def __init__(self, src_filename, ref_filename, win_size=[3, 3], model=Model.GAIN_ONLY):
        """
        Class for homogenising images, model found in reference space as per original method

        Parameters
        ----------
        src_filename : str
            Source image filename
        ref_filename: str
            Reference image filename
        win_size: numpy.array_like
            (optional) Window size [height, width] in reference image pixels
        model : Model
            (optional) Model type
        """
        self._src_filename = pathlib.Path(src_filename)
        self._ref_filename = pathlib.Path(ref_filename)
        self._check_rasters()
        if not np.all(np.mod(win_size, 2) == 1):
            raise Exception('win_size must be odd in both dimensions')
        self.win_size = win_size

    def _check_rasters(self):
        """
        Check bounds, band count, and compression type of source and reference images
        """
        if not self._src_filename.exists():
            raise Exception(f'Source file {self._src_filename.stem} does not exist' )
        if not self._ref_filename.exists():
            raise Exception(f'Reference file {self._ref_filename.stem} does not exist' )

        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            # check we can read the images
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

            with rio.open(self._src_filename, 'r') as src_im:
                with rio.open(self._ref_filename, 'r') as ref_im:
                    # check reference covers source
                    src_box = box(*src_im.bounds)
                    ref_box = box(*ref_im.bounds)

                    # reproject the reference bounds if necessary
                    if src_im.crs != ref_im.crs:    # CRS's don't match
                        ref_box = shape(transform_geom(ref_im.crs, src_im.crs, ref_box))

                    if not ref_box.covers(src_box):
                        raise Exception(f'Reference image {self._ref_filename.stem} does not cover source image '
                                        f'{self._src_filename.stem}')

                    # check reference has enough bands for the source
                    if src_im.count > ref_im.count:
                        raise Exception(f'Reference image {self._ref_filename.stem} has fewer bands than source image '
                                        f'{self._src_filename.stem}')

                    # if the band counts don't match assume the first src_im.count bands of ref_im match those of src_im
                    if src_im.count != ref_im.count:
                        logger.warning('Reference bands do not match source bands.  \n'
                                       'Using the first {src_im.count} bands of reference image  {self._ref_filename.stem}')

                    # if src_im.crs != ref_im.crs:    # CRS's don't match
                    #     logger.warning('The reference will be re-projected to the source CRS.  \n'
                    #                    'To avoid this step, provide a reference image in the source CRS')

    def _read_ref(self):
        """
        Read the source region from the reference image in the source CRS
        """
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            with rio.open(self._src_filename, 'r') as src_im:
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
                    _ref_array = ref_im.read(ref_bands, window=ref_src_win).astype('float32')

                    if src_im.crs.to_proj4() != ref_im.crs.to_proj4():    # re-project the reference image to source CRS
                        # TODO: here we reproject from transform on the ref grid and that include source bounds
                        #   we could just project into src_im transform though too.
                        logger.warning('Reprojecting reference image to the source CRS. '
                                       'To avoid this step, provide reference and source images in the same CRS')

                        # transform and dimensions of reprojected ref ROI
                        ref_transform, width, height = calculate_default_transform(ref_im.crs, src_im.crs, ref_src_win.width,
                                                                               ref_src_win.height, *ref_src_bounds)
                        ref_array = np.zeros((src_im.count, height, width), dtype=np.float32)
                        # TODO: another thing to think of is that we ideally shouldn't reproject the reference unnecessarily to avoid damaging radiometric info
                        #  this could mean we reproject the source to/from ref CRS rather than just resample it
                        _, xform = reproject(_ref_array, ref_array, src_transform=ref_src_transform, src_crs=ref_im.crs,
                            dst_transform=ref_transform, dst_crs=src_im.crs, resampling=Resampling.bilinear,
                                  num_threads=multiprocessing.cpu_count())
                    else:
                        ref_transform = ref_src_transform
                        ref_array = _ref_array

                    ref_profile = ref_im.profile
                    ref_profile['transform'] = ref_transform
                    ref_profile['res'] = list(ref_im.res)    # TODO: find a better way of passing this
                    ref_profile['bounds'] = ref_src_bounds
                    ref_profile['crs'] = src_im.crs
                    ref_profile['width'] = ref_array.shape[2]
                    ref_profile['height'] = ref_array.shape[1]

        return ref_array, ref_profile

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
        src_nodata = 0
        src_mask = (src_array != src_nodata).astype(np.int32)
        ratio_array = np.zeros_like(src_array, dtype=np.float32)
        _ = np.divide(ref_array, src_array, out=ratio_array, where=src_mask.astype('bool', copy=False))

        # find the integral arrays for ratios and mask (prepend a row and column of zeros)
        int_ratio_array = np.zeros(np.array(ratio_array.shape)+1, dtype=np.float32)
        int_ratio_array[1:, 1:] = ratio_array
        int_mask_array = np.zeros(np.array(ratio_array.shape)+1, dtype=np.int32)
        int_mask_array[1:, 1:] = src_mask
        for i in range(2):
            int_ratio_array = int_ratio_array.cumsum(axis=i)
            int_mask_array = int_mask_array.cumsum(axis=i)

        # initialise the sliding window operation
        win_offset = np.floor(np.array(win_size)/2).astype(np.int32)      # the window center
        param_array = np.empty_like(ref_array, dtype=np.float32)
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

    def _find_gains_cv(self, ref_array, src_array, win_size=[3,3], normalise=False, param_nodata=np.nan):
        """
        Find sliding window gains for a band using opencv filter2D

        Parameters
        ----------
        ref_array : numpy.array_like
            a reference band in an MxN array
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

        src_nodata = 0
        src_mask = (src_array != src_nodata).astype(np.int32)   # TODO: why .astype(np.int32)?

        if normalise:    # find image-wide offset
            _src_array = src_array[src_mask.astype('bool', copy=False)].reshape(-1, 1)
            _src_array = np.hstack((_src_array, np.ones_like(_src_array)))
            _ref_array = ref_array[src_mask.astype('bool', copy=False)].flatten()
            if False:
                norm_model = np.linalg.lstsq(_src_array, _ref_array, rcond=None)[0]
            else:
                norm_model = np.zeros(2)
                norm_model[0] = _ref_array.std() / _src_array[:, 0].std()
                norm_model[1] = _ref_array.mean() - (_src_array[:, 0].mean() * norm_model[0])
                norm_model[1] = np.percentile(_ref_array, 1) - np.percentile((_src_array[:, 0] * norm_model[0]), 1)
            logger.info(f'Image normalisation gain / offset: {norm_model[0]:.4f} / {norm_model[1]:.4f}')
            src_array[src_mask.astype('bool', copy=False)] = np.dot(_src_array, norm_model)


        # find ratios avoiding divide by nodata=0
        ratio_array = np.zeros_like(src_array, dtype=np.float32)
        _ = np.divide(ref_array, src_array, out=ratio_array, where=src_mask.astype('bool', copy=False))

        # sum the ratio and mask over sliding windows (uses DFT for large kernels)
        kernel = np.ones(win_size, dtype=np.float32)
        ratio_winsums = cv.filter2D(ratio_array, -1, kernel, borderType=cv.BORDER_CONSTANT)
        mask_winsums = cv.filter2D(src_mask.astype('uint8', copy=False), cv.CV_32F, kernel, borderType=cv.BORDER_CONSTANT)

        # calculate gains, ignoring invalid pixels
        # cv.divide(ratio_winsums, mask_winsums, dst=param_array, dtype=np.float32)
        if normalise:
            param_array = np.zeros((2, *src_array.shape), dtype=np.float32)
            _ = np.divide(ratio_winsums, mask_winsums, out=param_array[0, :, :],
                          where=src_mask.astype('bool', copy=False))

            param_array[1, :, :] = param_array[0, :, :] * norm_model[1]
            param_array[0, :, :] *= norm_model[0]
        else:
            param_array = np.zeros((1, *src_array.shape), dtype=np.float32)
            _ = np.divide(ratio_winsums, mask_winsums, out=param_array[0, :, :],
                          where=src_mask.astype('bool', copy=False))

        # param_array[np.isnan(src_array)] = param_nodata
        return param_array

    def __find_gains_winview(self, ref_array, src_array, win_size=[3,3]):
        """
        Find the sliding window calibration parameters for a band

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
        ratio_array = np.zeros_like(src_array, dtype=np.float32)
        _ = np.divide(ref_array, src_array, out=ratio_array)  # find ratios once
        ratio_winview = sliding_window_view(ratio_array, win_size)  # apply the sliding window to the ratios

        param_array = np.empty_like(ref_array, dtype=np.float32)
        param_array[:] = np.nan
        win_offset = np.floor(np.array(win_size) / 2).astype(np.int32)  # the window center
        # TODO : how to do nodata masking with numpy.ma and masked arrays, nans, or what?
        _ = np.nanmean(ratio_winview, out=param_array[win_offset[0]:-win_offset[0], win_offset[1]:-win_offset[1]],
                       axis=(2, 3))  # find mean ratio over sliding windows

        # TODO : what is the most efficient way to iterate over these view arrays? np.nditer?
        #   or might we use a cython inner to speed things up?  See np.nditer docs
        #   is the above mean over sliding win views faster than the nested loop below
        # for win_i in np.ndindex(ratio_winview.shape[2]):
        #     for win_j in np.ndindex(ratio_winview.shape[3]):
        #         ratio_win = ratio_winview[:, :, win_i, win_j]
        #         param_array[win_i + win_offset[0], win_j + win_offset[1]] = np.mean(ratio_win)  # gain only
        param_array[np.isnan(src_array)] = np.nan
        return param_array

    def _find_gain_and_offset_winview(self, ref_array, src_array, win_size=[15, 15], normalise=False):
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
        src_nodata = 0
        src_mask = (src_array != src_nodata)

        if normalise:    # find image-wide offset
            _src_array = src_array[src_mask.astype('bool', copy=False)].reshape(-1, 1)
            _src_array = np.hstack((_src_array, np.ones_like(_src_array)))
            _ref_array = ref_array[src_mask.astype('bool', copy=False)].flatten()
            if False:
                norm_model = np.linalg.lstsq(_src_array, _ref_array, rcond=None)[0]
            else:
                norm_model = np.zeros(2)
                norm_model[0] = _ref_array.std() / _src_array[:, 0].std()
                # norm_model[1] = _ref_array.mean() - (_src_array[:, 0].mean() * norm_model[0])
                norm_model[1] = np.percentile(_ref_array, 1) - np.percentile((_src_array[:, 0] * norm_model[0]), 1)
            logger.info(f'Image normalisation gain / offset: {norm_model[0]:.4f} / {norm_model[1]:.4f}')

            # src_array[src_mask.astype('bool', copy=False)] = np.dot(_src_array, norm_model)
            if False:
                res = cv2.fitLine(np.array([_src_array[:, 0], _ref_array]).T, cv2.DIST_HUBER, 0, 1e-9, 1e-9)
                m_cv = res[1] / res[0]
                c_cv = res[3] - m_cv*res[2]
                from matplotlib import pyplot
                pyplot.figure()
                pyplot.plot(_src_array[:, 0], ref_array[src_mask.astype('bool', copy=False)].flatten(), '.')
                x = np.arange(_src_array[:, 0].min(), _src_array[:, 0].max())
                y = norm_model[0] * x + norm_model[1]
                m = ref_array[src_mask.astype('bool', copy=False)].flatten().std() / _src_array[:, 0].std()
                c = ref_array[src_mask.astype('bool', copy=False)].flatten().mean() - (_src_array[:, 0] * m).mean()
                y2 = m * x + c
                y3 = m_cv * x + c_cv
                pyplot.plot(x, y, label='ls')
                pyplot.plot(x, y2, label='norm')
                pyplot.plot(x, y3, label='cv')
                pyplot.legend()

        src_winview = sliding_window_view(src_array, win_size)
        ref_winview = sliding_window_view(ref_array, win_size)
        src_mask_winview = sliding_window_view(src_mask, win_size)

        param_array = np.zeros((2, *ref_array.shape), dtype=np.float32)
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
                    r2 = 1 - (soln[1] / (ref_win_v.size * ref_win_v.var()))
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


                else:
                    soln = lsq_linear(src_a, ref_win_v.ravel(), bounds=[[0, -np.inf], [np.inf, np.inf]])
                    params = soln.x
                    r2 = 1 - (sum(soln.fun**2) / (ref_win_v.size * ref_win_v.var()))

                    # r2 = 1 - (soln[1] / (ref_win_v.size * ref_win_v.var()))

                param_array[:, win_i + win_offset[0], win_j + win_offset[1]] = params.reshape(-1, 1)

        if normalise:   # inpaint negative gain areas
            gain_mask = param_array[0, :, :] >= 0
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            gain_mask = cv2.erode(gain_mask.astype('uint8'), se)
            valid_mask = src_mask & (param_array[0, :, :] != 0)
            gain_mask = gain_mask & valid_mask
            for bi in range(param_array.shape[0]):
                param_array[bi, :, :] = fillnodata(param_array[bi, :, :], gain_mask)

            param_array[:, np.logical_not(valid_mask)] = 0
        else:
            param_array[:, np.logical_not(src_mask)] = src_nodata

        return param_array

    def homogenise(self, out_filename, method=None, normalise=False):
        raise NotImplementedError()

    def build_overviews(self, out_filename):
        """
        Builds internal overviews for an existing image
        """
        if out_filename.exists():
            with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
                with rio.open(out_filename, 'r+', num_threads='all_cpus') as homo_im:
                    homo_im.build_overviews([2, 4, 8, 16, 32], Resampling.average)

    def copy_ref_metadata(self, out_filename):
        """
        Populate metadata
        """
        if out_filename.exists():
            with rio.open(self._ref_filename, 'r') as ref_im:
                with rio.open(out_filename, 'r+', num_threads='all_cpus') as homo_im:
                    for bi in range(1, homo_im.count+1):
                        ref_meta_dict = ref_im.tags(bi)
                        homo_meta_dict = {k: v for k, v in ref_meta_dict.items() if k in ['ABBREV', 'ID', 'NAME']}
                        homo_im.update_tags(bi, **homo_meta_dict)


class HomonimRefSpace(HomonImBase):

    def homogenise(self, out_filename, method='gain_only', normalise=False, debug=True):
        """
        Perform homogenisation

        Parameters
        ----------
        out_filename : str, pathlib.Path
            name of the file to save the homogenised image to
        """
        ref_array, ref_profile = self._read_ref()


        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            with rio.open(self._src_filename, 'r') as src_im:
                # src_mask = src_im.dataset_mask()
                # src_mask_poly = [poly for poly in dataset_features(src_im, sampling=10, band=False, as_mask=True,
                #                                    with_nodata=False, geographic=False, precision=1)]
                # src_ds_mask = np.zeros((ref_array.shape[1], ref_array.shape[2]), dtype=np.uint8)
                #
                # _, xform = reproject(src_mask, destination=src_ds_mask,
                #                      src_transform=src_im.transform, src_crs=src_im.crs,
                #                      dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'],
                #                      resampling=Resampling.average, num_threads=multiprocessing.cpu_count())

                calib_profile = src_im.profile
                calib_profile.update(num_threads=multiprocessing.cpu_count(), compress='deflate', interleave='band',
                                     nodata=0, dtype=rio.float32)
                with rio.open(out_filename, 'w', **calib_profile) as homo_im:
                    if debug:
                        param_profile = ref_profile.copy()
                        param_profile.update(num_threads=multiprocessing.cpu_count(), compress='deflate',
                                             interleave='band',
                                             nodata=None, dtype=rio.float32, count=src_im.profile['count'])
                        if method != 'gain_only':
                            param_profile['count'] *= 2
                        param_out_file_name = out_filename.parent.joinpath(out_filename.stem + '_PARAMS.tif')
                        param_im = rio.open(param_out_file_name, 'w', **param_profile)

                    # process by band to limit memory usage
                    bands = list(range(1, src_im.count + 1))
                    for bi in bands:
                        src_array = src_im.read(bi, out_dtype=np.float32)  # NB bands along first dim

                        # downsample the source window into the reference CRS and gid
                        # specify nodata so that these pixels are excluded from calculations
                        src_ds_array = np.zeros((ref_array.shape[1], ref_array.shape[2]), dtype=np.float32)
                        # TODO: resample rather than reproject?
                        # TODO: also think if there is a neater way we can do this, rather than having arrays and transforms in mem
                        #   we have datasets / memory images ?

                        _, xform = reproject(src_array, destination=src_ds_array,
                                             src_transform=src_im.transform, src_crs=src_im.crs,
                                             dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'],
                                             resampling=Resampling.average, num_threads=multiprocessing.cpu_count(),
                                             src_nodata=src_im.profile['nodata'], dst_nodata=0)

                        # find the calibration parameters for this band
                        if method.lower() == 'gain_only':
                            param_ds_array = self._find_gains_cv(ref_array[bi-1, :, :], src_ds_array,
                                                                 win_size=self.win_size, normalise=normalise)
                        else:
                            param_ds_array = self._find_gain_and_offset_winview(ref_array[bi - 1, :, :], src_ds_array,
                                                                                win_size=self.win_size, normalise=normalise)

                        # upsample the parameter array
                        param_array = np.zeros((param_ds_array.shape[0], *src_array.shape), dtype=np.float32)
                        _, xform = reproject(param_ds_array, destination=param_array,
                                             src_transform=ref_profile['transform'], src_crs=ref_profile['crs'],
                                             dst_transform=src_im.transform, dst_crs=src_im.crs,
                                             resampling=Resampling.cubic_spline, num_threads=multiprocessing.cpu_count(),
                                             src_nodata=0, dst_nodata=0)

                        # apply the calibration and write
                        calib_src_array = param_array[0, :, :] * src_array
                        if param_array.shape[0] > 1:
                            calib_src_array += param_array[1, :, :]
                        homo_im.write(calib_src_array.astype(homo_im.dtypes[bi-1]), indexes=bi)

                        if debug:
                            param_im.write(param_ds_array[0, :, :].astype(param_im.dtypes[bi - 1]), indexes=bi)
                            if param_ds_array.shape[0] > 1:
                                _bi = bi + ref_profile['count']
                                param_im.write(param_ds_array[1, :, :].astype(param_im.dtypes[_bi - 1]), indexes=_bi)

                    # store the homogenisation parameters as metadata in the output file
                    homo_im.update_tags(HOMO_METHOD=method, HOMO_NORM=str(normalise), HOMO_WIN=str(self.win_size),
                                        HOMO_SRC=self._src_filename.name, HOMO_REF=self._ref_filename.name)
                    if debug:
                        param_im.close()

##

class HomonimSrcSpace(HomonImBase):


    def homogenise(self, out_filename, method=None, normalise=False):
        """
        Perform homogenisation

        Parameters
        ----------
        out_filename : str
            name of the file to save the homogenised image to
        """
        ref_array, ref_profile = self._read_ref()
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'):
            with rio.open(self._src_filename, 'r') as src_im:
                src_mask = src_im.dataset_mask()
                # src_mask_poly = [poly for poly in dataset_features(src_im, sampling=10, band=False, as_mask=True,
                #                                    with_nodata=False, geographic=False, precision=1)]
                # src_ds_mask = np.zeros((ref_array.shape[1], ref_array.shape[2]), dtype=np.uint8)
                #
                # _, xform = reproject(src_mask, destination=src_ds_mask,
                #                      src_transform=src_im.transform, src_crs=src_im.crs,
                #                      dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'],
                #                      resampling=Resampling.average, num_threads=multiprocessing.cpu_count())

                calib_profile = src_im.profile
                calib_profile.update(num_threads=multiprocessing.cpu_count(), compress='deflate', interleave='band',
                                     nodata=0)
                with rio.open(out_filename, 'w', **calib_profile) as calib_im:
                    # process by band to limit memory usage
                    bands = list(range(1, src_im.count + 1))
                    for bi in bands:
                        src_array = src_im.read(bi, out_dtype=np.float32)  # NB bands along first dim

                        # upsample ref to source CRS and grid
                        ref_src_array = np.zeros_like(src_array, dtype=np.float32)
                        _, xform = reproject(ref_array[bi-1, :, :], destination=ref_src_array,
                                             src_transform=ref_profile['transform'], src_crs=ref_profile['crs'],
                                             dst_transform=src_im.transform, dst_crs=src_im.crs,
                                             resampling=Resampling.cubic_spline,
                                             num_threads=multiprocessing.cpu_count(),
                                             src_nodata=ref_profile['nodata'], dst_nodata=0)

                        # find the calibration parameters for this band
                        win_size = self.win_size * np.round(np.array(ref_profile['res'])/np.array(src_im.res)).astype(int)
                        # src_array[np.logical_not(src_mask)] = np.nan    # TODO get rid of this step
                        # TODO parallelise this, use opencv and or gpu, and or use integral image!
                        param_array = self._find_gains_cv(ref_src_array, src_array, win_size=win_size)

                        # apply the calibration and write
                        calib_src_array = param_array[0, :, :] * src_array
                        if param_array.shape[0] > 1:
                            calib_src_array += param_array[1, :, :]
                        calib_im.write(calib_src_array.astype(calib_im.dtypes[bi - 1]), indexes=bi)

##
