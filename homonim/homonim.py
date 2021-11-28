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
import cProfile
import concurrent.futures
import multiprocessing
import pathlib
import pstats
import threading
import tracemalloc
from collections import namedtuple
from enum import Enum

import cv2
import cv2 as cv
import numpy as np
import rasterio as rio
from rasterio import windows
from rasterio.fill import fillnodata
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, Resampling, transform_geom, transform_bounds
from shapely.geometry import box, shape
from tqdm import tqdm

from homonim import get_logger, hom_dtype, hom_nodata
from homonim.raster_array import RasterArray

logger = get_logger(__name__)


class Model(Enum):
    gain_only = 1
    gain_and_image_offset = 2
    gain_and_offset = 3


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
    exp_win = rio.windows.Window(col_off.astype('int'), row_off.astype('int'), width.astype('int'), height.astype('int'))
    return exp_win


"""Projection related raster properties"""
RasterProps = namedtuple('RasterProps', ['crs', 'transform', 'shape', 'res', 'bounds', 'nodata', 'count', 'profile'])

"""Overlapping window object"""
OvlBlock = namedtuple('OvlBlock', ['src_block', 'src_transform', 'ref_block', 'ref_transform', 'out_block', 'outer'])


class HomonImBase:
    def __init__(self, src_filename, ref_filename, homo_config=None, out_config=None):
        """
        Class for homogenising images

        Parameters
        ----------
        src_filename : str, pathlib.Path
            Source image filename.
        ref_filename: str, pathlib.Path
            Reference image filename.
        homo_config: dict, optional
            Dictionary for advanced homogenisation configuration ().
        out_config: dict, optional
            Dictionary for configuring output file format.
        """
        # TODO: refactor which parameters get passed here, and which to homogenise()
        self._src_props = None
        self._ref_props = None
        self._src_filename = pathlib.Path(src_filename)
        self._ref_filename = pathlib.Path(ref_filename)
        self._check_rasters()

        # self._ref_array = None

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

    # @property
    # def ref_array(self):
    #     """Reference image array collocated and covering source region in source CRS."""
    #     if self._ref_array is None:
    #         self._ref_array = self._read_ref()
    #     return self._ref_array

    @property
    def src_props(self):
        """Source raster properties."""
        if self._src_props is None:
            self._check_rasters()
        return self._src_props

    @property
    def ref_props(self):
        """Reference raster properties."""
        if self._ref_props is None:
            self._check_rasters()
        return self._ref_props

    def _check_rasters(self):
        """Check bounds, band count, and compression type of source and reference images"""
        if not self._src_filename.exists():
            raise Exception(f'Source file {self._src_filename.stem} does not exist')
        if not self._ref_filename.exists():
            raise Exception(f'Reference file {self._ref_filename.stem} does not exist')

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

            with rio.open(self._src_filename, 'r') as src_im, rio.open(self._ref_filename, 'r') as _ref_im:
                if src_im.crs.to_proj4() != _ref_im.crs.to_proj4():  # re-project the reference image to source CRS
                    # TODO: here we project from transform on the ref grid and that include source bounds
                    #   we could just project into src_im transform though too.
                    logger.warning('Reprojecting reference image to the source CRS. '
                                   'To avoid this step, provide reference and source images in the same CRS')
                with WarpedVRT(_ref_im, crs=src_im.crs, resampling=Resampling.bilinear) as ref_im:

                    # check reference covers source
                    src_box = box(*src_im.bounds)
                    ref_box = box(*ref_im.bounds)

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
                            logger.warning(f'{fn} has no nodata value, defaulting to {hom_nodata}.\n'
                                           'Any invalid pixels in this image should be first be masked with nodata.')

                    src_nodata = hom_nodata if src_im.nodata is None else src_im.nodata
                    ref_nodata = hom_nodata if ref_im.nodata is None else ref_im.nodata
                    ref_win = expand_window_to_grid(ref_im.window(*src_im.bounds))
                    ref_transform = ref_im.window_transform(ref_win)
                    ref_shape = (ref_win.height, ref_win.width)
                    ref_bounds = ref_im.window_bounds(ref_win)

                    self._src_props = RasterProps(crs=src_im.crs, transform=src_im.transform, shape=src_im.shape,
                                                  res=list(src_im.res), bounds=src_im.bounds, nodata=src_nodata,
                                                  count=src_im.count, profile=src_im.profile)
                    self._ref_props = RasterProps(crs=src_im.crs, transform=ref_transform, shape=ref_shape,
                                                  res=list(ref_im.res), bounds=ref_bounds, nodata=ref_nodata,
                                                  count=ref_im.count, profile=ref_im.profile)

                # if src_im.crs != ref_im.crs:    # CRS's don't match
                #     logger.warning('The reference will be re-projected to the source CRS.  \n'
                #                    'To avoid this step, provide a reference image in the source CRS')

    def _read_ref(self):
        """
        Read the source region from the reference image in the source CRS, and populate image properties.
        """
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._src_filename, 'r') as src_im:
            with rio.open(self._ref_filename, 'r') as _ref_im:
                with WarpedVRT(_ref_im, crs=src_im.crs, resampling=Resampling.bilinear) as ref_im:
                    ref_win = expand_window_to_grid(ref_im.window(*src_im.bounds))
                    ref_bands = range(1, src_im.count + 1)
                    ref_array = ref_im.read(ref_bands, window=ref_win).astype(hom_dtype)

                    if (ref_im.nodata is not None) and (ref_im.nodata != hom_nodata):
                        ref_array[ref_array == ref_im.nodata] = hom_nodata

        return ref_array


    def _overlap_blocks(self, block_size=(1024, 1024), overlap=(0, 0)):
        overlap = np.array(overlap)
        block_size = np.array(block_size)

        src_block_ul = np.mgrid[0:(self.src_props.shape[0] - 2 * overlap[0]):(block_size[0] - 2 * overlap[0]),
                     0:(self.src_props.shape[1] - 2 * overlap[1]):(block_size[1] - 2 * overlap[1])]
        src_block_br = src_block_ul + block_size.reshape(-1, 1, 1)
        src_block_br[0, -1, :] = self.src_props.shape[0]
        src_block_br[1, :, -1] = self.src_props.shape[1]

        ovl_blocks = []
        for ovl_ul, ovl_br in zip(src_block_ul.reshape(2, -1).T, src_block_br.reshape(2, -1).T):
            src_block = rio.windows.Window.from_slices((ovl_ul[0], ovl_br[0]), (ovl_ul[1], ovl_br[1]))
            out_block = rio.windows.Window.from_slices((ovl_ul[0] + overlap[0], ovl_br[0] -  overlap[0]),
                                                          (ovl_ul[1] + overlap[1], ovl_br[1] - overlap[1]))

            src_block_transform = rio.windows.transform(src_block, self.src_props.transform)
            src_block_bounds = windows.bounds(src_block, self.src_props.transform)
            ref_block = windows.from_bounds(*src_block_bounds, transform=self.ref_props.transform)
            ref_block = expand_window_to_grid(ref_block)
            ref_block_transform = rio.windows.transform(ref_block, self.ref_props.transform)

            outer = np.any(ovl_ul==0) or np.any(ovl_br==self.src_props.shape)
            ovl_win = OvlBlock(src_block, src_block_transform, ref_block, ref_block_transform, out_block, outer)
            ovl_blocks.append(ovl_win)
        return ovl_blocks

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
        param_profile.update(dtype=hom_dtype, count=self.src_props.count * 3, nodata=hom_nodata, tiled=True)
        return param_profile

    def _create_param_filename(self, filename):
        """Return a debug parameter raster filename, given the homogenised raster filename"""
        filename = pathlib.Path(filename)
        return filename.parent.joinpath(f'{filename.stem}_PARAMS{filename.suffix}')

    def build_overviews(self, filename):
        """
        Builds internal overviews for a existing raster file.

        Parameters
        ----------
        filename: str, pathlib.Path
                  Path to the raster file to build overviews for.
        """
        filename = pathlib.Path(filename)

        if not filename.exists():
            raise Exception(f'{filename} does not exist')
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(filename, 'r+') as homo_im:
            homo_im.build_overviews([2, 4, 8, 16, 32], Resampling.average)

    def set_metadata(self, filename, **kwargs):
        """
        Copy various metadata to a homogenised raster (GeoTIFF) file.

        Parameters
        ----------
        filename: str, pathlib.Path
                  Path to the GeoTIFF raster file to copy metadata to.
        kwargs: dict
                Dictionary of metadata items to copy to raster.
        """
        filename = pathlib.Path(filename)

        if not filename.exists():
            raise Exception(f'{filename} does not exist')

        with rio.open(self._ref_filename, 'r') as ref_im, rio.open(filename, 'r+') as homo_im:
            # Set user-supplied metadata
            homo_im.update_tags(**kwargs)
            # Copy any geedim generated metadata from the reference file
            for bi in range(1, homo_im.count + 1):
                ref_meta_dict = ref_im.tags(bi)
                homo_meta_dict = {k: v for k, v in ref_meta_dict.items() if k in ['ABBREV', 'ID', 'NAME']}
                homo_im.update_tags(bi, **homo_meta_dict)

    def _project_src_to_ref(self, src_array, src_nodata=hom_nodata, src_transform=None, dst_dtype=hom_dtype,
                            dst_nodata=hom_nodata, dst_transform=None, resampling=None):
        """
        Re-project an array from source to reference CRS

        Parameters
        ----------
        src_array: numpy.array_like
                   Source raster array
        src_nodata: int, float, optional
                    Nodata value for src_array (Default: 0)
        src_transform: rasterio.Affine, optional
                        Affine transform for src_array (Default: transform for the full source raster)
        dst_dtype: str, type, optional
                    Data type for re-projected array. (Default: float32)
        dst_nodata: int, float, optional
                    Nodata value for re-projected array (Default: 0)
        dst_transform: rasterio.Affine, optional
                        Affine transform for re-projected array (Default: transform for the reference raster)
        resampling: rasterio.warp.Resampling, optional
                       Resampling method (Default: configured 'ref2src_interp' value)

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

        if src_transform is None:
            src_transform = self.src_props.transform

        if dst_transform is None:
            dst_transform = self.ref_props.transform

        # TODO: checks on nodata vals, that a dest_nodata==0 will not conflict with pixel values
        _, _ = reproject(
            src_array,
            destination=ref_array,
            src_transform=src_transform,
            src_crs=self.src_props.crs,
            dst_transform=dst_transform,
            dst_crs=self.src_props.crs,
            resampling=resampling,
            num_threads=multiprocessing.cpu_count(),
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
        )
        return ref_array

    def _project_ref_to_src(self, ref_array, ref_nodata=hom_nodata, ref_transform=None, dst_dtype=hom_dtype,
                            dst_nodata=hom_nodata, dst_transform=None, resampling=None):
        """
        Re-project an array from reference to source CRS

        Parameters
        ----------
        ref_array: numpy.array_like
                   Reference CRS raster array to re-project
        ref_nodata: int, float, optional
                    Nodata value for ref_array (Default: 0)
        ref_transform: rasterio.Affine, optional
                        Affine transform for ref_array (Default: transform for the reference raster)
        dst_dtype: str, type, optional
                    Data type for re-projected array. (Default: float32)
        dst_nodata: int, float, optional
                    Nodata value for re-projected array (Default: 0)
        dst_transform: rasterio.Affine, optional
                        Affine transform for re-projected array (Default: transform for the source raster)
        resampling: rasterio.warp.Resampling, optional
                       Resampling method (Default: configured 'ref2src_interp' value)

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

        if ref_transform is None:
            ref_transform = self.ref_props.transform

        if dst_transform is None:
            dst_transform = self.src_props.transform


        # TODO: checks on nodata vals, that a dest_nodata==0 will not conflict with pixel values
        _, _ = reproject(
            ref_array,
            destination=src_array,
            src_transform=ref_transform,
            src_crs=self.src_props.crs,
            dst_transform=dst_transform,
            dst_crs=self.src_props.crs,
            resampling=resampling,
            num_threads=multiprocessing.cpu_count(),
            src_nodata=ref_nodata,
            dst_nodata=dst_nodata,
        )
        return src_array

    def _sliding_window_view(self, array, win_size):
        """
        Return a 3D strided view of 2D array to allow sliding window operations without data copying.
        Sliding windows are stacked along the third dimension.

        Parameters
        ----------
        array: numpy.array_like
               2D array to find sliding window view of
        win_size: tuple, list, numpy.array_like
                Size of the sliding window

        Returns
        -------
        : numpy.array_like
          3D sliding window view into array (shape = (win_size[0], win_size[1], number of sliding windows))
        """
        xstep = 1
        shape = array.shape[:-1] + (win_size[0], int(1 + (array.shape[-1] - win_size[0]) / xstep))
        strides = array.strides[:-1] + (array.strides[-1], xstep * array.strides[-1])
        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides, writeable=False)

    def _src_image_offset(self, ref_array, src_array):
        """
        Offset source image band (in place) to match reference image band. Uses basic dark object subtraction (DOS)
        type approach.

        Parameters
        ----------
        ref_array: numpy.array_like
                   Reference image band.
        src_array: numpy.array_like
                   Source image band.

        Returns
        -------
        : numpy.array
        A two element array of linear offset model i.e. [1, offset]
        """
        src_mask = src_array != hom_nodata
        # TODO: can we avoid these masked copies?
        _ref_array = ref_array[src_mask]
        _src_array = src_array[src_mask]

        norm_model = np.zeros(2)
        norm_model[0] = _ref_array.std() / _src_array.std()
        norm_model[1] = np.percentile(_ref_array, 1) - np.percentile(_src_array, 1) * norm_model[0]

        # logger.info(f'Image normalisation gain / offset: {norm_model[0]:.4f} / {norm_model[1]:.4f}')
        src_array[src_mask] = norm_model[0]*_src_array + norm_model[1]
        return norm_model


    def _find_gains_cv(self, ref_array, src_array, win_size=(5, 5)):
        """
        Find sliding window gains for a band using opencv convolution.

        Parameters
        ----------
        ref_array : numpy.array_like
            Reference band in an MxN array.
        src_array : numpy.array_like
            Source band, collocated with ref_array and the same shape.
        win_size : numpy.array_like, list, tuple, optional
            Sliding window [width, height] in pixels.

        Returns
        -------
        param_array : numpy.array_like
        1 x M x N array of gains, corresponding to M x N src_ and ref_array
        """

        mask = src_array != hom_nodata
        win_size = tuple(win_size)  # convert to tuple for opencv
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)  # common opencv arguments

        # find ref/src pixel ratios avoiding divide by nodata=0
        ratio_array = np.full_like(src_array, fill_value=hom_nodata, dtype=hom_dtype)
        _ = np.divide(ref_array, src_array, out=ratio_array, where=mask.astype('bool', copy=False))

        # sum the ratio and mask over sliding windows (uses DFT for large kernels)
        # (mask_sum effectively finds N, the number of valid pixels for each window)
        ratio_sum = cv.boxFilter(ratio_array, -1, win_size, **filter_args)
        mask_sum = cv.boxFilter(mask.astype(hom_dtype), -1, win_size, **filter_args)

        # calculate gains for valid pixels
        param_array = np.full((1, *src_array.shape), fill_value=hom_nodata, dtype=hom_dtype)
        _ = np.divide(ratio_sum, mask_sum, out=param_array[0, :, :], where=mask.astype('bool', copy=False))

        return param_array

    def _find_gain_and_offset_cv(self, ref_array, src_array, win_size=(15, 15)):
        """
        Find sliding window gain and offset for a band using opencv convolution.

        ref_array : numpy.array_like
            Reference band in an MxN array.
        src_array : numpy.array_like
            Source band, collocated with ref_array and the same shape.
        win_size : numpy.array_like, list, tuple, optional
            Sliding window [width, height] in pixels.

        Returns
        -------
        param_array : numpy.array_like
        2 x M x N array of gains and offsets, corresponding to M x N src_ and ref_array
        """
        # Least squares formulae adapted from https://www.mathsisfun.com/data/least-squares-regression.html

        mask = src_array != hom_nodata
        ref_array[~mask] = hom_nodata  # apply src mask to ref, so we are summing on same pixels
        win_size = tuple(win_size)  # force to tuple for opencv

        # find the numerator for the gain i.e. cov(ref, src)
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)
        src_sum = cv.boxFilter(src_array, -1, win_size, **filter_args)
        ref_sum = cv.boxFilter(ref_array, -1, win_size, **filter_args)
        src_ref_sum = cv.boxFilter(src_array * ref_array, -1, win_size, **filter_args)
        mask_sum = cv.boxFilter(mask.astype(hom_dtype), -1, win_size, **filter_args)
        m_num_array = (mask_sum * src_ref_sum) - (src_sum * ref_sum)
        del (src_ref_sum)  # free memory when possible

        # find the denominator for the gain i.e. var(src)
        src2_sum = cv.sqrBoxFilter(src_array, -1, win_size, **filter_args)
        m_den_array = (mask_sum * src2_sum) - (src_sum ** 2)
        del (src2_sum)

        # find the gain = cov(ref, src) / var(src)
        param_array = np.full((2, *src_array.shape), fill_value=hom_nodata, dtype=hom_dtype)
        _ = np.divide(m_num_array, m_den_array, out=param_array[0, :, :], where=mask.astype('bool', copy=False))

        # find the offset c = y - mx, using the fact that the LS linear model passes through (mean(ref), mean(src))
        _ = np.divide(ref_sum - (param_array[0, :, :] * src_sum), mask_sum, out=param_array[1, :, :],
                      where=mask.astype('bool', copy=False))

        # refit any areas with low R2 using offset "inpainting"
        if self._homo_config['r2_threshold'] is not None:
            # Find R2 of the models for each pixel
            ref2_sum = cv.sqrBoxFilter(ref_array, -1, win_size, **filter_args)
            ss_tot_array = (mask_sum * ref2_sum) - (ref_sum ** 2)
            res_array = (param_array[0, :, :] * src_array + param_array[1, :, :]) - ref_array
            ss_res_array = mask_sum * cv.sqrBoxFilter(res_array, -1, win_size, **filter_args)

            r2_array = np.full(src_array.shape, fill_value=hom_nodata, dtype=hom_dtype)
            np.divide(ss_res_array, ss_tot_array, out=r2_array, where=mask.astype('bool', copy=False))
            np.subtract(1, r2_array, out=r2_array, where=mask.astype('bool', copy=False))

            # fill ("inpaint") low R2 areas in the offset parameter
            rf_mask = (r2_array < self._homo_config['r2_threshold']) | (param_array[0, :, :] <= 0)
            param_array[1, :, :] = fillnodata(param_array[1, :, :], ~rf_mask)
            param_array[1, ~mask] = hom_nodata  # re-set nodata as nodata areas will have been filled above

            # recalculate the gain for the filled areas (linear LS line passes through (mean(src), mean(ref)))
            rf_mask &= mask
            np.divide(ref_sum - mask_sum * param_array[1, :, :], src_sum, out=param_array[0, :, :],
                      where=rf_mask.astype('bool', copy=False))
            if self._homo_config['debug']:
                # append R2 to parameters so they can all be written to a debug raster
                # TODO: avoid a copy here somehow?
                param_array = np.concatenate((param_array, r2_array.reshape(1, *r2_array.shape)), axis=0)

        return param_array

    def _homogenise_array(self, ref_array, src_array, method='gain_only', win_size=(5, 5), normalise=False,
                          ref_transform=None, src_transform=None):
        """
        Wrapper to homogenise an array of source image data

        Parameters
        ----------
        ref_array: numpy.array_like
                   M x N array of reference data, collocated, and of similar spectral content, to src_array
        src_array: numpy.array_like
                   M x N array of source data, collocated, and of similar spectral content, to ref_array
        method: str, optional
                The homogenisation method: ['gain_only'|'gain_offset'].  (Default: 'gain_only')
        win_size : numpy.array_like, list, tuple, optional
                   Sliding window (width, height) in pixels.
        normalise: bool, optional
                   Perform image-wide normalisation prior to homogenisation.  (Default: False)
        Returns
        -------
        param_array: K x M x N array of linear model parameters corresponding to src_ and ref_array.
                     K=1 for method='gain_only' i.e. the gains and K=2 for method='gain_offset'.
                     param_array[0, :, :] contains the gains, and param_array[1, :, :] contains the offsets for
                     method='gain_offset'.
        """
        raise NotImplementedError()

    def homogenise_by_band(self, out_filename, method='gain_only', win_size=(5, 5), normalise=False):
        """
        Homogenise a raster file by band.

        Parameters
        ----------
        out_filename: str, pathlib.Path
                      Path of the homogenised raster file to create.
        method: str, optional
                The homogenisation method: ['gain_only'|'gain_offset'].  (Default: 'gain_only')
        win_size : numpy.array_like, list, tuple, optional
                   Sliding window (width, height) in reference pixels.
        normalise: bool, optional
                   Perform image-wide normalisation prior to homogenisation.  (Default: False)
        """

        if not np.all(np.mod(win_size, 2) == 1):
            raise Exception('win_size must be odd in both dimensions')

        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._src_filename, 'r') as src_im:
            with rio.open(self._ref_filename, 'r') as _ref_im, WarpedVRT(_ref_im, crs=src_im.crs, resampling=Resampling.bilinear) as ref_im:
                if self._homo_config['debug']:
                    # setup profiling
                    tracemalloc.start()
                    proc_profile = cProfile.Profile()
                    proc_profile.enable()
                    # TODO: NB adjust param_profile according to src-space or ref-space (self.ref_props.profile)
                    param_profile = self._create_param_profile(src_im.profile.copy())

                    # create parameter raster file
                    param_out_file_name = self._create_param_filename(out_filename)
                    param_im = rio.open(param_out_file_name, 'w', **param_profile)

                # create the output raster file
                out_profile = self._create_out_profile(src_im.profile)
                out_im = rio.open(out_filename, 'w', **out_profile)  # avoid too many nested indents with 'with' statements

                # initialise process by band
                bands = list(range(1, src_im.count + 1))
                bar = tqdm(total=len(bands) + 1)
                bar.update(0)
                src_read_lock = threading.Lock()
                ref_read_lock = threading.Lock()
                write_lock = threading.Lock()
                param_lock = threading.Lock()
                try:
                    def process_band(bi):
                        """Thread-safe function to homogenise band bi of src_im"""

                        with src_read_lock:
                            _src_array = src_im.read(bi, out_dtype=hom_dtype)
                            src_array = RasterArray.from_profile(_src_array, src_im.profile)

                        with ref_read_lock:
                            ref_win = expand_window_to_grid(ref_im.window(*src_im.bounds))
                            _ref_array = ref_im.read(bi, window=ref_win, out_dtype=hom_dtype)
                            ref_array = RasterArray.from_profile(_ref_array, ref_im.profile, window=ref_win)

                        out_array, param_array = self._homogenise_array(ref_array, src_array, method=method,
                                                                        win_size=win_size, normalise=normalise)

                        if out_im.nodata != hom_nodata:
                            out_array[out_array == hom_nodata] = out_im.nodata

                        if self._homo_config['debug']:
                            with param_lock:
                                for pi in range(param_array.shape[0]):
                                    _bi = bi + pi * src_im.count
                                    param_im.write(param_array[pi, :, :].astype(param_im.dtypes[_bi - 1]), indexes=_bi)

                        with write_lock:
                            out_im.write(out_array.astype(out_im.dtypes[bi - 1]), indexes=bi)
                            bar.update(1)

                    if self._homo_config['multithread']:
                        # process bands in concurrent threads
                        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                            executor.map(process_band, bands)
                    else:
                        # process bands consecutively
                        for bi in bands:
                            process_band(bi)
                finally:
                    out_im.close()
                    if self._homo_config['debug']:
                        param_im.close()
                    bar.update(1)
                    bar.close()

            if self._homo_config['debug']:
                # print profiling info
                # (tottime is the total time spent in the function alone. cumtime is the total time spent in the function
                # plus all functions that this function called)
                proc_profile.disable()
                proc_stats = pstats.Stats(proc_profile).sort_stats('cumtime')
                logger.debug(f'Processing time:')
                proc_stats.print_stats(20)

                current, peak = tracemalloc.get_traced_memory()
                logger.debug(f"Memory usage: current: {current / 10 ** 6:.1f} MB, peak: {peak / 10 ** 6:.1f} MB")

    def homogenise_by_block(self, out_filename, method='gain_only', win_size=(5, 5), normalise=False):
        """
        Homogenise a raster file by block.

        Parameters
        ----------
        out_filename: str, pathlib.Path
                      Path of the homogenised raster file to create.
        method: str, optional
                The homogenisation method: ['gain_only'|'gain_offset'].  (Default: 'gain_only')
        win_size : numpy.array_like, list, tuple, optional
                   Sliding window (width, height) in reference pixels.
        normalise: bool, optional
                   Perform image-wide normalisation prior to homogenisation.  (Default: False)
        """
        win_size = np.array(win_size)
        if not np.all(np.mod(win_size, 2) == 1):
            raise Exception('win_size must be odd in both dimensions')

        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._src_filename, 'r') as src_im:
            if self._homo_config['debug']:
                # setup profiling
                tracemalloc.start()
                proc_profile = cProfile.Profile()
                proc_profile.enable()
                param_profile = self._create_param_profile(src_im.profile.copy())

                # create parameter raster file
                param_out_file_name = self._create_param_filename(out_filename)
                param_im = rio.open(param_out_file_name, 'w', **param_profile)

            # create the output raster file
            out_profile = self._create_out_profile(src_im.profile)
            out_im = rio.open(out_filename, 'w', **out_profile)  # avoid too many nested indents with 'with' statements

            # initialise process by band
            bands = list(range(1, src_im.count + 1))
            # TODO: what happens when src res is not an integer factor of ref res?
            overlap = (np.floor(win_size / 2) * np.round(np.array(self.ref_props.res) / np.array(self.src_props.res))).astype(int)
            ovl_wins = self._overlap_blocks(block_size=(252, 252), overlap=overlap)
            bar = tqdm(total=len(bands) + 1)
            bar.update(0)
            read_lock = threading.Lock()
            write_lock = threading.Lock()
            param_lock = threading.Lock()
            try:
                for bi in bands:
                    def process_block(ovl_block: OvlBlock):
                        """Thread-safe function to homogenise a window of src_im"""

                        with read_lock:
                            src_array = src_im.read(bi, window=ovl_block.src_block, out_dtype=hom_dtype)

                        # TODO: the below is wrong, how do we get the ref window corresponding to ovl_block.ovl_block?
                        ref_array = self.ref_array[bi - 1, :, :][ovl_block.ref_block.toslices()]
                        out_array, param_array = self._homogenise_array(
                            ref_array, src_array, method=method, win_size=win_size, normalise=normalise,
                            ref_transform=ovl_block.ref_transform, src_transform=ovl_block.src_transform
                        )

                        if out_im.nodata != hom_nodata:
                            out_array[out_array == hom_nodata] = out_im.nodata

                        if self._homo_config['debug']:
                            with param_lock:
                                for pi in range(param_array.shape[0]):
                                    _bi = bi + pi * src_im.count
                                    _param_array = param_array[pi, overlap[0]:-overlap[0], overlap[1]:-overlap[1]]
                                    param_im.write(_param_array.astype(param_im.dtypes[_bi - 1]), window=ovl_block.out_block, indexes=_bi)

                        with write_lock:
                            _out_array = out_array[:, overlap[0]:-overlap[0], overlap[1]:-overlap[1]]
                            out_im.write(_out_array.astype(out_im.dtypes[bi - 1]), window=ovl_block.out_block, indexes=bi)
                            bar.update(1)

                    if self._homo_config['multithread']:
                        # process bands in concurrent threads
                        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                            executor.map(process_block, ovl_wins)
                    else:
                        # process bands consecutively
                        for ovl_win in ovl_wins:
                            process_block(ovl_win)
            finally:
                out_im.close()
                if self._homo_config['debug']:
                    param_im.close()
                bar.update(1)
                bar.close()

        if self._homo_config['debug']:
            # print profiling info
            # (tottime is the total time spent in the function alone. cumtime is the total time spent in the function
            # plus all functions that this function called)
            proc_profile.disable()
            proc_stats = pstats.Stats(proc_profile).sort_stats('cumtime')
            logger.debug(f'Processing time:')
            proc_stats.print_stats(20)

            current, peak = tracemalloc.get_traced_memory()
            logger.debug(f"Memory usage: current: {current / 10 ** 6:.1f} MB, peak: {peak / 10 ** 6:.1f} MB")


class HomonimRefSpace(HomonImBase):
    """Class for homogenising images in reference image space"""

    def _homogenise_array(self, ref_array, src_array, method='gain_only', win_size=(5, 5), normalise=False,
                          ref_transform=None, src_transform=None):

        # downsample src_array to reference grid
        # src_ds_array = self._project_src_to_ref(src_array, src_nodata=self.src_props.nodata,
        #                                         src_transform=src_transform, transform=ref_transform)
        # src_ds_array = src_array.project(crs=ref_array.crs, transform=ref_array.transform, shape=ref_array.shape,
        #                                    nodata=hom_nodata, resampling=self._homo_config['src2ref_interp'])
        src_ds_array = src_array.reproject(**ref_array.proj_profile, resampling=self._homo_config['src2ref_interp'])

        if self._homo_config['mask_partial_pixel']:
            # mask src_ds_array pixels that were not completely covered by src_array
            # TODO is this faster, or setting param_array[src_array == 0] = 0 below
            mask_ds_array = src_array.mask.reproject(**src_ds_array.proj_profile, resampling=Resampling.average)
            src_ds_array.array[np.logical_not(mask_ds_array.array == 1)] = hom_nodata
            # mask = (src_array.array != src_array.nodata).astype('uint8')
            # mask_ds_array = self._project_src_to_ref(mask, src_nodata=None, resampling=Resampling.average,
            #                                          src_transform=src_transform, transform=ref_transform)
            # src_ds_array[np.logical_not(mask_ds_array == 1)] = hom_nodata

        if normalise:
            norm_model = self._src_image_offset(ref_array.array, src_ds_array.array)

        if method.lower() == 'gain_only':
            _param_ds_array = self._find_gains_cv(ref_array.array, src_ds_array.array, win_size=win_size)
        else:
            _param_ds_array = self._find_gain_and_offset_cv(ref_array.array, src_ds_array.array, win_size=win_size)

        # upsample the parameter array to source grid
        # param_array = self._project_ref_to_src(param_ds_array[:2, :, :], ref_transform=ref_transform,
        #                                        transform=src_transform)
        param_ds_array = RasterArray.from_profile(_param_ds_array, src_ds_array.profile)
        # param_array = param_ds_array.project(crs=src_array.crs, transform=src_array.transform, shape=src_array.shape,
        #                                        nodata=hom_nodata, resampling=self._homo_config['ref2src_interp'])
        param_array = param_ds_array.reproject(**src_array.proj_profile, resampling=self._homo_config['ref2src_interp'])

        if self._homo_config['mask_partial_interp'] or self._homo_config['mask_partial_window']:
            # mask boundary param_array pixels that not fully covered by a window, or were extrapolated
            # ("partially interpolated") rather than purely interpolated
            param_mask = np.all(param_array.array == hom_nodata, axis=0)
            res_ratio = np.divide(ref_array.res, param_array.res)
            # mask_partial_window covers mask_partial_interp, so don't do both
            if self._homo_config['mask_partial_window']:
                kernel_size = np.ceil(res_ratio*win_size).astype('int32')
            elif self._homo_config['mask_partial_interp']:
                kernel_size = np.ceil(res_ratio).astype('int32')  # /2
            se = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(kernel_size))
            param_mask = cv2.dilate(param_mask.astype('uint8', copy=False), se).astype('bool', copy=False)
            param_array.array[:, param_mask] = hom_nodata

        # apply the model to src_array
        if normalise:
            out_array = param_array.array[0, :, :] * norm_model[0] * src_array.array
            out_array += param_array.array[0, :, :] * norm_model[1]
        else:
            out_array = param_array.array[0, :, :] * src_array.array

        if param_array.shape[0] > 1:
            out_array += param_array.array[1, :, :]

        return out_array, param_ds_array.array


##

class HomonimSrcSpace(HomonImBase):
    """Class for homogenising images in source image space"""

    def _homogenise_array(self, ref_array, src_array, method='gain_only', win_size=(5, 5), normalise=False,
                          ref_transform=None, src_transform=None):
        # re-assign source nodata if necessary
        if src_array.nodata != hom_nodata:
            src_array.array[src_array.array == src_array.nodata] = hom_nodata

        # upsample reference to source grid
        # ref_us_array = self._project_ref_to_src(ref_array, ref_nodata=self.ref_props.nodata, ref_transform=ref_transform,
        #                                         transform=src_transform)
        ref_us_array = ref_array.reproject(**src_array.proj_profile, resampling=self._homo_config['ref2src_interp'])

        if normalise:  # normalise src_array in place
            self._src_image_offset(ref_us_array.array, src_array.array)

        # find win_size in source pixels
        win_size = win_size * np.round(ref_array.res / src_array.res).astype(int)

        if method.lower() == 'gain_only':
            param_array = self._find_gains_cv(ref_us_array.array, src_array.array, win_size=win_size)
        else:
            param_array = self._find_gain_and_offset_cv(ref_us_array.array, src_array.array, win_size=win_size)

        if self._homo_config['mask_partial_window']:
            # mask boundary param_array pixels that not fully covered by a window
            param_mask = np.all(param_array == hom_nodata, axis=0)
            se = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(win_size))
            param_mask = cv2.dilate(param_mask.astype('uint8', copy=False), se).astype('bool', copy=False)
            param_array[:, param_mask] = hom_nodata

        # apply the model to src_array
        out_array = param_array[0, :, :] * src_array.array
        if param_array.shape[0] > 1:
            out_array += param_array[1, :, :]

        return out_array, param_array
