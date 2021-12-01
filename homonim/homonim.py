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
from itertools import product

import cv2
import cv2 as cv
import numpy as np
import rasterio as rio
from rasterio.windows import Window
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
    exp_win = Window(col_off.astype('int'), row_off.astype('int'), width.astype('int'), height.astype('int'))
    return exp_win


"""Projection related raster properties"""
RasterProps = namedtuple('RasterProps', ['crs', 'transform', 'shape', 'res', 'bounds', 'nodata', 'count', 'profile'])

"""Overlapping block object"""
OvlBlock = namedtuple('OvlBlock', ['band_i', 'src_block', 'ref_block', 'out_block', 'outer'])


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
                'mask_partial_kernel': False,
                'mask_partial_interp': False,
                'multithread': True,
                'r2_threshold': 0.25,
                'max_block_mem': 100,
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
                    self._ref_props = RasterProps(crs=ref_im.crs, transform=ref_transform, shape=ref_shape,
                                                  res=list(ref_im.res), bounds=ref_bounds, nodata=ref_nodata,
                                                  count=ref_im.count, profile=ref_im.profile)

                # if src_im.crs != ref_im.crs:    # CRS's don't match
                #     logger.warning('The reference will be re-projected to the source CRS.  \n'
                #                    'To avoid this step, provide a reference image in the source CRS')

    def _auto_block_shape(self, src_shape, src_kernel_shape=None):
        if src_kernel_shape is None:
            src_kernel_shape = (0, 0)
        max_block_mem = self._homo_config['max_block_mem'] * (2**20)    # MB to Bytes
        dtype_size = np.dtype(hom_dtype).itemsize

        div_dim = np.argmax(src_shape)
        block_shape = np.array(src_shape)
        while (np.product(block_shape)*dtype_size > max_block_mem) and np.all(block_shape > src_kernel_shape):
            block_shape[div_dim] /= 2
            div_dim = np.mod(div_dim + 1, 2)
        # block_shape += src_kernel_shape
        return np.round(block_shape).astype('int')


    def _overlap_blocks(self, block_shape, overlap=(0, 0)):
        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._src_filename, 'r') as src_im:
            with WarpedVRT(rio.open(self._ref_filename, 'r'), crs=src_im.crs) as ref_im:
                src_shape = np.array(src_im.shape)
                overlap = np.array(overlap)
                block_shape = np.array(block_shape)
                ovl_blocks = []

                for band_i in range(src_im.count):
                    for ul_row, ul_col in product(range(-overlap[0], (src_shape[0] - 2 * overlap[0]), block_shape[0]),
                                                  range(-overlap[1], (src_shape[1] - 2 * overlap[1]), block_shape[1])):
                        ul = np.array((ul_row, ul_col))
                        br = ul + block_shape + 2 * overlap
                        src_ul = np.fmax(ul, (0, 0))
                        src_br = np.fmin(br, src_shape - 1)
                        out_ul = ul + overlap
                        out_br = br - overlap

                        src_block = Window.from_slices((src_ul[0], src_br[0]), (src_ul[1], src_br[1]))
                        ref_block = expand_window_to_grid(ref_im.window(*src_im.window_bounds(src_block)))
                        out_block = Window.from_slices((out_ul[0], out_br[0]), (out_ul[1], out_br[1]))

                        outer = np.any(src_ul==0) or np.any(src_br==src_shape-1)
                        ovl_blocks.append(OvlBlock(band_i+1, src_block, ref_block, out_block, outer))
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
        if not np.any(src_mask):
            return np.zeros(2)
        # TODO: can we avoid these masked copies?
        _ref_array = ref_array[src_mask]
        _src_array = src_array[src_mask]

        norm_model = np.zeros(2)
        norm_model[0] = _ref_array.std() / _src_array.std()
        norm_model[1] = np.percentile(_ref_array, 1) - np.percentile(_src_array, 1) * norm_model[0]

        # logger.info(f'Image normalisation gain / offset: {norm_model[0]:.4f} / {norm_model[1]:.4f}')
        src_array[src_mask] = norm_model[0]*_src_array + norm_model[1]
        return norm_model


    def _find_gains_cv(self, ref_array, src_array, kernel_shape=(5, 5)):
        """
        Find sliding kernel gains for a band using opencv convolution.

        Parameters
        ----------
        ref_array : numpy.array_like
            Reference band in an MxN array.
        src_array : numpy.array_like
            Source band, collocated with ref_array and the same shape.
        kernel_shape : numpy.array_like, list, tuple, optional
            Sliding kernel [width, height] in pixels.

        Returns
        -------
        param_array : numpy.array_like
        1 x M x N array of gains, corresponding to M x N src_ and ref_array
        """

        mask = src_array != hom_nodata
        kernel_shape = tuple(kernel_shape)  # convert to tuple for opencv
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)  # common opencv arguments

        # find ref/src pixel ratios avoiding divide by nodata=0
        ratio_array = np.full_like(src_array, fill_value=hom_nodata, dtype=hom_dtype)
        _ = np.divide(ref_array, src_array, out=ratio_array, where=mask.astype('bool', copy=False))

        # sum the ratio and mask over sliding kernel (uses DFT for large kernels)
        # (mask_sum effectively finds N, the number of valid pixels for each kernel)
        ratio_sum = cv.boxFilter(ratio_array, -1, kernel_shape, **filter_args)
        mask_sum = cv.boxFilter(mask.astype(hom_dtype), -1, kernel_shape, **filter_args)

        # mask out parameter pixels that were not completely covered by a window of valid data
        # TODO: this interacts with mask_partial_interp, so rather do it in homogenise after --ref-space US?
        #   this is sort of the same question as, is it better to do pure extrapolation, or interpolation with semi-covered data?
        # if self._homo_config['mask_partial_kernel']:
        #     mask &= (mask_sum >= np.product(kernel_shape))

        # calculate gains for valid pixels
        param_array = np.full((1, *src_array.shape), fill_value=hom_nodata, dtype=hom_dtype)
        _ = np.divide(ratio_sum, mask_sum, out=param_array[0, :, :], where=mask.astype('bool', copy=False))

        return param_array

    def _find_gain_and_offset_cv(self, ref_array, src_array, kernel_shape=(15, 15)):
        """
        Find sliding kernel gain and offset for a band using opencv convolution.

        ref_array : numpy.array_like
            Reference band in an MxN array.
        src_array : numpy.array_like
            Source band, collocated with ref_array and the same shape.
        kernel_shape : numpy.array_like, list, tuple, optional
            Sliding kernel [width, height] in pixels.

        Returns
        -------
        param_array : numpy.array_like
        2 x M x N array of gains and offsets, corresponding to M x N src_ and ref_array
        """
        # Least squares formulae adapted from https://www.mathsisfun.com/data/least-squares-regression.html

        mask = src_array != hom_nodata
        ref_array[~mask] = hom_nodata  # apply src mask to ref, so we are summing on same pixels
        kernel_shape = tuple(kernel_shape)  # force to tuple for opencv

        # find the numerator for the gain i.e. cov(ref, src)
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)
        src_sum = cv.boxFilter(src_array, -1, kernel_shape, **filter_args)
        ref_sum = cv.boxFilter(ref_array, -1, kernel_shape, **filter_args)
        src_ref_sum = cv.boxFilter(src_array * ref_array, -1, kernel_shape, **filter_args)
        mask_sum = cv.boxFilter(mask.astype(hom_dtype), -1, kernel_shape, **filter_args)
        m_num_array = (mask_sum * src_ref_sum) - (src_sum * ref_sum)
        del (src_ref_sum)  # free memory when possible

        # find the denominator for the gain i.e. var(src)
        src2_sum = cv.sqrBoxFilter(src_array, -1, kernel_shape, **filter_args)
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
            ref2_sum = cv.sqrBoxFilter(ref_array, -1, kernel_shape, **filter_args)
            ss_tot_array = (mask_sum * ref2_sum) - (ref_sum ** 2)
            res_array = (param_array[0, :, :] * src_array + param_array[1, :, :]) - ref_array
            ss_res_array = mask_sum * cv.sqrBoxFilter(res_array, -1, kernel_shape, **filter_args)

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

        # TODO: this interacts with mask_partial_interp, so rather do it in homogenise after --ref-space US?
        #   this is sort of the same question as, is it better to do pure extrapolation, or interpolation with semi-covered data?
        # mask out parameter pixels that were not completely covered by a window of valid data
        # if self._homo_config['mask_partial_kernel']:
        #     mask &= (mask_sum >= np.product(kernel_shape))
        #     param_array[:, ~mask] = hom_nodata

        return param_array

    def _homogenise_array(self, ref_array, src_array, method='gain_only', kernel_shape=(5, 5), normalise=False):
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
        kernel_shape : numpy.array_like, list, tuple, optional
                   Sliding kernel (width, height) in pixels.
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

    def homogenise_by_band(self, out_filename, method='gain_only', kernel_shape=(5, 5), normalise=False):
        """
        Homogenise a raster file by band.

        Parameters
        ----------
        out_filename: str, pathlib.Path
                      Path of the homogenised raster file to create.
        method: str, optional
                The homogenisation method: ['gain_only'|'gain_offset'].  (Default: 'gain_only')
        kernel_shape : numpy.array_like, list, tuple, optional
                   Sliding kernel (width, height) in reference pixels.
        normalise: bool, optional
                   Perform image-wide normalisation prior to homogenisation.  (Default: False)
        """

        if not np.all(np.mod(kernel_shape, 2) == 1):
            raise Exception('kernel_shape must be odd in both dimensions')

        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._src_filename, 'r') as src_im:
            with WarpedVRT(rio.open(self._ref_filename, 'r'), crs=src_im.crs, resampling=Resampling.bilinear) as ref_im:
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
                                                                        kernel_shape=kernel_shape, normalise=normalise)

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

    def homogenise_by_block(self, out_filename, method='gain_only', kernel_shape=(5, 5), normalise=False):
        """
        Homogenise a raster file by block.

        Parameters
        ----------
        out_filename: str, pathlib.Path
                      Path of the homogenised raster file to create.
        method: str, optional
                The homogenisation method: ['gain_only'|'gain_offset'].  (Default: 'gain_only')
        kernel_shape : numpy.array_like, list, tuple, optional
                   Sliding kernel (width, height) in reference pixels.
        normalise: bool, optional
                   Perform image-wide normalisation prior to homogenisation.  (Default: False)
        """
        kernel_shape = np.array(kernel_shape)
        if not np.all(np.mod(kernel_shape, 2) == 1):
            raise Exception('kernel_shape must be odd in both dimensions')

        with rio.Env(GDAL_NUM_THREADS='ALL_CPUs'), rio.open(self._src_filename, 'r') as src_im:
            with WarpedVRT(rio.open(self._ref_filename, 'r'), crs=src_im.crs, resampling=Resampling.bilinear) as ref_im:
                res_ratio = np.ceil(np.array(ref_im.res) / np.array(src_im.res))
                src_kernel_shape = (kernel_shape * res_ratio).astype(int)
                overlap = np.ceil(res_ratio + src_kernel_shape/2).astype(int)
                block_shape = self._auto_block_shape(src_im.shape, src_kernel_shape)
                ovl_blocks = self._overlap_blocks(block_shape=block_shape, overlap=overlap)

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
                # bands = list(range(1, src_im.count + 1))
                # TODO: what happens when src res is not an integer factor of ref res?
                bar = tqdm(total=len(ovl_blocks) + 1)
                bar.update(0)
                src_read_lock = threading.Lock()
                ref_read_lock = threading.Lock()
                write_lock = threading.Lock()
                param_lock = threading.Lock()
                try:
                    def process_block(ovl_block: OvlBlock):
                        """Thread-safe function to homogenise a block of src_im"""
                        with src_read_lock:
                            _src_array = src_im.read(ovl_block.band_i, window=ovl_block.src_block, out_dtype=hom_dtype)
                            src_array = RasterArray.from_profile(_src_array, src_im.profile, window=ovl_block.src_block)

                        with ref_read_lock:
                            _ref_array = ref_im.read(ovl_block.band_i, window=ovl_block.ref_block, out_dtype=hom_dtype)
                            ref_array = RasterArray.from_profile(_ref_array, ref_im.profile, window=ovl_block.ref_block)

                        out_array, param_array = self._homogenise_array(
                            ref_array, src_array, method=method, kernel_shape=kernel_shape, normalise=normalise
                        )

                        if out_im.nodata != hom_nodata:
                            out_array[out_array == hom_nodata] = out_im.nodata

                        if self._homo_config['debug']:
                            with param_lock:
                                for pi in range(param_array.shape[0]):
                                    _bi = ovl_block.band_i + pi * src_im.count
                                    _param_array = param_array[pi, overlap[0]:-overlap[0], overlap[1]:-overlap[1]]
                                    param_im.write(_param_array.astype(param_im.dtypes[_bi - 1]), window=ovl_block.out_block, indexes=_bi)

                        with write_lock:
                            out_arr_win = Window(ovl_block.out_block.col_off - ovl_block.src_block.col_off,
                                                 ovl_block.out_block.row_off - ovl_block.src_block.row_off,
                                                 ovl_block.out_block.width, ovl_block.out_block.height)
                            _out_array = out_array[out_arr_win.toslices()].astype(out_im.dtypes[ovl_block.band_i - 1])
                            out_im.write(_out_array, window=ovl_block.out_block, indexes=ovl_block.band_i)
                            bar.update(1)

                    if self._homo_config['multithread']:
                        # process bands in concurrent threads
                        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                            executor.map(process_block, ovl_blocks)
                    else:
                        # process bands consecutively
                        for ovl_block in ovl_blocks:
                            process_block(ovl_block)
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

    def _homogenise_array(self, ref_array, src_array, method='gain_only', kernel_shape=(5, 5), normalise=False):

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

        if normalise:
            norm_model = self._src_image_offset(ref_array.array, src_ds_array.array)

        if method.lower() == 'gain_only':
            _param_ds_array = self._find_gains_cv(ref_array.array, src_ds_array.array, kernel_shape=kernel_shape)
        else:
            _param_ds_array = self._find_gain_and_offset_cv(ref_array.array, src_ds_array.array, kernel_shape=kernel_shape)

        # upsample the parameter array to source grid
        # param_array = self._project_ref_to_src(param_ds_array[:2, :, :], ref_transform=ref_transform,
        #                                        transform=src_transform)
        param_ds_array = RasterArray.from_profile(_param_ds_array, src_ds_array.profile)
        # param_array = param_ds_array.project(crs=src_array.crs, transform=src_array.transform, shape=src_array.shape,
        #                                        nodata=hom_nodata, resampling=self._homo_config['ref2src_interp'])
        param_array = param_ds_array.reproject(**src_array.proj_profile, resampling=self._homo_config['ref2src_interp'])

        if self._homo_config['mask_partial_interp'] or self._homo_config['mask_partial_kernel']:
            param_mask = np.all(param_array.array == hom_nodata, axis=0)
            res_ratio = np.divide(ref_array.res, param_array.res)
            if self._homo_config['mask_partial_kernel']:
                morph_kernel_shape = np.ceil(res_ratio * kernel_shape).astype('int32')
            elif self._homo_config['mask_partial_interp'] or self._homo_config['mask_partial_pixel']:
                # no need to do mask_partial_interp when mask_partial_kernel==True
                morph_kernel_shape = np.ceil(res_ratio).astype('int32')
            se = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(morph_kernel_shape))
            param_mask = cv2.dilate(param_mask.astype('uint8', copy=False), se).astype('bool', copy=False)
            param_array.array[:, param_mask] = hom_nodata
        elif not self._homo_config['mask_partial_pixel']:
            # if no mask_partial_* was done, then mask with source mask
            param_array.array[:, (src_array.array == src_array.nodata)] = hom_nodata

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

    def _homogenise_array(self, ref_array, src_array, method='gain_only', kernel_shape=(5, 5), normalise=False):
        # re-assign source nodata if necessary
        if src_array.nodata != hom_nodata:
            src_array.array[src_array.array == src_array.nodata] = hom_nodata

        # upsample reference to source grid
        # ref_us_array = self._project_ref_to_src(ref_array, ref_nodata=self.ref_props.nodata, ref_transform=ref_transform,
        #                                         transform=src_transform)
        ref_us_array = ref_array.reproject(**src_array.proj_profile, resampling=self._homo_config['ref2src_interp'])

        if normalise:  # normalise src_array in place
            self._src_image_offset(ref_us_array.array, src_array.array)

        # find kernel_shape in source pixels
        src_kernel_shape = kernel_shape * np.round(ref_array.res / src_array.res).astype(int)

        if method.lower() == 'gain_only':
            param_array = self._find_gains_cv(ref_us_array.array, src_array.array, kernel_shape=src_kernel_shape)
        else:
            param_array = self._find_gain_and_offset_cv(ref_us_array.array, src_array.array, kernel_shape=src_kernel_shape)

        if self._homo_config['mask_partial_kernel']:
            # mask boundary param_array pixels that not fully covered by a kernel
            param_mask = np.all(param_array == hom_nodata, axis=0)
            se = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(src_kernel_shape))
            param_mask = cv2.dilate(param_mask.astype('uint8', copy=False), se).astype('bool', copy=False)
            param_array[:, param_mask] = hom_nodata

        # apply the model to src_array
        out_array = param_array[0, :, :] * src_array.array
        if param_array.shape[0] > 1:
            out_array += param_array[1, :, :]

        return out_array, param_array
