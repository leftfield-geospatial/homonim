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
import cv2 as cv
import numpy as np
from rasterio.fill import fillnodata
from rasterio.warp import Resampling

from homonim import hom_dtype, hom_nodata
from homonim.raster_array import RasterArray, nan_equals


class KernelModel():
    def __init__(self, method='gain_im_offset', kernel_shape=(5, 5), debug=False, r2_inpaint_thresh=None,
                 src2ref_interp=Resampling.average, ref2src_interp=Resampling.cubic_spline):
        if not method in ['gain', 'gain_im_offset', 'gain_offset']:
            raise ValueError('method must be one of gain | gain_im_offset | gain_offset')
        self._method = method
        if not np.all(np.mod(kernel_shape, 2) == 1):
            raise ValueError('kernel_shape must be odd in both dimensions')
        self._kernel_shape = np.array(kernel_shape)
        self._debug = debug
        self._r2_inpaint_thresh = r2_inpaint_thresh
        self._src2ref_interp = src2ref_interp
        self._ref2src_interp = ref2src_interp

    def _r2_array(self, ref_array, src_array, param_array, kernel_shape=(5, 5), mask=None, mask_sum=None,
                  ref_sum=None, src_sum=None, ref2_sum=None, src2_sum=None, src_ref_sum=None, dest_array=None):
        kernel_shape = tuple(kernel_shape)  # convert to tuple for opencv
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)  # common opencv arguments

        # if mask is passed, it is assumed invalid pixels in ref_ and src_array have been zeroed
        if mask is None:
            mask = (nan_equals(src_array, hom_nodata) & nan_equals(ref_array, hom_nodata))
            ref_array[~mask] = 0
            src_array[~mask] = 0
        if mask_sum is None:
            mask_sum = cv.boxFilter(mask.astype(hom_dtype), -1, kernel_shape, **filter_args)
        if ref2_sum is None:
            ref2_sum = cv.sqrBoxFilter(ref_array, -1, kernel_shape, **filter_args)
        if src2_sum is None:
            src2_sum = cv.sqrBoxFilter(src_array, -1, kernel_shape, **filter_args)
        if src_ref_sum is None:
            src_ref_sum = cv.boxFilter(src_array * ref_array, -1, kernel_shape, **filter_args)

        ss_tot_array = (mask_sum * ref2_sum) - (ref_sum ** 2)

        if param_array.shape[0] > 1:  # gain and offset
            if ref_sum is None:
                ref_sum = cv.boxFilter(ref_array, -1, kernel_shape, **filter_args)
            if src_sum is None:
                src_sum = cv.boxFilter(src_array, -1, kernel_shape, **filter_args)

            ss_res_array = (((param_array[0] ** 2) * src2_sum) +
                            (2 * np.product(param_array[:2], axis=0) * src_sum) -
                            (2 * param_array[0] * src_ref_sum) -
                            (2 * param_array[1] * ref_sum) +
                            ref2_sum + (mask_sum * (param_array[1] ** 2)))
        else:
            ss_res_array = (((param_array[0] ** 2) * src2_sum) -
                            (2 * param_array[0] * src_ref_sum) +
                            ref2_sum)

        ss_res_array *= mask_sum

        if dest_array is None:
            dest_array = np.full(src_array.shape, fill_value=hom_nodata, dtype=hom_dtype)
        np.divide(ss_res_array, ss_tot_array, out=dest_array, where=mask.astype('bool', copy=False))
        np.subtract(1, dest_array, out=dest_array, where=mask.astype('bool', copy=False))
        return dest_array

    def _fit_im_offset(self, ref_ra, src_ra):
        """
        Offset source image band (in place) to match reference image band. Uses basic dark object subtraction (DOS)
        type approach.

        Parameters
        ----------
        ref_ra : RasterArray
            Reference block in a RasterArray.
        src_ra : RasterArray
            Source block, co-located with ref_ra and the same shape.

        Returns
        -------
        : numpy.array
        A two element array of linear offset model i.e. [1, offset]
        """
        offset_model = np.zeros(2)
        mask = ref_ra.mask & src_ra.mask
        if not np.any(mask):
            return offset_model
        offset_model[0] = np.std(ref_ra.array, where=mask) / np.std(src_ra.array, where=mask)
        offset_model[1] = np.percentile(ref_ra.array[mask], 1) - np.percentile(src_ra.array[mask], 1) * offset_model[0]
        return offset_model

    def _fit_gain(self, ref_ra: RasterArray, src_ra: RasterArray, kernel_shape=(5, 5)):
        """
        Find sliding kernel gains for a band using opencv convolution.

        Parameters
        ----------
        ref_ra : RasterArray
            Reference block in a RasterArray.
        src_ra : RasterArray
            Source block, co-located with ref_ra and the same shape.
        kernel_shape : numpy.array_like, list, tuple, optional
            Sliding kernel [width, height] in pixels.

        Returns
        -------
        param_ra :RasterArray
        RasterArray of sliding kernel model parameters. Gains in first band, offsets in the second, and optionally
        R2 for each kernel model in the third band when debug is on.
        """
        # adapted from https://www.mathsisfun.com/data/least-squares-regression.html with c=0
        # get arrays and find a combined ref & src mask
        ref_array = ref_ra.array
        src_array = src_ra.array
        param_profile = src_ra.profile.copy()
        param_profile.update(count=3 if self._debug else 2)
        mask = ref_ra.mask & src_ra.mask
        # mask invalid pixels with 0 so that these do not contribute to kernel sums in *boxFilter()
        ref_array[~mask] = 0
        src_array[~mask] = 0
        kernel_shape = tuple(kernel_shape)  # force to tuple for opencv

        # convolve the kernel with src and ref to get kernel sums (uses DFT for large kernels)
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)  # common opencv arguments
        src_sum = cv.boxFilter(src_array, -1, kernel_shape, **filter_args)
        ref_sum = cv.boxFilter(ref_array, -1, kernel_shape, **filter_args)

        # create parameter array filled with nodata
        param_ra = RasterArray.from_profile(None, param_profile)
        param_ra.array[1, mask] = 0  # set offsets to 0

        # find sliding kernel gains, avoiding divide by 0
        np.divide(ref_sum, src_sum, out=param_ra.array[0], where=mask)

        if self._debug:
            # Find R2 of the sliding kernel models
            self._r2_array(ref_array, src_array, param_ra.array[:2], kernel_shape=kernel_shape, mask=mask,
                           ref_sum=ref_sum, src_sum=src_sum, dest_array=param_ra.array[2])
        return param_ra

    def _fit_gain_im_offset(self, ref_ra: RasterArray, src_ra: RasterArray, kernel_shape=(5, 5), dest_src_ra=None):
        """
        Find sliding kernel gains and 'image' (i.e. block) offset for a band using opencv convolution.

        Parameters
        ----------
        ref_ra : RasterArray
            Reference block in a RasterArray.
        src_ra : RasterArray
            Source block, co-located with ref_ra and the same shape.
        kernel_shape : numpy.array_like, list, tuple, optional
            Sliding kernel [width, height] in pixels.
        dest_src_ra: RasterArray, optional
            Destination RasterArray to write the offset src_ra into (useful to make the operation in-place with
            dest_src_ra=src_ra)

        Returns
        -------
        param_ra :RasterArray
        RasterArray of sliding kernel model parameters. Gains in first band, offsets in the second, and optionally
        R2 for each kernel model in the third band when debug is on.
        """
        # TODO: sort out ref/src in-place differences
        offset_model = self._fit_im_offset(ref_ra, src_ra)
        # create RasterArray to hold offset src image, and force nodata to nan so that operation below remains
        # correctly masked
        src_offset_ra = dest_src_ra or src_ra.copy()
        src_offset_ra.nodata = RasterArray.default_nodata

        # apply the offset model
        src_offset_ra.array = src_ra.array * offset_model[0] + offset_model[1]

        # find gains for offset src
        param_ra = self._fit_gain(ref_ra, src_offset_ra, kernel_shape=kernel_shape)

        # incorporate the offset model in the parameter RasterArray
        param_ra.array[1] = param_ra.array[0] * offset_model[1]
        param_ra.array[0] *= offset_model[0]
        return param_ra

    def _fit_gain_offset(self, ref_ra: RasterArray, src_ra: RasterArray, kernel_shape=(15, 15)):
        """
        Find sliding kernel full linear model for a band using opencv convolution.

        Parameters
        ----------
        ref_ra : RasterArray
            Reference block in a RasterArray.
        src_ra : RasterArray
            Source block, co-located with ref_ra and the same shape.
        kernel_shape : numpy.array_like, list, tuple, optional
            Sliding kernel [width, height] in pixels.

        Returns
        -------
        param_ra :RasterArray
        RasterArray of sliding kernel model parameters. Gains in first band, offsets in the second, and optionally
        R2 for each kernel model in the third band when debug is on.
        """
        # Least squares formulae adapted from https://www.mathsisfun.com/data/least-squares-regression.html

        # get arrays and find a combined ref & src mask
        ref_array = ref_ra.array
        src_array = src_ra.array
        param_profile = src_ra.profile.copy()
        param_profile.update(count=3 if (self._debug or self._r2_inpaint_thresh) else 2)
        mask = ref_ra.mask & src_ra.mask
        # mask invalid pixels with 0 so that these do not contribute to kernel sums in *boxFilter()
        ref_array[~mask] = 0
        src_array[~mask] = 0
        kernel_shape = tuple(kernel_shape)  # force to tuple for opencv

        # find the numerator for the gain i.e N*cov(ref, src)
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)
        src_sum = cv.boxFilter(src_array, -1, kernel_shape, **filter_args)
        ref_sum = cv.boxFilter(ref_array, -1, kernel_shape, **filter_args)
        src_ref_sum = cv.boxFilter(src_array * ref_array, -1, kernel_shape, **filter_args)
        mask_sum = cv.boxFilter(mask.astype(hom_dtype, copy=False), -1, kernel_shape, **filter_args)
        m_num_array = (mask_sum * src_ref_sum) - (src_sum * ref_sum)

        # find the denominator for the gain i.e. N*var(src)
        src2_sum = cv.sqrBoxFilter(src_array, -1, kernel_shape, **filter_args)
        m_den_array = (mask_sum * src2_sum) - (src_sum ** 2)

        # create parameter RasterArray filled with nodata
        param_ra = RasterArray.from_profile(None, param_profile)

        # find the gain = cov(ref, src) / var(src), avoiding divide by 0
        np.divide(m_num_array, m_den_array, out=param_ra.array[0], where=mask)

        # solve for the offset c = y - mx, given that the linear model passes through (mean(ref), mean(src))
        np.divide(ref_sum - (param_ra.array[0] * src_sum), mask_sum, out=param_ra.array[1], where=mask)

        if (self._debug) or (self._r2_inpaint_thresh is not None):
            # Find R2 of the sliding kernel models
            self._r2_array(ref_array, src_array, param_ra.array[:2], kernel_shape=kernel_shape, mask=mask,
                           mask_sum=mask_sum, ref_sum=ref_sum, src_sum=src_sum, src2_sum=src2_sum,
                           src_ref_sum=src_ref_sum, dest_array=param_ra.array[2])

        if self._r2_inpaint_thresh is not None:
            # fill/inpaint low R2 areas and negative gain areas in the offset parameter
            r2_mask = (param_ra.array[2] > self._r2_inpaint_thresh) & (param_ra.array[0] > 0) & mask
            param_ra.array[1] = fillnodata(param_ra.array[1], r2_mask)
            param_ra.mask = mask  # re-mask as nodata areas will have been filled above

            # recalculate the gain for the filled areas using m = (y - c)/x and and the point (mean(ref), mean(src)))
            r2_mask = ~r2_mask & mask
            np.divide(ref_sum - mask_sum * param_ra.array[1], src_sum, out=param_ra.array[0], where=r2_mask)

        return param_ra

    def fit(self, ref_ra, src_ra):
        """
        Fits sliding kernel models to reference and source arrays

        Parameters
        ----------
        ref_ra: homonim.RasterArray
                   M x N RasterArray of reference data, collocated, and of similar spectral content, to src_ra
        src_ra: homonim.RasterArray
                   M x N RasterArray of source data, collocated, and of similar spectral content, to ref_ra
        """
        if ((ref_ra.transform != src_ra.transform) or (ref_ra.crs.to_proj4() != src_ra.crs.to_proj4()) or
                (ref_ra.shape != src_ra.shape)):
            raise ValueError('ref_ra and src_ra must have the same CRS, transform and shape')

        if self._method == 'gain':
            param_ra = self._fit_gain(ref_ra, src_ra, kernel_shape=self._kernel_shape)
        elif self._method == 'gain_im_offset':  # normalise src_ds_ra in place
            param_ra = self._fit_gain_im_offset(ref_ra, src_ra, kernel_shape=self._kernel_shape)
        else:
            param_ra = self._fit_gain_offset(ref_ra, src_ra, kernel_shape=self._kernel_shape)

        return param_ra

    def apply(self, src_ra: RasterArray, param_ra: RasterArray):
        """
        Applies sliding kernel models to a source array

        Parameters
        ----------
        src_ra: homonim.RasterArray
                   M x N RasterArray of source data, collocated, and of similar spectral content, to ref_ra
        """
        out_array = param_ra.array[0] * src_ra.array + param_ra.array[1]
        return RasterArray.from_profile(out_array, param_ra.profile)

    def mask_partial(self, out_ra, ref_res):
        res_ratio = np.ceil(np.divide(ref_res, out_ra.res))
        morph_kernel_shape = np.ceil(res_ratio * self._kernel_shape).astype('int')
        mask = out_ra.mask
        se = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(morph_kernel_shape))
        mask = cv2.erode(mask.astype('uint8', copy=False), se).astype('bool', copy=False)
        out_ra.mask = mask
        return out_ra


class RefSpaceModel(KernelModel):
    def fit(self, ref_ra, src_ra):
        # downsample src_ra to ref crs and grid
        src_ds_ra = src_ra.reproject(**ref_ra.proj_profile, resampling=self._src2ref_interp)
        return KernelModel.fit(self, ref_ra, src_ds_ra)

    def apply(self, src_ra: RasterArray, param_ra: RasterArray):
        # upsample the param_ra to src crs and grid
        _param_ra = RasterArray.from_profile(param_ra.array[:2], param_ra.profile)
        param_us_ra = _param_ra.reproject(**src_ra.proj_profile, resampling=self._ref2src_interp)
        return KernelModel.apply(self, src_ra, param_us_ra)


class SrcSpaceModel(KernelModel):
    def fit(self, ref_ra, src_ra):
        # upsample ref_ra to src crs and grid
        ref_us_ra = ref_ra.reproject(**src_ra.proj_profile, resampling=self._ref2src_interp)
        return KernelModel.fit(self, ref_us_ra, src_ra)
