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

from typing import Tuple, Dict, Optional

import cv2 as cv
import numpy as np
from rasterio.enums import Resampling
from rasterio.fill import fillnodata

from homonim import utils
from homonim.enums import Model
from homonim.raster_array import RasterArray

ONdArray = Optional[np.ndarray]
OShape = Optional[Tuple[int, int]]


class KernelModel:

    default_kernel_shape = (5, 5)           # default kernel shape
    default_model = Model.gain_blk_offset   # default model

    def __init__(self,
        model: Model = default_model,
        kernel_shape: Tuple[int, int] = default_kernel_shape,
        find_r2: bool = False,
        **kwargs
    ):
        """
        A base class for surface reflectance modelling and correction of blocks of image data.

        The surface reflectance relationship between source and reference image blocks is approximated with localised
        linear models.  Models are estimated for each pixel location inside a small rectangular kernel (window),
        using a fast DFT approach.  The corrected output is produced by applying the model parameters to the source
        image block.

        Based on the paper:
        Harris, Dugal & Van Niekerk, Adriaan. (2018). Radiometric homogenisation of aerial images by calibrating with
        satellite data. International Journal of Remote Sensing. 40. 1-25. 10.1080/01431161.2018.1528404.
        https://www.researchgate.net/publication
        /328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data

        Parameters
        ----------
        model: homonim.enums.Model, optional
            Surface reflectance correction model.
        kernel_shape: tuple, optional
            (height, width) of the kernel in pixels.
        find_r2: bool, optional
            Whether to calculate *R*\\ :sup:`2` (coefficient of determination) for each kernel model, and include in
            parameter arrays.
        kwargs:
            Optional configuration arguments.  See :meth:`KernelModel.create_config` for keys and defaults values.
        """
        self._model = Model(model)
        self._kernel_shape = utils.validate_kernel_shape(kernel_shape, model=model)
        self._find_r2 = find_r2

        # update config defaults with any passed values, and set attributes
        config = self.create_config(**kwargs)
        self._r2_inpaint_thresh: float = config['r2_inpaint_thresh']
        self._mask_partial: bool = config['mask_partial']
        self._downsampling: Resampling = config['downsampling']
        self._upsampling: Resampling = config['upsampling']

    @property
    def model(self) -> Model:
        """ Correction model type. """
        return self._model

    @property
    def kernel_shape(self) -> Tuple[int, int]:
        """ Kernel (height, width) in pixels. """
        return tuple(self._kernel_shape)

    @property
    def find_r2(self) -> bool:
        """ Whether to create and include R2 (coefficient of determination) in parameter arrays. """
        return self._find_r2

    @staticmethod
    def create_config(
        r2_inpaint_thresh: float = 0.25, mask_partial: bool = False, downsampling: Resampling = Resampling.average,
        upsampling: Resampling = Resampling.cubic_spline,
    ) -> Dict:
        """
        Utility method to create a KernelModel configuration dictionary that can be passed to
        :meth:`KernelModel.__init__` and :meth:`homonim.RasterFuse.process`.  Without arguments, the default
        configuration is returned.

        Parameters
        ----------
        r2_inpaint_thresh: float, optional
            *R*\\ :sup:`2` (coefficient of determination) threshold below which to `in-paint` kernel model parameters
            from surrounding areas.  Applies to the :attr:`~homonim.enums.Model.gain_offset` model only.  For
            pixels where the model gives a poor approximation to the data (this can occur in areas where source and
            reference differ due to e.g. shadowing, land cover changes etc.), model offsets are interpolated from
            surrounding areas, and gains re-estimated.  `None` turns off in-painting.
        mask_partial: bool, optional
            Mask output pixels not produced by full kernel or source/reference image coverage.  Useful for ensuring
            strict model validity, and reducing seam-lines between overlapping images.
        downsampling: rasterio.enums.Resampling, optional
            Resampling method to use when downsampling.  See the `rasterio docs
            <https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_ for
            available options.
        upsampling: rasterio.enums.Resampling, optional
            Resampling method to use when upsampling. See the `rasterio docs
            <https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_ for
            available options.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        return dict(
            r2_inpaint_thresh=r2_inpaint_thresh, mask_partial=mask_partial, downsampling=downsampling,
            upsampling=upsampling,
        )

    def _get_resampling(self, from_res: Tuple[float, float], to_res: Tuple[float, float]):
        """ Return the resampling method for re-projecting from resolution ``from_res`` to resolution ``to_res``. """
        return self._downsampling if np.prod(np.abs(from_res)) <= np.prod(np.abs(to_res)) else self._upsampling

    def _r2_array(
        self, ref_array: np.ndarray, src_array: np.ndarray, param_array: np.ndarray, mask: ONdArray = None,
        mask_sum: ONdArray = None, ref_sum: ONdArray = None, src_sum: ONdArray = None, ref2_sum: ONdArray = None,
        src2_sum: ONdArray = None, src_ref_sum: ONdArray = None, dest_array: ONdArray = None,
        kernel_shape: OShape = None
    ) -> np.ndarray:
        """
        Utility function to return R2 (coefficient of determination) at each pixel/kernel location for the given
        arrays.
        """
        if kernel_shape is None:
            kernel_shape = self._kernel_shape
        kernel_shape = tuple(kernel_shape)  # force to tuple for opencv
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)  # common opencv arguments

        # find the keyword arguments that were not provided
        if mask is None:
            # if mask is passed, assume that it has been applied to ref_array and src_array, otherwise do that here
            mask = (
                ~utils.nan_equals(src_array, RasterArray.default_nodata) &
                ~utils.nan_equals(ref_array, RasterArray.default_nodata)
            )
            ref_array[~mask] = 0
            src_array[~mask] = 0
        if mask_sum is None:
            mask_sum = cv.boxFilter(mask.astype(RasterArray.default_dtype), -1, kernel_shape[::-1], **filter_args)
        if ref_sum is None:
            ref_sum = cv.boxFilter(ref_array, -1, kernel_shape[::-1], **filter_args)
        if ref2_sum is None:
            ref2_sum = cv.sqrBoxFilter(ref_array, -1, kernel_shape[::-1], **filter_args)
        if src2_sum is None:
            src2_sum = cv.sqrBoxFilter(src_array, -1, kernel_shape[::-1], **filter_args)
        if src_ref_sum is None:
            src_ref_sum = cv.boxFilter(src_array * ref_array, -1, kernel_shape[::-1], **filter_args)

        # R2 is found using: R2 = 1 - (residual sum of squares)/(total sum of squares) = 1 - RSS/TSS
        # TSS = sum((ref - mean(ref))**2), which can be expanded and expressed in terms of cv.boxFilter kernel sums as:
        ss_tot_array = (mask_sum * ref2_sum) - (ref_sum ** 2)

        if param_array.shape[0] > 1:
            # find RSS for model == Model.gain_offset
            if src_sum is None:
                src_sum = cv.boxFilter(src_array, -1, kernel_shape[::-1], **filter_args)

            # RSS = sum((ref - ref_hat)**2)
            #     = sum((ref - (m*src + c))**2), where m and c are the first 2 bands of param_array
            # The above can be expanded and expressed in terms of cv.boxFilter kernel sums as:
            ss_res_array = (
                ((param_array[0] ** 2) * src2_sum) +
                (2 * np.product(param_array[:2], axis=0) * src_sum) -
                (2 * param_array[0] * src_ref_sum) -
                (2 * param_array[1] * ref_sum) +
                ref2_sum + (mask_sum * (param_array[1] ** 2))
            )  # yapf: disable
        else:
            # find RSS for model == Model.gain or Model.gain_blk_offset

            # RSS = sum((ref - m*src)**2), where m is the first band of param_array
            # The above can be expanded and expressed in terms of cv.boxFilter kernel sums as:
            ss_res_array = (((param_array[0] ** 2) * src2_sum) - (2 * param_array[0] * src_ref_sum) + ref2_sum)

        ss_res_array *= mask_sum

        if dest_array is None:
            # assign a destination array to write R2 into, if it was not provided
            dest_array = np.full(
                src_array.shape, fill_value=RasterArray.default_nodata, dtype=RasterArray.default_dtype
            )

        # find R2 = 1 - RSS/TSS, and write into dest_array
        np.divide(ss_res_array, ss_tot_array, out=dest_array, where=mask)
        np.subtract(1, dest_array, out=dest_array, where=mask)
        return dest_array

    @staticmethod
    def _fit_block_norm(src_ra: RasterArray, ref_ra: RasterArray) -> np.ndarray:
        """
        Return a two element [gain, offset] model to 'normalise' the source block, so that the standard deviation and
        first percentile of the source and reference blocks match.  (Can be thought of as a basic dark object
        subtraction (DOS) approach).
        """
        norm_model = np.zeros(2)
        mask = ref_ra.mask & src_ra.mask
        if not np.any(mask):
            return norm_model
        norm_model[0] = np.std(ref_ra.array[mask]) / np.std(src_ra.array[mask])
        norm_model[1] = np.percentile(ref_ra.array[mask], 1) - np.percentile(src_ra.array[mask], 1) * norm_model[0]
        return norm_model

    def _fit_gain(self, src_ra: RasterArray, ref_ra: RasterArray, kernel_shape: OShape = None) -> RasterArray:
        """
        Find sliding kernel gains, for a source & reference band, using opencv convolution.
        Returns a RasterArray of sliding kernel model parameters. Gains in first band, offsets in the second,
        and optionally R2 for each kernel model in the third band when :attr:`find_r2` is True.
        """
        # adapted from https://www.mathsisfun.com/data/least-squares-regression.html with c=0
        if kernel_shape is None:
            kernel_shape = self._kernel_shape
        kernel_shape = tuple(kernel_shape)  # force to tuple for opencv

        # mask invalid pixels with 0 so that these do not contribute to kernel sums in *boxFilter()
        ref_array = ref_ra.array
        src_array = src_ra.array
        mask = ref_ra.mask & src_ra.mask
        ref_array[~mask] = 0
        src_array[~mask] = 0

        # set up a RasterArray profile for the parameters
        param_profile = src_ra.profile.copy()
        param_profile.update(
            count=3 if self._find_r2 else 2, nodata=RasterArray.default_nodata, dtype=RasterArray.default_dtype
        )

        # convolve the kernel with src and ref to get kernel sums (uses DFT for large kernels)
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)  # common opencv arguments
        src_sum = cv.boxFilter(src_array, -1, kernel_shape[::-1], **filter_args)
        ref_sum = cv.boxFilter(ref_array, -1, kernel_shape[::-1], **filter_args)

        # create parameter RasterArray filled with nodata
        param_ra = RasterArray.from_profile(None, param_profile)
        param_ra.array[1, mask] = 0  # set offsets to 0

        # find sliding kernel gains, avoiding divide by 0
        np.divide(ref_sum, src_sum, out=param_ra.array[0], where=mask)

        if self._find_r2:
            # Find R2 of the sliding kernel models
            self._r2_array(
                ref_array, src_array, param_ra.array[:1], mask=mask, ref_sum=ref_sum, src_sum=src_sum,
                dest_array=param_ra.array[2], kernel_shape=kernel_shape
            )

        return param_ra

    def _fit_gain_blk_offset(
        self, src_ra: RasterArray, ref_ra: RasterArray, kernel_shape: OShape = None
    ) -> RasterArray:  # yapf: disable
        """
        Find sliding kernel gains and 'image' (i.e. block) offset, for a band, using opencv convolution.
        Returns a RasterArray of sliding kernel model parameters. Gains in first band, offsets in the second,
        and optionally R2 for each kernel model in the third band when :attr:`find_r2` is True.
        """
        if kernel_shape is None:
            kernel_shape = self._kernel_shape
        kernel_shape = tuple(kernel_shape)  # force to tuple for opencv

        # find the source->reference normalisation
        norm_model = self._fit_block_norm(src_ra, ref_ra)

        # force src nodata to nan so that operation below remains correctly masked
        src_ra.nodata = RasterArray.default_nodata

        # apply the normalisation (block gain and offset)
        src_ra.array = (src_ra.array * norm_model[0]) + norm_model[1]

        # find gains for normalised source
        param_ra = self._fit_gain(src_ra, ref_ra, kernel_shape=kernel_shape)

        # incorporate the normalisation model in the parameter RasterArray
        param_ra.array[1] = param_ra.array[0] * norm_model[1]
        param_ra.array[0] *= norm_model[0]
        return param_ra

    def _fit_gain_offset(self, src_ra: RasterArray, ref_ra: RasterArray, kernel_shape: OShape = None) -> RasterArray:
        """
        Find sliding kernel full linear model for a band using opencv convolution.
        Returns a RasterArray of sliding kernel model parameters. Gains in first band, offsets in the second,
        and optionally R2 for each kernel model in the third band when :attr:`find_r2` is True.
        """
        # Least squares formulae adapted from https://www.mathsisfun.com/data/least-squares-regression.html
        if kernel_shape is None:
            kernel_shape = self._kernel_shape
        kernel_shape = tuple(kernel_shape)  # force to tuple for opencv

        # mask invalid pixels with 0 so that these do not contribute to kernel sums in *boxFilter()
        ref_array = ref_ra.array
        src_array = src_ra.array
        mask = ref_ra.mask & src_ra.mask
        ref_array[~mask] = 0
        src_array[~mask] = 0

        # set up a RasterArray profile for the parameters
        param_profile = src_ra.profile.copy()
        find_r2 = self._find_r2 or (self._r2_inpaint_thresh is not None)
        param_profile.update(
            count=3 if find_r2 else 2, nodata=RasterArray.default_nodata, dtype=RasterArray.default_dtype
        )

        # find the numerator for the gain i.e N*cov(ref, src)
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)  # common opencv arguments
        src_sum = cv.boxFilter(src_array, -1, kernel_shape[::-1], **filter_args)
        ref_sum = cv.boxFilter(ref_array, -1, kernel_shape[::-1], **filter_args)
        src_ref_sum = cv.boxFilter(src_array * ref_array, -1, kernel_shape[::-1], **filter_args)
        mask_sum = cv.boxFilter(
            mask.astype(RasterArray.default_dtype, copy=False), -1, kernel_shape[::-1], **filter_args
        )
        m_num_array = (mask_sum * src_ref_sum) - (src_sum * ref_sum)

        # find the denominator for the gain i.e. N*var(src)
        src2_sum = cv.sqrBoxFilter(src_array, -1, kernel_shape[::-1], **filter_args)
        m_den_array = (mask_sum * src2_sum) - (src_sum ** 2)

        # create parameter RasterArray filled with nodata
        param_ra = RasterArray.from_profile(None, param_profile)

        # find the gain = cov(ref, src) / var(src), avoiding divide by 0
        np.divide(m_num_array, m_den_array, out=param_ra.array[0], where=mask)

        # solve for the offset c = y - mx, given that the linear model passes through (mean(ref), mean(src))
        np.divide(ref_sum - (param_ra.array[0] * src_sum), mask_sum, out=param_ra.array[1], where=mask)

        if find_r2:
            # Find R2 of the sliding kernel models
            self._r2_array(
                ref_array, src_array, param_ra.array[:2], mask=mask, mask_sum=mask_sum, ref_sum=ref_sum,
                src_sum=src_sum, src2_sum=src2_sum, src_ref_sum=src_ref_sum, dest_array=param_ra.array[2],
                kernel_shape=kernel_shape
            )

        if self._r2_inpaint_thresh is not None:
            # fill/inpaint low R2 and negative gain areas in the offset parameter
            r2_mask = (param_ra.array[2] > self._r2_inpaint_thresh) & (param_ra.array[0] > 0) & mask
            # NOTE: fillnodata does not release the GIL, so this can slow down processing, especially for proc-crs=src
            # TODO: raise an issue with rasterio to release the GIL on fillnodata
            param_ra.array[1] = fillnodata(param_ra.array[1], r2_mask)
            param_ra.mask = mask  # re-mask as nodata areas will have been filled above

            # recalculate the gain for the filled areas using m = (y - c)/x and and the point (mean(ref), mean(src)))
            r2_mask = ~r2_mask & mask
            np.divide(ref_sum - mask_sum * param_ra.array[1], src_sum, out=param_ra.array[0], where=r2_mask)

        return param_ra

    def _full_coverage_mask(self, in_mask_ra: RasterArray, param_ra: RasterArray) -> RasterArray:
        """
        Utility function to create a full coverage mask i.e. a mask of output pixels fully covered by source/reference
        data, sliding (model) kernels, and resampling kernels.

        Note: this is a strict approach that avoids any kind of partial coverage.

        Parameters
        ----------
        in_mask_ra: RasterArray
            Initial mask of valid source/reference pixels, as a RasterArray with dtype=='uint8'.
        param_ra: RasterArray
            A parameter RasterArray as returned by :meth:`fit` i.e. a RasterArray in the CRS and grid corresponding
            to :attr:`proc_crs`, whose mask corresponds to the combined reference & source masks.

        Returns
        -------
        RasterArray
            Full coverage mask as a RasterArray with dtype=='uint8'
        """

        # re-project the initial mask into proc_crs (the CRS and grid corresponding to the proc_crs attribute)
        mask_ra = in_mask_ra.reproject(**param_ra.proj_profile, nodata=None, resampling=Resampling.average)
        # find the mask of proc_crs pixels fully covered by mask_ra
        mask = (mask_ra.array >= 1).astype('uint8', copy=False)  # ref pixels fully covered by src
        # combine mask with the other (src/ref) mask via param_ra
        mask &= param_ra.mask

        # Mask out partial kernel coverage.
        # Similar to the block overlap amount, this removes ceil(kernel_shape/2) pixels from the nodata edge.  Note,
        # that this is the strict approach for proc_crs == ref, it could be floor(kernel_shape/2) for proc_crs == src,
        # which avoids the additional upsampling step.
        se = cv.getStructuringElement(cv.MORPH_RECT, tuple(np.array(self._kernel_shape[::-1]) + 2))
        mask_ra.array = cv.erode(mask, se, borderType=cv.BORDER_CONSTANT, borderValue=0)
        return mask_ra

    def fit(self, src_ra: RasterArray, ref_ra: RasterArray) -> RasterArray:
        """
        Fit sliding kernel models to reference and source blocks.

        Parameters
        ----------
        src_ra: RasterArray
            Source data block in a RasterArray, with the same CRS, shape & extents as ``ref_ra``.
        ref_ra: RasterArray
            Reference data block in a RasterArray.

        Returns
        -------
        RasterArray
            RasterArray of sliding kernel model parameters. Gains in first band, offsets in the second, and optionally
            *R*\\ :sup:`2` for each kernel model in the third band when :attr:`find_r2` is True.
        """
        # TODO : include a CRS comparison below i.e. one that is faster that rasterio's current implementation, and or
        #  raise an issue with rasterio about the speed of crs comparison
        if (ref_ra.transform != src_ra.transform) or (ref_ra.shape != src_ra.shape):
            raise ValueError("'ref_ra' and 'src_ra' must have the same CRS, transform and shape")

        if self._model == Model.gain:
            param_ra = self._fit_gain(src_ra, ref_ra, kernel_shape=self._kernel_shape)
        elif self._model == Model.gain_blk_offset:
            param_ra = self._fit_gain_blk_offset(src_ra, ref_ra, kernel_shape=self._kernel_shape)
        else:
            param_ra = self._fit_gain_offset(src_ra, ref_ra, kernel_shape=self._kernel_shape)

        return param_ra

    def apply(self, src_ra: RasterArray, param_ra: RasterArray) -> RasterArray:
        """
        Apply kernel models to a source block.

        Parameters
        ----------
        src_ra: RasterArray
            Source data block in a RasterArray.
        param_ra: RasterArray
            RasterArray of sliding kernel model parameters. Gains in first band, offsets in the second, and the same
            CRS, shape & extents as src_ra.

        Returns
        -------
        corr_ra :RasterArray
            Corrected block in a RasterArray.
        """
        if (param_ra.transform != src_ra.transform) or (param_ra.shape != src_ra.shape):
            raise ValueError("'param_ra' and 'src_ra' must have the same CRS, transform and shape")
        corr_array = (param_ra.array[0] * src_ra.array) + param_ra.array[1]
        corr_ra = RasterArray.from_profile(corr_array, param_ra.profile)
        return corr_ra


class RefSpaceModel(KernelModel):
    """
    A KernelModel subclass, for estimating model parameters in the reference image CRS and grid.

    The source image block is re-projected into the reference CRS to estimate the parameters.  Estimated parameters are
    subsequently re-projected to the source CRS for application to the (original) source image block.

    Recommended for the most common use case where the reference image has a lower resolution than the source image.
    """

    def fit(self, src_ra: RasterArray, ref_ra: RasterArray) -> RasterArray:
        # choose resampling method based on whether we are up- or downsampling
        resampling = self._get_resampling(src_ra.res, ref_ra.res)
        # downsample src_ra to reference CRS and grid
        src_ds_ra = src_ra.reproject(**ref_ra.proj_profile, resampling=resampling)
        # call base class fit with reference and source RasterArrays in the reference CRS & grid
        return KernelModel.fit(self, src_ds_ra, ref_ra)

    def apply(self, src_ra, param_ra):
        # create a parameter RasterArray containing only the first two bands of param_ra
        # (to speed up the re-projection below)
        _param_ra = RasterArray.from_profile(param_ra.array[:2], param_ra.profile)
        # choose resampling method based on whether we are up- or downsampling
        resampling = self._get_resampling(param_ra.res, src_ra.res)
        # re-project param_ra to source CRS and grid
        param_us_ra = _param_ra.reproject(**src_ra.proj_profile, resampling=resampling)

        if self._mask_partial:
            # find the mask of fully covered pixels in reference CRS and grid
            mask_ra = self._full_coverage_mask(src_ra.mask_ra, _param_ra)
            # re-project the mask to source CRS and grid, and apply to the parameters
            mask_us_ra = mask_ra.reproject(**src_ra.proj_profile, nodata=0, resampling=Resampling.nearest)
            param_us_ra.mask = mask_us_ra.array.astype('bool', copy=False)
        else:
            param_us_ra.mask = src_ra.mask

        # call base class apply with source and parameter RasterArrays in the source CRS & grid
        return KernelModel.apply(self, src_ra, param_us_ra)


class SrcSpaceModel(KernelModel):
    """
    A KernelModel subclass, for estimating model parameters in the source image CRS and grid.

    The reference image block is re-projected into the source CRS to estimate the parameters.  Estimated parameters are
    subsequently applied (directly) to the source image block.

    Recommended for the unusual use case where the source image has a lower resolution than the reference image.
    """

    def fit(self, src_ra: RasterArray, ref_ra: RasterArray) -> RasterArray:
        # choose resampling method based on whether we are up- or downsampling
        resampling = self._get_resampling(ref_ra.res, src_ra.res)
        # upsample ref_ra to the source CRS and grid
        ref_us_ra = ref_ra.reproject(**src_ra.proj_profile, resampling=resampling)

        _src_ra = src_ra.copy()  # copy the source to avoid in-place changes in fit() below
        # call base class fit with source and parameter RasterArrays in the source CRS & grid
        param_ra = KernelModel.fit(self, _src_ra, ref_us_ra)

        if self._mask_partial:
            # remove r2 band from param masking
            _param_ra = RasterArray.from_profile(param_ra.array[:2], param_ra.profile)
            # find the mask of fully covered pixels in source CRS and grid, and apply it to the parameters
            mask_ra = self._full_coverage_mask(ref_ra.mask_ra, _param_ra)
            param_ra.mask = mask_ra.array.astype('bool', copy=False)
        else:
            param_ra.mask = src_ra.mask

        return param_ra
