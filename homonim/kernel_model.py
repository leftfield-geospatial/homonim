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
import warnings

import cv2 as cv
import numpy as np
from rasterio.enums import Resampling
from rasterio.fill import fillnodata

from homonim import errors, utils
from homonim.enums import Model
from homonim.raster_array import RasterArray

ONdArray = np.ndarray | None
OShape = tuple[int, int] | None


def full_coverage_mask(
    proc_ra: RasterArray, other_ra: RasterArray, kernel_shape: tuple[int, int]
) -> RasterArray:
    """
    Find the full coverage mask.

    :param proc_ra:
        proc_crs array, in the proc_crs CRS and grid.
    :param other_ra:
        Other (non-proc_crs) array in the non-proc_crs CRS and grid.
    :param kernel_shape:
        Kernel (height, width) in pixels.

    :return:
        Full coverage mask as an array with uint8 dtype, in the proc_crs CRS and grid.
    """
    # re-project the other_ra mask into the proc_ra CRS and grid
    mask_ra = other_ra.mask_ra.reproject(
        **proc_ra.proj_profile, nodata=None, resampling=Resampling.average
    )
    # find the mask of fully covered other_ra & proc_ra pixels in the proc_ra CRS and
    # grid
    mask = mask_ra.array >= 1
    mask &= proc_ra.mask

    # Mask out partial kernel coverage.
    # Similar to the block overlap amount, this removes ceil(kernel_shape/2)
    # pixels from the nodata edge.  Note, that this is the strict approach for
    # proc_crs == ref, it could be floor(kernel_shape/2) for proc_crs == src,
    # which avoids the additional upsampling step.
    se = cv.getStructuringElement(
        cv.MORPH_RECT, (kernel_shape[1] + 2, kernel_shape[0] + 2)
    )
    mask_ra.array = cv.erode(
        mask.view('uint8'), se, borderType=cv.BORDER_CONSTANT, borderValue=0
    )
    return mask_ra


class KernelModel:
    default_kernel_shape = (5, 5)
    default_model = Model.gain_blk_offset

    def __init__(
        self,
        model: Model = default_model,
        kernel_shape: tuple[int, int] = default_kernel_shape,
        find_r2: bool = False,
        r2_inpaint_thresh: float = 0.25,
        mask_partial: bool = False,
        downsampling: Resampling = Resampling.average,
        upsampling: Resampling = Resampling.cubic_spline,
    ):
        """
        Base class for estimating and applying kernel model parameters, where the
        source and reference are in the same CRS and grid.

        Based on the paper: https://doi.org/10.1080/01431161.2018.1528404

        :param model:
            Correction model type.
        :param kernel_shape:
            Kernel (height, width) in pixels.
        :param find_r2:
            Whether to find R\N{SUPERSCRIPT TWO} (coefficient of determination) and
            include it with the parameters returned by :meth:`fit`.
        :param r2_inpaint_thresh:
            R\N{SUPERSCRIPT TWO} (coefficient of determination) threshold below which to
            interpolate ("in-paint") model offsets from surrounding values.  Applies
            to the :attr:`~enums.Model.gain_offset` model only.  If ``None``, no
            interpolation is performed.
        :param mask_partial:
            Whether to mask corrected pixels not produced by full kernel or source /
            reference image coverage.  Can help reduce seam-lines between overlapping
            images.
        :param downsampling:
            Resampling method to use when downsampling.
        :param upsampling:
            Resampling method to use when upsampling.
        """
        self._model = Model(model)
        kernel_shape = np.array(kernel_shape)
        if not np.all(kernel_shape >= 1) or not np.all(kernel_shape % 2 == 1):
            raise errors.HomonimError(
                "'kernel_shape' must integer, greater than or equal to one, and odd "
                'in both dimensions.'
            )
        if self._model is Model.gain_offset:
            if np.prod(kernel_shape) < 2:
                raise errors.HomonimError(
                    "'kernel_shape' should consist at least 2 pixels for the "
                    "'gain-offset' model."
                )
            elif np.prod(kernel_shape) < 25:
                warnings.warn(
                    "A 'kernel_shape' consiting of at least 25 pixels is recommended "
                    "for the 'gain-offset' model.",
                    category=errors.HomonimWarning,
                    stacklevel=2,
                )
        self._kernel_shape = tuple(kernel_shape.astype('int').tolist())
        self._find_r2 = find_r2
        self._r2_inpaint_thresh = r2_inpaint_thresh
        self._mask_partial = mask_partial
        self._downsampling = downsampling
        self._upsampling = upsampling

    @property
    def model(self) -> Model:
        """Correction model type."""
        return self._model

    @property
    def kernel_shape(self) -> tuple[int, int]:
        """Kernel (height, width) in pixels."""
        return self._kernel_shape

    @property
    def find_r2(self) -> bool:
        """Whether R\N{SUPERSCRIPT TWO} (coefficient of determination) will be found
        and included with the parameters.
        """
        return self._find_r2

    def _get_resampling(
        self, from_res: tuple[float, float], to_res: tuple[float, float]
    ) -> Resampling:
        """Return the resampling method for re-projecting from resolution
        ``from_res`` to resolution ``to_res``.
        """
        return (
            self._downsampling
            if np.prod(np.abs(from_res)) <= np.prod(np.abs(to_res))
            else self._upsampling
        )

    def _r2_array(
        self,
        ref_array: np.ndarray,
        src_array: np.ndarray,
        param_array: np.ndarray,
        mask: ONdArray = None,
        mask_sum: ONdArray = None,
        ref_sum: ONdArray = None,
        src_sum: ONdArray = None,
        ref2_sum: ONdArray = None,
        src2_sum: ONdArray = None,
        src_ref_sum: ONdArray = None,
        dest_array: ONdArray = None,
        kernel_shape: OShape = None,
    ) -> np.ndarray:
        """Return an R2 array, using keyword arguments instead of re-calculating them
        when they are given.
        """
        if kernel_shape is None:
            kernel_shape = self._kernel_shape
        kernel_shape = tuple(kernel_shape)  # force to tuple for opencv
        # common opencv arguments
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)

        # find the keyword arguments that were not provided
        if mask is None:
            # if mask is passed, assume that it has been applied to ref_array and
            # src_array, otherwise do that here
            mask = ~utils.nan_equals(
                src_array, RasterArray.default_nodata
            ) & ~utils.nan_equals(ref_array, RasterArray.default_nodata)
            ref_array[~mask] = 0
            src_array[~mask] = 0
        if mask_sum is None:
            mask_sum = cv.boxFilter(
                mask.astype(RasterArray.default_dtype),
                -1,
                kernel_shape[::-1],
                **filter_args,
            )
        if ref_sum is None:
            ref_sum = cv.boxFilter(ref_array, -1, kernel_shape[::-1], **filter_args)
        if ref2_sum is None:
            ref2_sum = cv.sqrBoxFilter(ref_array, -1, kernel_shape[::-1], **filter_args)
        if src2_sum is None:
            src2_sum = cv.sqrBoxFilter(src_array, -1, kernel_shape[::-1], **filter_args)
        if src_ref_sum is None:
            src_ref_sum = cv.boxFilter(
                src_array * ref_array, -1, kernel_shape[::-1], **filter_args
            )

        # R2 is found using:
        # R2 = 1 - (residual sum of squares)/(total sum of squares) = 1 - RSS/TSS
        # TSS = sum((ref - mean(ref))**2),
        # which can be expanded and expressed in terms of cv.boxFilter kernel sums as:
        ss_tot_array = (mask_sum * ref2_sum) - (ref_sum**2)

        if param_array.shape[0] > 1:
            # find RSS for model == Model.gain_offset
            if src_sum is None:
                src_sum = cv.boxFilter(src_array, -1, kernel_shape[::-1], **filter_args)

            # RSS = sum((ref - ref_hat)**2)
            #     = sum((ref - (m*src + c))**2),
            # where m and c are the first 2 bands of param_array. This can be expanded
            # and expressed in terms of cv.boxFilter kernel sums as:
            ss_res_array = (
                ((param_array[0] ** 2) * src2_sum)
                + (2 * np.prod(param_array[:2], axis=0) * src_sum)
                - (2 * param_array[0] * src_ref_sum)
                - (2 * param_array[1] * ref_sum)
                + ref2_sum
                + (mask_sum * (param_array[1] ** 2))
            )
        else:
            # find RSS for model == Model.gain or Model.gain_blk_offset

            # RSS = sum((ref - m*src)**2), where m is the first band of param_array
            # This can be expanded and expressed in terms of cv.boxFilter kernel sums
            # as:
            ss_res_array = (
                ((param_array[0] ** 2) * src2_sum)
                - (2 * param_array[0] * src_ref_sum)
                + ref2_sum
            )

        ss_res_array *= mask_sum

        if dest_array is None:
            # assign a destination array to write R2 into, if it was not provided
            dest_array = np.full(
                src_array.shape,
                fill_value=RasterArray.default_nodata,
                dtype=RasterArray.default_dtype,
            )

        # find R2 = 1 - RSS/TSS, and write into dest_array
        np.divide(ss_res_array, ss_tot_array, out=dest_array, where=mask)
        np.subtract(1, dest_array, out=dest_array, where=mask)
        return dest_array

    @staticmethod
    def _fit_block_norm(
        src_ra: RasterArray, ref_ra: RasterArray
    ) -> tuple[float, float]:
        """Return a two element (gain, offset) model to "normalise" the source array,
        so that the standard deviation and first percentile of the source and
        reference arrays match.  (Can be thought of as a basic dark object subtraction).
        """
        norm_model = [0.0, 0.0]
        mask = ref_ra.mask & src_ra.mask
        if not np.any(mask):
            return norm_model
        masked_src = src_ra.array[mask]
        masked_ref = ref_ra.array[mask]
        norm_model[0] = np.std(masked_ref) / np.std(masked_src)
        norm_model[1] = (
            np.percentile(masked_ref, 1) - np.percentile(masked_src, 1) * norm_model[0]
        )
        return tuple(norm_model)

    def _fit_gain(
        self, src_ra: RasterArray, ref_ra: RasterArray, kernel_shape: OShape = None
    ) -> RasterArray:
        """Find kernel gains, for a source & reference array.

        Returns an array of model parameters.  Gains in first band, offsets in the
        second, and optionally R2 in the third band when :attr:`find_r2` is True.
        """
        # adapted from https://www.mathsisfun.com/data/least-squares-regression.html
        # with c=0
        if kernel_shape is None:
            kernel_shape = self._kernel_shape
        kernel_shape = tuple(kernel_shape)  # force to tuple for opencv

        # mask invalid pixels with 0 so that these do not contribute to kernel sums
        # in *boxFilter()
        ref_array = ref_ra.array
        src_array = src_ra.array
        mask = ref_ra.mask & src_ra.mask
        ref_array[~mask] = 0
        src_array[~mask] = 0

        # set up a RasterArray profile for the parameters
        param_profile = src_ra.profile.copy()
        param_profile.update(
            count=3 if self._find_r2 else 2,
            nodata=RasterArray.default_nodata,
            dtype=RasterArray.default_dtype,
        )

        # convolve the kernel with src_array and ref_array to get kernel sums (uses
        # DFT for large kernels)
        # common opencv arguments
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)
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
                ref_array,
                src_array,
                param_ra.array[:1],
                mask=mask,
                ref_sum=ref_sum,
                src_sum=src_sum,
                dest_array=param_ra.array[2],
                kernel_shape=kernel_shape,
            )

        return param_ra

    def _fit_gain_blk_offset(
        self, src_ra: RasterArray, ref_ra: RasterArray, kernel_shape: OShape = None
    ) -> RasterArray:
        """Find kernel gains and block offset, for a source & reference array.

        Returns an array of model parameters.  Gains in first band, offsets in the
        second, and optionally R2 in the third band when :attr:`find_r2` is True.
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

    def _fit_gain_offset(
        self, src_ra: RasterArray, ref_ra: RasterArray, kernel_shape: OShape = None
    ) -> RasterArray:
        """Find kernel gains and offsets for a source & reference array.

        Returns an array of model parameters.  Gains in first band, offsets in the
        second, and optionally R2 in the third band when :attr:`find_r2` is True.
        """
        # Least squares formulae adapted from
        # https://www.mathsisfun.com/data/least-squares-regression.html
        if kernel_shape is None:
            kernel_shape = self._kernel_shape
        kernel_shape = tuple(kernel_shape)  # force to tuple for opencv

        # mask invalid pixels with 0 so that these do not contribute to kernel sums
        # in *boxFilter()
        ref_array = ref_ra.array
        src_array = src_ra.array
        mask = ref_ra.mask & src_ra.mask
        ref_array[~mask] = 0
        src_array[~mask] = 0

        # set up a RasterArray profile for the parameters
        param_profile = src_ra.profile.copy()
        find_r2 = self._find_r2 or (self._r2_inpaint_thresh is not None)
        param_profile.update(
            count=3 if find_r2 else 2,
            nodata=RasterArray.default_nodata,
            dtype=RasterArray.default_dtype,
        )

        # find the numerator for the gain i.e N*cov(ref, src)
        # common opencv arguments
        filter_args = dict(normalize=False, borderType=cv.BORDER_CONSTANT)
        src_sum = cv.boxFilter(src_array, -1, kernel_shape[::-1], **filter_args)
        ref_sum = cv.boxFilter(ref_array, -1, kernel_shape[::-1], **filter_args)
        src_ref_sum = cv.boxFilter(
            src_array * ref_array, -1, kernel_shape[::-1], **filter_args
        )
        mask_sum = cv.boxFilter(
            mask.astype(RasterArray.default_dtype, copy=False),
            -1,
            kernel_shape[::-1],
            **filter_args,
        )
        m_num_array = (mask_sum * src_ref_sum) - (src_sum * ref_sum)

        # find the denominator for the gain i.e. N*var(src)
        src2_sum = cv.sqrBoxFilter(src_array, -1, kernel_shape[::-1], **filter_args)
        m_den_array = (mask_sum * src2_sum) - (src_sum**2)

        # create parameter RasterArray filled with nodata
        param_ra = RasterArray.from_profile(None, param_profile)

        # find the gain = cov(ref, src) / var(src), avoiding divide by 0
        np.divide(m_num_array, m_den_array, out=param_ra.array[0], where=mask)

        # solve for the offset c = y - mx, given that the linear model passes through
        # (mean(ref_array), mean(src_array))
        np.divide(
            ref_sum - (param_ra.array[0] * src_sum),
            mask_sum,
            out=param_ra.array[1],
            where=mask,
        )

        if find_r2:
            # Find R2 of the sliding kernel models
            self._r2_array(
                ref_array,
                src_array,
                param_ra.array[:2],
                mask=mask,
                mask_sum=mask_sum,
                ref_sum=ref_sum,
                src_sum=src_sum,
                src2_sum=src2_sum,
                src_ref_sum=src_ref_sum,
                dest_array=param_ra.array[2],
                kernel_shape=kernel_shape,
            )

        if self._r2_inpaint_thresh is not None:
            # fill/inpaint low R2 and negative gain areas in the offset parameter
            r2_mask = (
                (param_ra.array[2] > self._r2_inpaint_thresh)
                & (param_ra.array[0] > 0)
                & mask
            )
            # NOTE: fillnodata does not release the GIL, so this can slow down
            # processing, especially for proc_crs=src
            param_ra.array[1] = fillnodata(param_ra.array[1], r2_mask)
            param_ra.mask = mask  # re-mask as nodata areas will have been filled above

            # recalculate the gain for the filled areas using m = (y - c)/x and and
            # the point (mean(ref_array), mean(src_array))
            r2_mask = ~r2_mask & mask
            np.divide(
                ref_sum - mask_sum * param_ra.array[1],
                src_sum,
                out=param_ra.array[0],
                where=r2_mask,
            )

        return param_ra

    def fit(self, src_ra: RasterArray, ref_ra: RasterArray) -> RasterArray:
        """
        Fit kernel models to source and reference arrays.

        :param src_ra:
            Source array.
        :param ref_ra:
            Reference array.

        :return:
            Model parameter array with gains in first band, offsets in the second,
            and optionally R\N{SUPERSCRIPT TWO} in the third band when
            :attr:`find_r2` is ``True``.
        """
        # TODO : include a CRS comparison below i.e. one that is faster that
        #  rasterio's current implementation?
        if (ref_ra.transform != src_ra.transform) or (ref_ra.shape != src_ra.shape):
            raise ValueError(
                "'ref_ra' and 'src_ra' must have the same CRS, transform and shape"
            )

        if self._model == Model.gain:
            param_ra = self._fit_gain(src_ra, ref_ra, kernel_shape=self._kernel_shape)
        elif self._model == Model.gain_blk_offset:
            param_ra = self._fit_gain_blk_offset(
                src_ra, ref_ra, kernel_shape=self._kernel_shape
            )
        else:
            param_ra = self._fit_gain_offset(
                src_ra, ref_ra, kernel_shape=self._kernel_shape
            )

        return param_ra

    def apply(self, src_ra: RasterArray, param_ra: RasterArray) -> RasterArray:
        """
        Apply kernel models to a source array.

        :param src_ra:
            Source array.
        :param param_ra:
            Model parameter array.

        :return:
            Corrected array.
        """
        if (param_ra.transform != src_ra.transform) or (param_ra.shape != src_ra.shape):
            raise ValueError(
                "'param_ra' and 'src_ra' must have the same CRS, transform and shape"
            )
        corr_array = (param_ra.array[0] * src_ra.array) + param_ra.array[1]
        corr_ra = RasterArray.from_profile(corr_array, param_ra.profile)
        return corr_ra


class RefSpaceModel(KernelModel):
    """Class for estimating and applying kernel model parameters, where the source
    and reference are not in the same CRS and grid.

    Parameters are estimated in the reference CRS and grid, and applied in the source
    image CRS and grid.
    """

    def fit(self, src_ra: RasterArray, ref_ra: RasterArray) -> RasterArray:
        # choose resampling method based on whether we are up- or downsampling
        resampling = self._get_resampling(src_ra.res, ref_ra.res)
        # downsample src_ra to reference CRS and grid
        src_ds_ra = src_ra.reproject(**ref_ra.proj_profile, resampling=resampling)
        # call base class fit with reference and source RasterArrays in the reference
        # CRS & grid
        return KernelModel.fit(self, src_ds_ra, ref_ra)

    def apply(self, src_ra, param_ra):
        # remove the R2 band of param_ra (to speed up the re-projection below)
        _param_ra = RasterArray.from_profile(param_ra.array[:2], param_ra.profile)
        # choose resampling method based on whether we are up- or downsampling
        resampling = self._get_resampling(_param_ra.res, src_ra.res)
        # re-project _param_ra to source CRS and grid
        param_src_ra = _param_ra.reproject(**src_ra.proj_profile, resampling=resampling)

        if self._mask_partial:
            # find the mask of fully covered pixels in reference CRS and grid
            mask_ra = full_coverage_mask(_param_ra, src_ra, self.kernel_shape)
            # re-project the mask to source CRS and grid, and apply to the parameters
            mask_src_ra = mask_ra.reproject(
                **src_ra.proj_profile,
                nodata=None,
                dtype='uint8',
                resampling=Resampling.nearest,
            )
            param_src_ra.mask = mask_src_ra.array.view('bool')
        else:
            param_src_ra.mask = src_ra.mask

        # call base class apply with source and parameter RasterArrays in the source
        # CRS & grid
        return KernelModel.apply(self, src_ra, param_src_ra)


class SrcSpaceModel(KernelModel):
    """Class for estimating and applying kernel model parameters, where the source
    and reference are not in the same CRS and grid.

    Parameters are estimated and applied in the source CRS and grid.
    """

    def fit(self, src_ra: RasterArray, ref_ra: RasterArray) -> RasterArray:
        # reproject ref_ra to the source CRS and grid
        resampling = self._get_resampling(ref_ra.res, src_ra.res)
        ref_src_ra = ref_ra.reproject(**src_ra.proj_profile, resampling=resampling)

        # copy the source to avoid in-place changes in fit() below
        _src_ra = src_ra.copy()
        # fit with source and parameter RasterArrays in the source CRS & grid
        param_ra = KernelModel.fit(self, _src_ra, ref_src_ra)

        if self._mask_partial:
            # remove R2 band from param_ra
            _param_ra = RasterArray.from_profile(param_ra.array[:2], param_ra.profile)
            # find the mask of fully covered pixels in source CRS and grid, and apply
            # to the parameters
            mask_ra = full_coverage_mask(_param_ra, ref_ra, self.kernel_shape)
            param_ra.mask = mask_ra.array.view('bool')
        else:
            param_ra.mask = src_ra.mask

        return param_ra
