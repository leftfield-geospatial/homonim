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
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from itertools import product
from os import PathLike, fspath
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
from rasterio.dtypes import can_cast_dtype
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import DatasetWriter
from tqdm.auto import tqdm

from homonim import utils
from homonim.enums import Driver, Model, ProcCrs
from homonim.errors import HomonimError
from homonim.kernel_model import KernelModel, RefSpaceModel, SrcSpaceModel
from homonim.matched_pair import MatchedPairReader
from homonim.raster_array import RasterArray
from homonim.raster_pair import BlockPair

logger = logging.getLogger(__name__)

# default output image creation options
_gtiff_creation_options = dict(
    tiled=True,
    blockxsize=512,
    blockysize=512,
    compress='deflate',
    interleave='band',
    photometric='minisblack',
    bigtiff='if_safer',
)
_cog_creation_options = dict(
    blocksize=512, compress='deflate', interleave='band', bigtiff='if_safer'
)


class RasterFuse(MatchedPairReader):
    def __init__(
        self,
        src_filename: str | PathLike,
        ref_filename: str | PathLike,
        proc_crs: str | ProcCrs = ProcCrs.auto,
        src_bands: tuple[int, ...] | None = None,
        ref_bands: tuple[int, ...] | None = None,
        force: bool = False,
    ):
        """
        Class to correct a source image to surface reflectance by fusion with a
        reference.

        For best results, source and reference images should be concurrent.
        Reference extents must encompass those of the source.

        The reference should contain bands that are approximate wavelength matches to
        the source bands.  When source and reference bands are RGB, or have
        ``center_wavelength`` tags, bands are matched automatically based on
        wavelength.  Otherwise, source and reference bands are assumed to be in
        matching order.  Subsets and ordering of bands can be specified with the
        ``src_bands`` and ``ref_bands`` parameters.

        :param src_filename:
            Path or URI of a source image.
        :param ref_filename:
            Path or URI of a reference image.
        :param proc_crs:
            Which of the source or reference CRS and pixel grids should be used for
            estimating correction parameters.  By default, the CRS and pixel grid with
            the lowest resolution is used (recommended).
        :param src_bands:
            Indexes of source bands to be corrected (1 based).  Defaults to all bands
            with a ``center_wavelength`` tag if any exist, otherwise to all non-alpha
            bands.
        :param ref_bands:
            Indexes of reference bands to match with source bands (1 based).
            Defaults to all bands with a ``center_wavelength`` tag if any exist,
            otherwise to all non-alpha bands.
        :param force:
            Whether to bypass wavelength band matching, and match source and reference
            bands in their given order.
        """
        super().__init__(
            src_filename,
            ref_filename,
            proc_crs=proc_crs,
            src_bands=src_bands,
            ref_bands=ref_bands,
            force=force,
        )
        self._corr_lock = threading.Lock()
        self._param_lock = threading.Lock()

    def _create_corr_profile(
        self,
        driver: str | Driver,
        dtype: str,
        nodata: int | float | None,
        creation_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a RasterIO profile for the corrected image."""
        driver = Driver(driver.lower())
        with np.errstate(invalid='ignore'):
            if nodata is not None and not can_cast_dtype(nodata, dtype):
                raise HomonimError(
                    f"'nodata' value: {nodata} cannot be safely cast to 'dtype': '"
                    f"{dtype}'"
                )
        creation_options = creation_options or (
            _gtiff_creation_options if driver is Driver.gtiff else _cog_creation_options
        )
        return dict(
            driver=str(driver),
            width=self.src_im.width,
            height=self.src_im.height,
            count=len(self.src_bands),
            dtype=str(dtype),
            nodata=nodata,
            crs=self.src_im.crs,
            transform=self.src_im.transform,
            **creation_options,
        )

    def _create_param_profile(self) -> dict[str, Any]:
        """Return a RasterIO profile for the parameter image."""
        proc_im = self.ref_im if self.proc_crs is ProcCrs.ref else self.src_im
        return dict(
            driver='GTiff',
            width=proc_im.width,
            height=proc_im.height,
            count=len(self.src_bands) * 3,
            dtype=RasterArray.default_dtype,
            nodata=RasterArray.default_nodata,
            crs=proc_im.crs,
            transform=proc_im.transform,
            **_gtiff_creation_options,
        )

    def _set_image_tags(self, im: DatasetWriter, **kwargs):
        """Set image tags from ``kwargs`` and RasterPairReader attributes."""
        kwargs_tags = {
            f'FUSE_{k.upper()}': getattr(v, 'name', str(v)) for k, v in kwargs.items()
        }
        im.update_tags(
            FUSE_SRC_FILE=Path(os.fspath(self._src_filename)).name,
            FUSE_REF_FILE=Path(os.fspath(self._ref_filename)).name,
            FUSE_PROC_CRS=self.proc_crs.name,
            **kwargs_tags,
        )

    def _set_corr_band_tags(self, im: DatasetWriter):
        """Copy band tags from the reference to the corrected image."""
        # TODO: should tags not first come from the source if they exist,
        #  then from the reference otherwise?  also for parameter band descriptions
        #  below...
        geedim_tags = ['center_wavelength', 'name', 'description']
        for corr_band, ref_band in enumerate(self.ref_bands, start=1):
            im.set_band_description(corr_band, self.ref_im.descriptions[ref_band - 1])
            ref_band_tags = self.ref_im.tags(ref_band)
            ref_band_tags = {k: v for k, v in ref_band_tags.items() if k in geedim_tags}
            im.update_tags(corr_band, **ref_band_tags)

    def _set_param_band_tags(self, im: DatasetWriter):
        """Set parameter image band tags."""
        for param_band, (param_name, ref_band) in enumerate(
            product(['GAIN', 'OFFSET', 'R2'], self.ref_bands), start=1
        ):
            ref_desc = self.ref_im.descriptions[ref_band - 1] or f'B{ref_band}'
            im.set_band_description(param_band, f'{ref_desc}_{param_name}')

    @staticmethod
    def _build_overviews(
        im: DatasetWriter, max_num_levels: int = 8, min_level_pixels: int = 256
    ):
        """Build internal overviews for an open rasterio dataset.  Each overview
        level is decimated by a factor of 2.  The number of overview levels is
        determined by whichever of the ``max_num_levels`` or ``min_level_pixels``
        limits is reached first.
        """
        max_ovw_levels = int(np.min(np.log2(im.shape)))
        min_level_shape_pow2 = int(np.log2(min_level_pixels))
        num_ovw_levels = np.min([max_num_levels, max_ovw_levels - min_level_shape_pow2])
        ovw_levels = [2**m for m in range(1, num_ovw_levels + 1)]
        im.build_overviews(ovw_levels, Resampling.average)

    def _process_block(
        self,
        block_pair: BlockPair,
        model: KernelModel,
        corr_im: DatasetWriter,
        param_im: DatasetWriter | None = None,
    ):
        """Thread-safe method to correct an image block to surface reflectance using
        ``model``.  Corrected, and optionally parameter, blocks are written to the
        supplied image dataset(s).
        """
        # read source and reference blocks
        src_ra, ref_ra = self.read(block_pair)
        # fit and apply the sliding kernel models
        param_ra = model.fit(src_ra, ref_ra)
        corr_ra = model.apply(src_ra, param_ra)

        # write the corrected block
        with self._corr_lock:
            corr_ra.to_rio_dataset(
                corr_im, indexes=block_pair.band_i + 1, window=block_pair.src_out_block
            )

        if param_im:
            # write the parameter block
            with self._param_lock:
                indexes = range(
                    block_pair.band_i + 1, param_im.count + 1, len(self.src_bands)
                )
                param_out_block = (
                    block_pair.ref_out_block
                    if self.proc_crs is ProcCrs.ref
                    else block_pair.src_out_block
                )
                param_ra.to_rio_dataset(
                    param_im, indexes=indexes, window=param_out_block
                )

    @staticmethod
    def create_model_config(
        r2_inpaint_thresh: float = 0.25,
        mask_partial: bool = False,
        downsampling: Resampling = Resampling.average,
        upsampling: Resampling = Resampling.cubic_spline,
    ) -> dict[str, Any]:
        """
        Return a model configuration that can be passed as the ``model_config``
        argument to :meth:`~RasterFuse.process`.

        .. deprecated:: 0.5.0

            This method is deprecated and will be removed in a future release. Please
            pass the arguments to :meth:`~RasterFuse.process` directly.

        :param r2_inpaint_thresh:
            R\N{SUPERSCRIPT TWO} (coefficient of determination) threshold below which
            to interpolate ("in-paint") model offsets from surrounding values.
            Applies to the :attr:`~homonim.enums.Model.gain_offset` model only.  If
            ``None``, no interpolation is performed.
        :param mask_partial:
            Whether to mask corrected pixels not produced by full kernel or source /
            reference image coverage.  Can help reduce seam-lines between overlapping
            images.
        :param downsampling:
            Resampling method to use when downsampling.
        :param upsampling:
            Resampling method to use when upsampling.

        :return:
            Configuration dictionary.
        """
        warnings.warn(
            'This method is deprecated and will be removed in a future release. '
            "Please pass the arguments to 'RasterFuse.process()' directly.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return dict(
            r2_inpaint_thresh=r2_inpaint_thresh,
            mask_partial=mask_partial,
            downsampling=downsampling,
            upsampling=upsampling,
        )

    @staticmethod
    def create_block_config(
        threads: int = 0, max_block_mem: float = 100
    ) -> dict[str, Any]:
        """
        Return a block processing configuration that can be passed as the
        ``block_config`` argument to :meth:`~RasterFuse.process`.

        .. deprecated:: 0.5.0

            This method is deprecated and will be removed in a future release. Please
            pass the arguments to :meth:`~RasterFuse.process` directly.

        :param threads:
            Number of image blocks to process concurrently.  If ``0``, the number of
            CPUs is used.
        :param max_block_mem:
            Maximum size of an image block in megabytes.  If ``0``, a block will
            correspond to a whole image band.

        :return:
            Configuration dictionary.
        """
        warnings.warn(
            'This method is deprecated and will be removed in a future release. '
            "Please pass the arguments to 'RasterFuse.process()' directly.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return dict(
            threads=utils.validate_threads(threads), max_block_mem=max_block_mem
        )

    @staticmethod
    def create_out_profile(
        driver: str | Driver = Driver.gtiff,
        dtype: str = RasterArray.default_dtype,
        nodata: int | float | None = RasterArray.default_nodata,
        creation_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Return a profile for the corrected image that can be passed as the
        ``out_profile`` argument to :meth:`~RasterFuse.process`.

        .. deprecated:: 0.5.0

            This method is deprecated and will be removed in a future release. Please
            pass the arguments to :meth:`~RasterFuse.process` directly.

        :param driver:
            Format driver.
        :param dtype:
            Data type (``uint8``, ``uint16``, ``int16``, ``uint32``, ``int32``,
            ``float32`` or ``float64``).
        :param nodata:
            Nodata value.  If ``None``, an internal mask is written (recommended when
            ``creation_options`` are configured for lossy, e.g. JPEG, compression).
        :param creation_options:
            Driver specific creation options as a dictionary of ``name: value``
            pairs.  See the GDAL `GTiff
            <https://gdal.org/en/latest/drivers/raster/gtiff.html#creation
            -options>`__ and `COG <https://gdal.org/en/latest/drivers/raster/cog.html
            #creation -options>`__ documentation for details on the options for those
            drivers.  If ``None``, default options are used.

        :return:
            Profile dictionary.
        """
        warnings.warn(
            'This method is deprecated and will be removed in a future release. '
            "Please pass the arguments to 'RasterFuse.process()' directly.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        driver = Driver(driver.lower())
        creation_options = creation_options or (
            _gtiff_creation_options if driver is Driver.gtiff else _cog_creation_options
        )
        return dict(
            driver=driver, dtype=dtype, nodata=nodata, creation_options=creation_options
        )

    def process(
        self,
        corr_filename: str | PathLike,
        model: Model = KernelModel.default_model,
        kernel_shape: tuple[int, int] = KernelModel.default_kernel_shape,
        param_filename: str | PathLike | None = None,
        build_ovw: bool = True,
        overwrite: bool = False,
        model_config: dict[str, Any] | None = None,
        out_profile: dict[str, Any] | None = None,
        block_config: dict[str, Any] | None = None,
        *,
        r2_inpaint_thresh: float = 0.25,
        mask_partial: bool = False,
        downsampling: Resampling = Resampling.average,
        upsampling: Resampling = Resampling.cubic_spline,
        driver: str | Driver = Driver.gtiff,
        dtype: str = RasterArray.default_dtype,
        nodata: int | float | None = RasterArray.default_nodata,
        creation_options: dict[str, Any] | None = None,
        threads: int = 0,
        max_block_mem: float = 100,
    ):
        """
        Correct the source image to surface reflectance.

        :param corr_filename:
            Path or URI of the corrected image.
        :param model:
            Correction model type to use.
        :param kernel_shape:
            Kernel (height, width) in pixels of the :attr:`proc_crs` image.
        :param param_filename:
            Optional path or URI of a GeoTIFF image to write with correction
            parameters and their R\N{SUPERSCRIPT TWO} values.  If ``None`` (the
            default), no parameter image is written.
        :param build_ovw:
            Whether to build overviews for the output image(s).
        :param overwrite:
            Whether to overwrite output image(s) if they exist.
        :param model_config:
            Correction model configuration as returned by :meth:`create_model_config`.

            .. deprecated:: 0.5.0

                This parameter will be removed in a future release.  Please pass the
                :meth:`create_model_config` arguments to this method directly.

        :param out_profile:
            Profile for the corrected image as returned by :meth:`create_out_profile`.

            .. deprecated:: 0.5.0

                This parameter will be removed in a future release.  Please pass the
                :meth:`create_out_profile` arguments to this method directly.

        :param block_config:
            Block processing configuration as returned by :meth:`create_block_config`.

            .. deprecated:: 0.5.0

                This parameter will be removed in a future release.  Please pass the
                :meth:`create_block_config` arguments to this method directly.

        :param r2_inpaint_thresh:
            R\N{SUPERSCRIPT TWO} (coefficient of determination) threshold below which
            to interpolate ("in-paint") model offsets from surrounding values.
            Applies to the :attr:`~homonim.enums.Model.gain_offset` model only.  If
            ``None``, no interpolation is performed.
        :param mask_partial:
            Whether to mask corrected pixels not produced by full kernel or source /
            reference image coverage.  Can help reduce seam-lines between overlapping
            images.
        :param downsampling:
            Resampling method to use when downsampling.
        :param upsampling:
            Resampling method to use when upsampling.
        :param driver:
            Corrected image format driver.
        :param dtype:
            Corrected image data type (``uint8``, ``uint16``, ``int16``, ``uint32``,
            ``int32``, ``float32`` or ``float64``).
        :param nodata:
            Corrected image nodata value.  Should be representable by ``dtype`` . If
            ``None``, an internal mask is written (recommended when
            ``creation_options`` are configured for lossy, e.g. JPEG, compression).
        :param creation_options:
            Driver specific creation options as a dictionary of ``name: value``
            pairs.  See the GDAL `GTiff
            <https://gdal.org/en/latest/drivers/raster/gtiff.html#creation
            -options>`__ and `COG <https://gdal.org/en/latest/drivers/raster/cog.html
            #creation -options>`__ documentation for details on the options for those
            drivers.  If ``None``, default options are used.
        :param threads:
            Number of image blocks to process concurrently.  If ``0``, the number of
            CPUs is used.
        :param max_block_mem:
            Maximum size of an image block in megabytes.  If ``0``, a block will
            correspond to the whole image band.
        """
        # TODO: is it possible to have an auto block_config that adjusts threads and
        #  block mem to available memory
        self._assert_open()
        model_type = Model(model)
        if threads > os.cpu_count():
            raise HomonimError(
                "'threads' should be less than or equal to the number of CPUs"
            )
        threads = threads if threads > 0 else os.cpu_count()

        # convert deprecated *_config argument items to keyword arguments
        warn_msg = (
            "The '{}' parameter is deprecated and will be removed in a future "
            'release. Please pass its items as keyword arguments to '
            "'RasterFuse.process()' directly."
        )
        if model_config:
            warnings.warn(
                warn_msg.format('model_config'),
                category=DeprecationWarning,
                stacklevel=2,
            )
            model_config = self.create_model_config(**model_config)
            r2_inpaint_thresh = model_config['r2_inpaint_thresh']
            mask_partial = model_config['mask_partial']
            downsampling = model_config['downsampling']
            upsampling = model_config['upsampling']

        if out_profile:
            warnings.warn(
                warn_msg.format('out_profile'),
                category=DeprecationWarning,
                stacklevel=2,
            )
            out_profile = self.create_out_profile(**out_profile)
            driver = out_profile['driver']
            dtype = out_profile['dtype']
            nodata = out_profile['nodata']
            creation_options = out_profile['creation_options']

        if block_config:
            warnings.warn(
                warn_msg.format('block_config'),
                category=DeprecationWarning,
                stacklevel=2,
            )
            block_config = self.create_block_config(**block_config)
            threads = block_config['threads']
            max_block_mem = block_config['max_block_mem']

        # create the KernelModel according to proc_crs
        model_cls = SrcSpaceModel if self.proc_crs == ProcCrs.src else RefSpaceModel
        model_kwargs = dict(
            r2_inpaint_thresh=r2_inpaint_thresh,
            mask_partial=mask_partial,
            downsampling=downsampling,
            upsampling=upsampling,
        )
        model = model_cls(
            model, kernel_shape, find_r2=param_filename is not None, **model_kwargs
        )

        # tqdm progress bar format
        bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} blocks [{elapsed}<{remaining}]'

        # open the output files and set their tags
        # TODO: this does not work with URIs
        if not overwrite and Path(corr_filename).exists():
            raise FileExistsError(f"Corrected image exists: '{fspath(corr_filename)}'")
        if not overwrite and param_filename and Path(param_filename).exists():
            raise FileExistsError(f"Parameter image exists: '{fspath(param_filename)}'")

        with ExitStack() as stack:
            corr_profile = self._create_corr_profile(
                driver=driver,
                dtype=dtype,
                nodata=nodata,
                creation_options=creation_options,
            )
            corr_im = stack.enter_context(rio.open(corr_filename, 'w', **corr_profile))
            corr_im.colorinterp = [
                self.src_im.colorinterp[sb - 1] for sb in self.src_bands
            ]
            tag_kwargs = dict(
                model=model_type,
                kernel_shape=kernel_shape,
                **model_kwargs,
                max_block_mem=max_block_mem,
            )
            self._set_image_tags(corr_im, **tag_kwargs)
            self._set_corr_band_tags(corr_im)

            if param_filename:
                param_profile = self._create_param_profile()
                param_im = stack.enter_context(
                    rio.open(param_filename, 'w', **param_profile)
                )
                self._set_image_tags(param_im, **tag_kwargs)
                self._set_param_band_tags(param_im)
            else:
                param_im = None

            # ignore NotGeoreferencedWarning from RasterArray.reproject()
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter('ignore', category=NotGeoreferencedWarning)

            # correct blocks in a thread pool
            overlap = utils.overlap_for_kernel(kernel_shape)
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=threads))
            futures = [
                executor.submit(
                    self._process_block, block_pair, model, corr_im, param_im
                )
                for block_pair in self.block_pairs(overlap, max_block_mem)
            ]

            for future in tqdm(
                as_completed(futures),
                bar_format=bar_format,
                total=len(futures),
                dynamic_ncols=True,
            ):
                try:
                    future.result()
                except Exception as ex:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError('Could not correct block.') from ex

            if build_ovw:
                self._build_overviews(corr_im)
                if param_im:
                    self._build_overviews(param_im)
