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

import json
import logging
import pathlib
import re
import warnings
from contextlib import contextmanager
from timeit import default_timer as timer

import click
import cloup
import numpy as np
import rasterio as rio
import yaml
from click.core import ParameterSource
from rasterio.errors import NotGeoreferencedWarning
from tqdm.auto import tqdm
from tqdm.contrib.logging import _TqdmLoggingHandler

from homonim import (
    Model,
    ParamStats,
    ProcCrs,
    RasterCompare,
    RasterFuse,
    utils,
)
from homonim.errors import ImageFormatError
from homonim.kernel_model import KernelModel
from homonim.raster_array import RasterArray
from homonim.version import __version__

logger = logging.getLogger(__name__)


class HomonimCommand(cloup.Command):
    """cloup.Command subclass for formatting help with RST markup."""

    def get_help(self, ctx: click.Context):
        """Strip some RST markup from the help text for CLI display.  Will not work with grid tables."""

        # Note that this can't easily be done in __init__, as each sub-command's __init__ gets called,
        # which ends up re-assigning self.wrap_text to reformat_text
        if not hasattr(self, 'wrap_text'):
            self.wrap_text = cloup.formatting._formatter.wrap_text
        sub_strings = {
            '\b\n': '\n\b',  # convert from RST friendly to click literal (unwrapped) block marker
            r'\| ': '',  # strip RST literal (unwrapped) marker in e.g. tables and bullet lists
            r'\n\.\. _.*:\n': '',  # strip RST ref directive '\n.. _<name>:\n'
            '::': ':',  # convert from RST '::' to ':'
            '``(.*?)``': r'\g<1>',  # convert from RST '``literal``' to 'literal'
            ':option:`(.*?)( <.*?>)?`': r'\g<1>',  # convert ':option:`--name <group-command --name>`' to '--name'
            ':option:`(.*?)`': r'\g<1>',  # convert ':option:`--name`' to '--name'
            '`([^<]*) <([^>]*)>`_': r'\g<1>',  # convert from RST cross-ref '`<name> <<link>>`_' to 'name'
        }

        def reformat_text(text: str, width: int, **kwargs):
            for sub_key, sub_value in sub_strings.items():
                text = re.sub(sub_key, sub_value, text, flags=re.DOTALL)
            wr_text = self.wrap_text(text, width, **kwargs)
            # change double newline to single newline separated list
            return re.sub(r'\n\n(\s*?)- ', '\n- ', wr_text, flags=re.DOTALL)

        cloup.formatting._formatter.wrap_text = reformat_text
        return cloup.Command.get_help(self, ctx)


class FuseCommand(HomonimCommand):
    """click.Command subclass for setting fuse command parameters from a yaml config file."""

    def invoke(self, ctx: click.Context):
        """Merge config file with command line and default parameter values."""
        # adapted from https://stackoverflow.com/questions/46358797/python-click-supply-arguments-and-options-from-a
        # -configuration-file/46391887
        config_file = ctx.params['conf']
        if config_file is not None:
            # read the config file into a dict
            with open(config_file) as f:
                config_dict = yaml.safe_load(f)

            for conf_key, conf_value in config_dict.items():
                if conf_key not in ctx.params:
                    raise click.BadParameter(
                        f'Unknown config file parameter "{conf_key}"',
                        ctx=ctx,
                        param_hint='conf',
                    )
                else:
                    param_src = ctx.get_parameter_source(conf_key)
                    # overwrite parameters not specified on command line with config file values
                    if (
                        ctx.params[conf_key] is None
                        or param_src == ParameterSource.DEFAULT
                    ):
                        ctx.params[conf_key] = conf_value
                        ctx.set_parameter_source(conf_key, ParameterSource.COMMANDLINE)

        # Set the default creation_options if no other driver or creation_options have been specified.
        # This can't be done in a callback as it depends on --driver.
        if (
            ctx.get_parameter_source('driver') == ParameterSource.DEFAULT
            and ctx.get_parameter_source('creation_options') == ParameterSource.DEFAULT
        ):
            ctx.params['creation_options'] = RasterFuse.create_out_profile()[
                'creation_options'
            ]

        return click.Command.invoke(self, ctx)


def _update_existing_keys(default_dict: dict, **kwargs) -> dict:
    """Update values in a dict with args from matching keys in **kwargs."""
    return {k: kwargs.get(k, v) for k, v in default_dict.items()}


@contextmanager
def _configure_logging(verbosity: int):
    """Context manager to configure logging level, redirect warnings to the logger
    and logs to ``tqdm.write()``.
    """
    # configure the package logger (adapted from rasterio:
    # https://github.com/rasterio/rasterio/blob/main/rasterio/rio/main.py)
    pkg_logger = logging.getLogger(__package__)
    pkg_log_level = pkg_logger.level
    pkg_logger.setLevel(max(10, 20 - 10 * verbosity))
    # route logs through tqdm handler so they don't interfere with progress bars
    handler = _TqdmLoggingHandler(tqdm_class=tqdm)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    pkg_logger.addHandler(handler)

    warnings_showwarning = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        """Log warnings with the package logger."""
        pkg_logger.warning(message)

    try:
        # redirect warnings to package logger
        warnings.showwarning = showwarning
        yield
    finally:
        # restore the initial state
        warnings.showwarning = warnings_showwarning
        pkg_logger.removeHandler(handler)
        pkg_logger.setLevel(pkg_log_level)


def _threads_cb(ctx: click.Context, param: click.Option, value):
    """click callback to validate --threads."""
    try:
        threads = utils.validate_threads(value)
    except Exception as ex:
        raise click.BadParameter(str(ex)) from None
    return threads


def _nodata_cb(ctx: click.Context, param: click.Option, value: str):
    """click callback to convert --nodata value to None, nan or float."""
    # adapted from rasterio https://github.com/rasterio/rasterio
    if value is None or value.lower() in ['null', 'nil', 'none', 'nada']:
        return None
    else:
        # check value is a number and can be cast to output dtype
        try:
            value = float(value.lower())
        except (TypeError, ValueError):
            raise click.BadParameter(
                f'{value} is not a number', param=param, param_hint='--nodata'
            ) from None

        return value


def _creation_options_cb(ctx: click.Context, param: click.Option, value):
    """
    click callback to validate and parse multiple creation options (e.g. `-co KEY1=VAL1 -co KEY2=VAL2).
    Note: `==VAL` breaks this as `str.split('=', 1)` is used.
    """
    # adapted from rasterio https://github.com/rasterio/rasterio
    if not value:
        return {}
    else:
        out = {}
        for pair in value:
            if '=' not in pair:
                raise click.BadParameter(f'Invalid syntax for KEY=VAL arg: {pair}')
            else:
                k, v = pair.split('=', 1)
                k = k.lower()
                v = v.lower()
                out[k] = (
                    None
                    if v.lower() in ['none', 'null', 'nil', 'nada']
                    else yaml.safe_load(v)
                )
        return out


def _param_file_cb(ctx: click.Context, param: click.Argument, value):
    """click callback to validate parameter image file(s)."""
    for filename in value:
        filename = pathlib.Path(filename)
        try:
            utils.validate_param_image(filename)
        except (FileNotFoundError, ImageFormatError):
            raise click.BadParameter(
                f'{filename.name} is not a valid parameter image.', param=param
            ) from None
    return value


# define click options and arguments common to more than one command
# use cloup's argument to auto print argument help on command line
# TODO: rasterio 1.4 does not accept Path wrapped URLs
ref_file_arg = cloup.argument(
    'ref-file',
    nargs=1,
    metavar='REFERENCE',
    type=click.Path(exists=False, dir_okay=False, path_type=pathlib.Path),
    help='Path or URL of a reference image.',
)
threads_option = click.option(
    '-t',
    '--threads',
    type=click.INT,
    default=RasterFuse.create_block_config()['threads'],
    show_default=True,
    callback=_threads_cb,
    help='Number of image blocks to process concurrently (0 = use all processors).',
)
max_block_mem_option = click.option(
    '-mbm',
    '--max-block-mem',
    type=click.FLOAT,
    default=RasterFuse.create_block_config()['max_block_mem'],
    show_default=True,
    help='Maximum image block size in megabytes (0 = block corresponds to a whole band).',
)
downsampling_option = click.option(
    '-ds',
    '--downsampling',
    type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
    default=RasterFuse.create_model_config()['downsampling'].name,
    show_default=True,
    help='Resampling method for re-projecting from high to low resolution.  See the `rasterio docs '
    '<https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_ for '
    'details.',
)
upsampling_option = click.option(
    '-us',
    '--upsampling',
    type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
    default=RasterFuse.create_model_config()['upsampling'].name,
    show_default=True,
    help='Resampling method for re-projecting from low to high resolution.  See the `rasterio docs '
    '<https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_ for '
    'details.',
)
output_option = click.option(
    '-op',
    '--output',
    type=click.Path(
        exists=False, dir_okay=False, writable=True, path_type=pathlib.Path
    ),
    default=None,
    help='Write results to this json file.',
)
src_bands_option = click.option(
    '-sb',
    '--src-band',
    'src_bands',
    type=click.INT,
    multiple=True,
    default=None,
    show_default='all spectral or non-alpha bands.',
    help='Source band index(es) to process (1 based).',
)
ref_bands_option = click.option(
    '-rb',
    '--ref-band',
    'ref_bands',
    type=click.INT,
    multiple=True,
    default=None,
    show_default='all spectral or non-alpha bands.',
    help='Reference band index(es) to match with source band(s) (1 based).',
)
# TODO: here the term 'spectral' is used to refer to center_wavelength tagged bands,
#  but it is not clear what that means
force_match_option = click.option(
    '-f',
    '--force-match',
    is_flag=True,
    default=False,
    show_default=True,
    help='Bypass auto wavelength matching, and any band-matching errors.  Use with caution.',
)
""" cloup context settings to print help in 'linear' layout with heading/option emphasis. """
context_settings = cloup.Context.settings(
    formatter_settings=cloup.HelpFormatter.settings(
        col2_min_width=np.inf,
        theme=cloup.HelpTheme(
            invoked_command=cloup.Style(fg='bright_white', bold=True),
            heading=cloup.Style(fg='bright_white', bold=True),
            col1=cloup.Style(fg='bright_white'),
        ),
    )
)


# define the click CLI
@cloup.group(context_settings=context_settings)
@click.option('--verbose', '-v', count=True, help='Increase verbosity.')
@click.option('--quiet', '-q', count=True, help='Decrease verbosity.')
@click.version_option(version=__version__, message='%(version)s')
@click.pass_context
def cli(ctx: click.Context, verbose: int, quiet: int):
    """Surface reflectance correction toolkit."""
    ctx.with_resource(_configure_logging(verbose - quiet))


# fuse command
@cli.command(cls=FuseCommand)
# standard options
@cloup.argument(
    'src-file',
    nargs=-1,
    metavar='SOURCE...',
    type=click.Path(exists=False, dir_okay=False, path_type=pathlib.Path),
    help='Path/URL(s) of source image(s) to be corrected.',
)
@ref_file_arg
@cloup.option_group(
    'Standard options',
    # note: either use click.option(...), or cloup.option(..., help=inspect.cleandoc(...)) for RST help strings,
    # if cloup's mutually exclusive etc. functionality is needed, it should be the latter.
    click.option(
        '-m',
        '--model',
        type=click.Choice([m.value for m in Model], case_sensitive=False),
        default=KernelModel.default_model.value,
        show_default=True,
        help="""Correction model.

        - `gain`: Gain-only model, suitable for haze-free and zero offset images.

        - `gain-blk-offset`: Gain-only model applied to offset normalised blocks.  Suitable for most source-reference combinations.

        - `gain-offset`: Gain and offset model.  Most accurate model, but sensitive to differences between source and reference.
        """,
    ),
    click.option(
        '-k',
        '--kernel-shape',
        type=click.Tuple([click.INT, click.INT]),
        nargs=2,
        default=KernelModel.default_kernel_shape,
        show_default=True,
        metavar='HEIGHT WIDTH',
        help='Kernel height and width in pixels of the :option:`--proc-crs <homonim-fuse --proc-crs>` image. Larger '
        'kernels are less susceptible to over-fitting, but provide lower resolution correction.',
    ),
    src_bands_option,
    ref_bands_option,
    # TODO: allow URIs?
    click.option(
        '-od',
        '--out-dir',
        type=click.Path(exists=True, file_okay=False, writable=True),
        show_default='source image directory.',
        help='Directory in which to place corrected image(s).',
    ),
    click.option(
        '-o',
        '--overwrite',
        is_flag=True,
        default=False,
        show_default=True,
        help='Overwrite existing output file(s).',
    ),
    # TODO: does this work in front of an argument?  or should it be handles like oty rpc's --gcp-refine?
    click.option(
        '-cmp',
        '--compare',
        'cmp_file',
        metavar='FILE',
        type=click.Path(exists=False, dir_okay=False, path_type=pathlib.Path),
        is_flag=False,
        flag_value='ref',
        help='Compare source and corrected images with this reference image.  If no ``FILE`` value is given, source '
        'and corrected images are compared with :option:`REFERENCE`.',
    ),
    click.option(
        '-cb',
        '--cmp-band',
        'cmp_bands',
        type=click.INT,
        multiple=True,
        default=None,
        show_default='all spectral or non-alpha bands.',
        help='Comparison reference band index(es) that correspond (spectrally) to '
        ':option:`--src-band <homonim-fuse --src-band>` (s).',
    ),
    click.option(
        '-bo/-nbo',
        '--build-ovw/--no-build-ovw',
        type=click.BOOL,
        default=True,
        show_default=True,
        help='Build overviews for the output image(s).',
    ),
    click.option(
        '-c',
        '--conf',
        type=click.Path(
            exists=True, dir_okay=False, readable=True, path_type=pathlib.Path
        ),
        required=False,
        default=None,
        show_default=True,
        help='Path to a yaml configuration file specifying advanced options (as follow below).',
    ),
)
# advanced options
@cloup.option_group(
    'Advanced options',
    click.option(
        '-pi/-npi',
        '--param-image/--no-param-image',
        type=click.BOOL,
        default=False,
        show_default=True,
        help='Write the  model parameters and R\N{SUPERSCRIPT TWO} values for each corrected image to a parameter '
        'image file.',
    ),
    click.option(
        '-mp/-nmp',
        '--mask-partial/--no-mask-partial',
        type=click.BOOL,
        default=RasterFuse.create_model_config()['mask_partial'],
        show_default=True,
        help='Mask output pixels produced from partial kernel or source / reference image coverage.',
    ),
    threads_option,
    max_block_mem_option,
    downsampling_option,
    upsampling_option,
    click.option(
        '-rit',
        '--r2-inpaint-thresh',
        type=click.FloatRange(min=0, max=1),
        default=RasterFuse.create_model_config()['r2_inpaint_thresh'],
        show_default=True,
        metavar='FLOAT 0-1',
        help='R\N{SUPERSCRIPT TWO} threshold below which to inpaint model parameters from surrounding areas '
        '(0 = turn off inpainting). Valid for `gain-offset` :option:`--model` only.',
    ),
    click.option(
        '-pc',
        '--proc-crs',
        type=click.Choice([pc.value for pc in ProcCrs], case_sensitive=False),
        default=ProcCrs.auto.value,
        show_default=True,
        help="""The image CRS in which to estimate correction parameters.
        \b

        - `auto`: lowest resolution of the source and reference CRS's (recommended).
        - `src`: source image CRS.
        - `ref`: reference image CRS.
        """,
    ),
    click.option(
        '--driver',
        type=click.Choice(
            tuple(set(rio.drivers.raster_driver_extensions().values())),
            case_sensitive=False,
        ),
        default=RasterFuse.create_out_profile()['driver'],
        show_default=True,
        metavar='TEXT',
        help='Output image format driver.  See the `GDAL docs '
        '<https://gdal.org/en/stable/drivers/raster/index.html>`_ for details.',
    ),
    click.option(
        '--dtype',
        type=click.Choice(
            list(rio.dtypes.dtype_fwd.values())[1:8], case_sensitive=False
        ),
        default=RasterFuse.create_out_profile()['dtype'],
        show_default=True,
        help=f'Output image data type.  If an integer type, values are rounded and clipped to its range.  Valid for '
        f'corrected images only, parameter images always use {RasterArray.default_dtype}.',
    ),
    click.option(
        '--nodata',
        'nodata',
        type=click.STRING,
        callback=_nodata_cb,
        metavar='[NUMBER|null|nan]',
        default=RasterFuse.create_out_profile()['nodata'],
        show_default=True,
        help=f'Output image nodata value.  Valid for corrected images only, parameter images always use '
        f'{RasterArray.default_nodata}.  If null, an internal mask is written (recommended for lossy '
        f'compression).',
    ),
    click.option(
        '-co',
        '--creation-options',
        metavar='NAME=VALUE',
        multiple=True,
        default=(),
        callback=_creation_options_cb,
        help='Driver specific image creation option(s) for the output image(s).  See the `GDAL docs '
        '<https://gdal.org/en/stable/drivers/raster/index.html>`_ for details.',
    ),
    force_match_option,
)
@click.pass_context
def fuse(
    ctx: click.Context,
    src_file: tuple[pathlib.Path, ...],
    ref_file: pathlib.Path,
    model: Model,
    kernel_shape: tuple[int, int],
    src_bands: tuple[int],
    ref_bands: tuple[int],
    out_dir: pathlib.Path,
    overwrite: bool,
    cmp_file: pathlib.Path,
    cmp_bands: tuple[int],
    build_ovw: bool,
    proc_crs: ProcCrs,
    conf: pathlib.Path,
    param_image: bool,
    force_match: bool,
    **kwargs,
):
    """
    Correct image(s) to surface reflectance.

    Correct source multi-spectral aerial or satellite imagery to approximate surface reflectance, by fusion with a
    reference satellite image.

    For best results, reference and source image(s) should be concurrent, co-located and spectrally similar.  Reference
    image extents must encompass those of the source image(s).

    The reference image should contain bands that are approximate (wavelength) matches to the source image bands.
    Where source and reference images are RGB, or have ``center_wavelength`` metadata, bands are matched automatically.
    Where there are the same number of source and reference bands, and no ``center_wavelength`` metadata, bands are
    assumed to be in matching order.  The :option:`--src-band <homonim-fuse --src-band>` and
    :option:`--ref-band <homonim-fuse --ref-band>` options allow subsets and ordering of source and reference bands
    to be specified.

    Corrected image(s) are named automatically based on the source file name and option values.
    \b

    Examples:
    ---------

    Correct `source.tif` with `reference.tif` using the default options::

        homonim fuse source.tif reference.tif

    Correct `source.tif` with `reference.tif` using the `gain-blk-offset` model, a kernel of 5 x 5 pixels,
    and place the corrected images in the `corrected` directory::

        homonim fuse --model gain-blk-offset --kernel-shape 5 5 --out-dir ./corrected source.tif reference.tif

    Correct files matching `source*.tif` with `reference1.tif` using the `gain-offset` model and a kernel of 15 x 15
    pixels.  Produce parameter images, mask partially covered pixels in the corrected images, and statistically
    compare source and corrected images with `reference2.tif`::

        homonim fuse -m gain-offset -k 15 15 --param-image --mask-partial --compare reference2.tif source*.tif reference1.tif

    Correct bands 2 and 3 of `source.tif`, with bands 7 and 8 of `reference.tif`, using the default correction options::

        homonim fuse -sb 2 -sb 3 -rb 7 -rb 8 source.tif reference.tif
    """
    # build configuration dictionaries for RasterFuse
    # block_config = _update_existing_keys(RasterFuse.create_block_config(), **kwargs)
    # model_config = _update_existing_keys(RasterFuse.create_model_config(), **kwargs)
    # out_profile = _update_existing_keys(RasterFuse.create_out_profile(), **kwargs)
    # config = dict(
    #     block_config=block_config, model_config=model_config, out_profile=out_profile
    # )
    comp_files = []

    # iterate over and correct source file(s)
    try:
        for src_i, src_filename in enumerate(src_file):
            out_path = (
                pathlib.Path(out_dir) if out_dir is not None else src_filename.parent
            )
            tqdm.write(
                f'\nCorrecting {src_filename.name} ({src_i + 1} of {len(src_file)})'
            )
            with RasterFuse(
                src_filename,
                ref_file,
                proc_crs=proc_crs,
                src_bands=src_bands,
                ref_bands=ref_bands,
                force=force_match,
            ) as raster_fuse:
                # construct output filenames
                post_fix = utils.create_out_postfix(
                    raster_fuse.proc_crs,
                    model=model,
                    kernel_shape=kernel_shape,
                    driver=kwargs.get('driver', 'GTiff'),
                )
                corr_filename = out_path.joinpath(src_filename.stem + post_fix)
                param_filename = (
                    utils.create_param_filename(corr_filename) if param_image else None
                )

                start_time = timer()
                raster_fuse.process(
                    corr_filename,
                    Model(model),
                    kernel_shape,
                    param_filename=param_filename,
                    build_ovw=build_ovw,
                    overwrite=overwrite,
                    **kwargs,
                )

            tqdm.write(f'Completed in {timer() - start_time:.2f} secs')
            comp_files += [
                src_filename,
                corr_filename,
            ]  # build a list of files to pass to compare

        # compare source and corrected files with reference (invokes compare command with relevant parameters)
        if cmp_file:
            if str(cmp_file) == 'ref':
                cmp_file = ref_file
                cmp_bands = (
                    ref_bands if not cmp_bands or not len(cmp_bands) else cmp_bands
                )

            cmp_cfg = {
                k: kwargs[k]
                for k, v in RasterCompare.create_config().items()
                if k in kwargs
            }
            ctx.invoke(
                compare,
                src_file=comp_files,
                ref_file=cmp_file,
                proc_crs=proc_crs,
                src_bands=[src_bands, None] * len(src_file),
                ref_bands=cmp_bands,
                force_match=force_match,
                **cmp_cfg,
            )

    except Exception:
        logger.exception('Exception caught during processing.')  # log exception info
        raise click.Abort() from None


# compare command
@cli.command(cls=HomonimCommand)
@cloup.argument(
    'src-file',
    nargs=-1,
    metavar='IMAGE...',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help='Path(s) to image(s) to compare with :option:`REFERENCE`.',
)
@ref_file_arg
@cloup.option_group(
    'Standard options', src_bands_option, ref_bands_option, output_option
)
@cloup.option_group(
    'Advanced options',
    threads_option,
    max_block_mem_option,
    downsampling_option,
    upsampling_option,
    click.option(
        '-pc',
        '--proc-crs',
        type=click.Choice([pc.value for pc in ProcCrs], case_sensitive=False),
        default=ProcCrs.auto.value,
        show_default=True,
        help="""The image CRS in which to compare images.
    \b

    - `auto`: lowest resolution of the source and reference CRS's (recommended).
    - `src`: source image CRS.
    - `ref`: reference image CRS.
    """,
    ),
    force_match_option,
)
def compare(
    src_file: tuple[pathlib.Path, ...],
    ref_file: pathlib.Path,
    src_bands: tuple[int],
    ref_bands: tuple[int],
    output: pathlib.Path,
    proc_crs: ProcCrs,
    force_match,
    **kwargs,
):
    """
    Compare image(s) with a reference.

    Report similarity statistics between input image(s) and a reference image.  Typically, this is used to compare the
    before and after accuracy of surface reflectance correction, by comparing source and corrected images with a
    new reference image.

    Reference and input image(s) should be co-located and spectrally similar.  Reference image extents must encompass
    those of the input image(s).

    The reference image should contain bands that are approximate (wavelength) matches to the input image bands.
    Where input and reference images are RGB, or have ``center_wavelength`` metadata, bands are matched automatically.
    Where there are the same number of input and reference bands, and no ``center_wavelength`` metadata, bands are
    assumed to be in matching order.  The :option:`--src-band <homonim-compare --src-band>` and
    :option:`--ref-band <homonim-compare --ref-band>` options allow subsets and ordering of input and reference bands
    to be specified.
    \b

    Examples:
    ---------

    Compare `source.tif` and `corrected.tif` with `reference.tif`::

        homonim compare source.tif corrected.tif reference.tif
    """

    # build configuration dictionary
    config = RasterCompare.create_config(**kwargs)
    try:
        stats_dict = {}
        # if src_bands comes from compare CLI, convert to list[src_bands, ...] with one element for each source file
        src_bands_list = (
            [src_bands] * len(src_file)
            if not src_bands
            or all([isinstance(src_band, int) for src_band in src_bands])
            else src_bands
        )
        # iterate over source files, comparing with reference
        for src_i, (src_filename, src_bands) in enumerate(
            zip(src_file, src_bands_list, strict=True)
        ):
            tqdm.write(
                f'\nComparing {src_filename.name} ({src_i + 1} of {len(src_file)})'
            )
            start_time = timer()
            with RasterCompare(
                src_filename,
                ref_file,
                proc_crs=proc_crs,
                src_bands=src_bands,
                ref_bands=ref_bands,
                force=force_match,
            ) as raster_compare:
                stats_dict[str(src_filename)] = raster_compare.process(**config)
            tqdm.write(f'Completed in {timer() - start_time:.2f} secs')

        # print a key for the following tables
        tqdm.write(f'\n\n{raster_compare.schema_table()}')

        # print a results table per source image file
        summ_dict = {}
        for src_filename, im_stats_dict in stats_dict.items():
            tqdm.write(
                f'\n\n{src_filename!s}:\n\n{RasterCompare.stats_table(im_stats_dict)}'
            )
            summ_dict[pathlib.Path(src_filename).name] = im_stats_dict['Mean'].copy()

        # print a summary results table comparing all source files
        if len(summ_dict) > 1:
            tqdm.write(
                f'\n\nSummary over bands:\n\n'
                f'{RasterCompare.stats_table(summ_dict, key_header="file")}'
            )

        if output is not None:
            stats_dict['Reference'] = str(ref_file)
            with open(output, 'w') as file:
                json.dump(stats_dict, file)

    except Exception:
        logger.exception('Exception caught during processing')
        raise click.Abort() from None


@cli.command(cls=HomonimCommand)
@cloup.argument(
    'param-files',
    nargs=-1,
    metavar='PARAM...',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    callback=_param_file_cb,
    help='Path(s) to parameter image(s).',
)
@output_option
def stats(param_files: tuple[pathlib.Path, ...], output: pathlib.Path):
    """
    Report parameter statistics.

    Report the minimum, maximum, mean etc. values of a parameter image generated with the
    :option:`--param-image <homonim-fuse --param-image>` option of the ``fuse`` command.
    """

    try:
        stats_dict = {}
        meta_dict = {}

        # process parameter file(s), storing results
        for param_i, param_filename in enumerate(param_files):
            tqdm.write(
                f'\nProcessing {param_filename.name} ({param_i + 1} of {len(param_files)})'
            )
            with ParamStats(param_filename) as param_stats:
                stats_dict[str(param_filename)] = param_stats.stats()
                meta_dict[str(param_filename)] = param_stats.metadata

        # print a key for the following tables
        tqdm.write(f'\n\n{param_stats.schema_table}')

        # iterate over stored result(s) and print
        for param_filename in stats_dict.values():
            tqdm.write(f'\n{pathlib.Path(param_filename).name}:\n')
            tqdm.write(meta_dict[param_filename])
            tqdm.write(
                f'Stats:\n\n{ParamStats.stats_table(stats_dict[param_filename])}\n'
            )

        if output is not None:
            with open(output, 'w') as file:
                json.dump(stats_dict, file, allow_nan=True)

    except Exception:
        logger.exception('Exception caught during processing')
        raise click.Abort() from None


if __name__ == '__main__':
    cli()
