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
import re
import warnings
from contextlib import contextmanager
from pathlib import Path
from timeit import default_timer as timer

import click
import rasterio as rio
import yaml
from rasterio.dtypes import dtype_fwd
from rasterio.errors import RasterioIOError
from tqdm.auto import tqdm
from tqdm.contrib.logging import _TqdmLoggingHandler
from yaml import YAMLError

from homonim import (
    Driver,
    Model,
    ParamStats,
    ProcCrs,
    RasterCompare,
    RasterFuse,
    utils,
)
from homonim.errors import HomonimError, ImageFormatError
from homonim.kernel_model import KernelModel
from homonim.raster_array import RasterArray
from homonim.version import __version__

logger = logging.getLogger(__name__)


class HomonimCommand(click.Command):
    """click.Command subclass for formatting help with RST markup."""

    def get_help(self, ctx: click.Context):
        """Strip some RST markup from the help text for CLI display.  Will not work with grid tables."""

        # Note that this can't easily be done in __init__, as each sub-command's __init__ gets called,
        # which ends up re-assigning self.wrap_text to reformat_text
        if not hasattr(self, 'wrap_text'):
            self.wrap_text = click.formatting.wrap_text
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

        click.formatting.wrap_text = reformat_text
        return click.Command.get_help(self, ctx)


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


def _conf_cb(ctx: click.Context, param: click.Option, value):
    """Click callback set default option values from a YAML configuration file."""
    # adapted from: https://jwodder.github.io/kbits/posts/click-config/
    if value is None:
        return
    with open(value) as f:
        try:
            conf_dict = yaml.safe_load(f)
        except YAMLError as ex:
            raise click.BadParameter(str(ex)) from None

    # transform creation_options to a list of NAME=VALUE strings for parsing in
    # _creation_options_cb()
    creation_options = conf_dict.get('creation_options', {})
    if not isinstance(creation_options, dict):
        raise click.BadParameter(
            "'creation_options' should be a dictionary of NAME: VALUE pairs."
        ) from None
    conf_dict['creation_options'] = [f'{k}={v}' for k, v in creation_options.items()]

    ctx.default_map = conf_dict


def _nodata_cb(ctx: click.Context, param: click.Option, value: str):
    """click callback to convert --nodata value to None, nan or float."""
    # adapted from rasterio https://github.com/rasterio/rasterio
    if value is None or value.lower() in ['null', 'nil', 'none']:
        return None
    else:
        try:
            value = float(value.lower())
        except (TypeError, ValueError):
            raise click.BadParameter(
                f'{value} is not a number', param=param, param_hint='--nodata'
            ) from None
        return value


def _creation_options_cb(ctx: click.Context, param: click.Option, value):
    """click callback to validate and parse multiple creation options (e.g. -co
    KEY1=VAL1 -co KEY2=VAL2).
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
                    None if v.lower() in ['none', 'null', 'nil'] else yaml.safe_load(v)
                )
        return out


# define click options and arguments common to more than one command
# TODO: allow URIs for all image options/args?
# TODO: test for path/URI existence here or leave it to called code?
ref_file_arg = click.argument(
    'ref_file',
    nargs=1,
    metavar='REFERENCE',
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
)
threads_option = click.option(
    '-t',
    '--threads',
    type=click.INT,
    default=RasterFuse._default_config['threads'],
    show_default=True,
    help='Number of image blocks to process concurrently (0 = use all processors).',
)
max_block_mem_option = click.option(
    '-mbm',
    '--max-block-mem',
    type=click.FLOAT,
    default=RasterFuse._default_config['max_block_mem'],
    show_default=True,
    help='Maximum image block size in megabytes (0 = block corresponds to a whole band).',
)
downsampling_option = click.option(
    '-ds',
    '--downsampling',
    type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
    default=KernelModel._default_config['downsampling'].name,
    show_default=True,
    help='Resampling method for re-projecting from high to low resolution.  See the `rasterio docs '
    '<https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_ for '
    'details.',
)
upsampling_option = click.option(
    '-us',
    '--upsampling',
    type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
    default=KernelModel._default_config['downsampling'].name,
    show_default=True,
    help='Resampling method for re-projecting from low to high resolution.  See the `rasterio docs '
    '<https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_ for '
    'details.',
)
output_option = click.option(
    '-op',
    '--output',
    type=click.Path(exists=False, dir_okay=False, writable=True, path_type=Path),
    help='Write results to this json file.',
)
src_bands_option = click.option(
    '-sb',
    '--src-band',
    'src_bands',
    type=click.INT,
    multiple=True,
    show_default='all spectral or non-alpha bands.',
    help='Source band index(es) to process (1 based).',
)
ref_bands_option = click.option(
    '-rb',
    '--ref-band',
    'ref_bands',
    type=click.INT,
    multiple=True,
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


# define the click CLI
@click.group()
@click.option('--verbose', '-v', count=True, help='Increase verbosity.')
@click.option('--quiet', '-q', count=True, help='Decrease verbosity.')
@click.version_option(version=__version__, message='%(version)s')
@click.pass_context
def cli(ctx: click.Context, verbose: int, quiet: int):
    """Surface reflectance correction toolkit."""
    ctx.with_resource(_configure_logging(verbose - quiet))


# fuse command
@cli.command(cls=HomonimCommand)
# standard options
@click.argument(
    'src_files',
    nargs=-1,
    metavar='SOURCE...',
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
)
@ref_file_arg
@click.option(
    '-m',
    '--model',
    type=click.Choice(Model, case_sensitive=False),
    default=KernelModel._default_config['model'],
    show_default=True,
    help="""Correction model.

    - `gain`: Gain-only model, suitable for haze-free and zero offset images.

    - `gain-blk-offset`: Gain-only model applied to offset normalised blocks.  Suitable for most source-reference combinations.

    - `gain-offset`: Gain and offset model.  Most accurate model, but sensitive to differences between source and reference.
    """,
)
@click.option(
    '-k',
    '--kernel-shape',
    type=click.Tuple([click.INT, click.INT]),
    nargs=2,
    default=KernelModel._default_config['kernel_shape'],
    show_default=True,
    metavar='HEIGHT WIDTH',
    help='Kernel height and width in pixels of the :option:`--proc-crs <homonim-fuse --proc-crs>` image. Larger '
    'kernels are less susceptible to over-fitting, but provide lower resolution correction.',
)
@src_bands_option
@ref_bands_option
@click.option(
    '-od',
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, writable=True),
    show_default='source image directory.',
    help='Path of the output image directory.',
)
@click.option(
    '-o',
    '--overwrite',
    is_flag=True,
    default=False,
    show_default=True,
    help='Overwrite existing output images(s).',
)
# TODO: does this work in front of an argument?  or should it be handles like oty rpc's --gcp-refine?
@click.option(
    '-cmp',
    '--compare',
    'cmp_file',
    metavar='FILE',
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    is_flag=False,
    flag_value='ref',
    help='Compare source and corrected images with this reference image.  If no ``FILE`` value is given, source '
    'and corrected images are compared with :option:`REFERENCE`.',
)
@click.option(
    '-cb',
    '--cmp-band',
    'cmp_bands',
    type=click.INT,
    multiple=True,
    show_default='all spectral or non-alpha bands.',
    help='Comparison reference band index(es) that correspond (spectrally) to '
    ':option:`--src-band <homonim-fuse --src-band>` (s).',
)
@click.option(
    '-bo/-nbo',
    '--build-ovw/--no-build-ovw',
    type=click.BOOL,
    default=True,
    show_default=True,
    help='Build overviews for the output image(s).',
)
@click.option(
    '-c',
    '--conf',
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    callback=_conf_cb,
    expose_value=False,
    is_eager=True,
    help='Path to a YAML option configuration file.',
)
@click.option(
    '-pi/-npi',
    '--param-image/--no-param-image',
    type=click.BOOL,
    default=False,
    show_default=True,
    help='Write the  model parameters and R\N{SUPERSCRIPT TWO} values for each corrected image to a parameter '
    'image file.',
)
@click.option(
    '-mp/-nmp',
    '--mask-partial/--no-mask-partial',
    type=click.BOOL,
    default=KernelModel._default_config['mask_partial'],
    show_default=True,
    help='Mask output pixels produced from partial kernel or source / reference image coverage.',
)
@threads_option
@max_block_mem_option
@downsampling_option
@upsampling_option
@click.option(
    '-rit',
    '--r2-inpaint-thresh',
    type=click.FloatRange(min=0, max=1),
    default=KernelModel._default_config['r2_inpaint_thresh'],
    show_default=True,
    metavar='FLOAT 0-1',
    help='R\N{SUPERSCRIPT TWO} threshold below which to inpaint model parameters from surrounding areas '
    '(0 = turn off inpainting). Valid for `gain-offset` :option:`--model` only.',
)
@click.option(
    '-pc',
    '--proc-crs',
    type=click.Choice(ProcCrs, case_sensitive=False),
    default=ProcCrs.auto,
    show_default=True,
    help="""The image CRS in which to estimate correction parameters.
    \b

    - `auto`: lowest resolution of the source and reference CRS's (recommended).
    - `src`: source image CRS.
    - `ref`: reference image CRS.
    """,
)
@click.option(
    '--driver',
    type=click.Choice(Driver, case_sensitive=False),
    default=RasterFuse._default_config['driver'],
    show_default=True,
    help='Corrected image format driver.',
)
@click.option(
    '--dtype',
    type=click.Choice(list(dtype_fwd.values())[1:8], case_sensitive=False),
    default=RasterArray.default_dtype,
    show_default=True,
    help=f'Output image data type.  If an integer type, values are rounded and clipped to its range.  Valid for '
    f'corrected images only, parameter images always use {RasterArray.default_dtype}.',
)
@click.option(
    '--nodata',
    'nodata',
    type=click.STRING,
    callback=_nodata_cb,
    metavar='[NUMBER|null|nan]',
    default=RasterArray.default_nodata,
    show_default=True,
    help=f'Output image nodata value.  Valid for corrected images only, parameter images always use '
    f'{RasterArray.default_nodata}.  If null, an internal mask is written (recommended for lossy '
    f'compression).',
)
@click.option(
    '-co',
    '--creation-options',
    metavar='NAME=VALUE',
    multiple=True,
    default=(),
    show_default='auto',
    callback=_creation_options_cb,
    help='Corrected image :option:`--driver` specific creation option(s).  If '
    'supplied, no defaults are set, and these are the only options used.  See the '
    'GDAL `GTiff <https://gdal.org/en/latest/drivers/raster/gtiff.html#creation'
    '-options>`__ and `COG <https://gdal.org/en/latest/drivers/raster/cog.html'
    '#creation-options>`__ docs for available options.',
)
@force_match_option
@click.pass_context
def fuse(
    ctx: click.Context,
    src_files: tuple[Path, ...],
    ref_file: Path,
    model: Model,
    kernel_shape: tuple[int, int],
    src_bands: tuple[int],
    ref_bands: tuple[int],
    out_dir: Path,
    overwrite: bool,
    cmp_file: Path,
    cmp_bands: tuple[int],
    build_ovw: bool,
    proc_crs: ProcCrs,
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
    cmp_src_files = []

    # iterate over and correct source file(s)
    for src_i, src_file in enumerate(src_files):
        tqdm.write(f'\nCorrecting {src_file.name} ({src_i + 1} of {len(src_files)})')
        out_path = Path(out_dir) if out_dir else src_file.parent
        try:
            with RasterFuse(
                src_file,
                ref_file,
                proc_crs=proc_crs,
                src_bands=src_bands,
                ref_bands=ref_bands,
                force=force_match,
            ) as fuse:
                # construct output filenames
                postfix = (
                    f'FUSE_c{fuse.proc_crs.upper()}_m{model.upper()}_'
                    f'k{kernel_shape[0]}_{kernel_shape[1]}'
                )
                corr_file = out_path.joinpath(f'{src_file.stem}_{postfix}.tif')
                param_file = (
                    out_path.joinpath(f'{corr_file.stem}_PARAM.tif')
                    if param_image
                    else None
                )

                start_time = timer()
                fuse.process(
                    corr_file,
                    Model(model),
                    kernel_shape,
                    param_filename=param_file,
                    build_ovw=build_ovw,
                    overwrite=overwrite,
                    **kwargs,
                )
        except (RasterioIOError, FileExistsError, HomonimError) as ex:
            raise click.UsageError(str(ex)) from None

        tqdm.write(f'Completed in {timer() - start_time:.2f} secs')
        # build a list of files to pass to compare
        cmp_src_files += [src_file, corr_file]

    # compare source and corrected files with reference (invokes compare command with relevant parameters)
    if cmp_file:
        if str(cmp_file) == 'ref':
            cmp_file = ref_file
            cmp_bands = ref_bands if not cmp_bands or not len(cmp_bands) else cmp_bands

        cmp_cfg = {
            k: kwargs[k] for k in RasterCompare._default_config.keys() if k in kwargs
        }
        ctx.invoke(
            compare,
            src_files=cmp_src_files,
            ref_file=cmp_file,
            proc_crs=proc_crs,
            src_bands=[src_bands, None] * len(src_files),
            ref_bands=cmp_bands,
            force_match=force_match,
            **cmp_cfg,
        )


# compare command
@cli.command(cls=HomonimCommand)
@click.argument(
    'src_files',
    nargs=-1,
    metavar='IMAGE...',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@ref_file_arg
@src_bands_option
@ref_bands_option
@output_option
@threads_option
@max_block_mem_option
@downsampling_option
@upsampling_option
@click.option(
    '-pc',
    '--proc-crs',
    type=click.Choice(ProcCrs, case_sensitive=False),
    default=ProcCrs.auto,
    show_default=True,
    help="""The image CRS in which to compare images.
\b

- `auto`: lowest resolution of the source and reference CRS's (recommended).
- `src`: source image CRS.
- `ref`: reference image CRS.
""",
)
@force_match_option
def compare(
    src_files: tuple[Path, ...],
    ref_file: Path,
    src_bands: tuple[int],
    ref_bands: tuple[int],
    output: Path,
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
    stats_dict = {}
    # if src_bands comes from compare CLI, convert to list[src_bands, ...] with one element for each source file
    src_bands_list = (
        [src_bands] * len(src_files)
        if not src_bands or all([isinstance(src_band, int) for src_band in src_bands])
        else src_bands
    )
    # iterate over source files, comparing with reference
    for src_i, (src_file, src_bands) in enumerate(
        zip(src_files, src_bands_list, strict=True)
    ):
        tqdm.write(f'\nComparing {src_file.name} ({src_i + 1} of {len(src_files)})')
        start_time = timer()
        try:
            with RasterCompare(
                src_file,
                ref_file,
                proc_crs=proc_crs,
                src_bands=src_bands,
                ref_bands=ref_bands,
                force=force_match,
            ) as raster_compare:
                stats_dict[str(src_file)] = raster_compare.process(**kwargs)
            tqdm.write(f'Completed in {timer() - start_time:.2f} secs')
        except (RasterioIOError, HomonimError) as ex:
            raise click.UsageError(str(ex)) from None

    # print a key for the following tables
    tqdm.write(f'\n\n{raster_compare.schema_table()}')

    # print a results table per source image file
    summ_dict = {}
    for src_file, im_stats_dict in stats_dict.items():
        tqdm.write(f'\n\n{src_file!s}:\n\n{RasterCompare.stats_table(im_stats_dict)}')
        summ_dict[Path(src_file).name] = im_stats_dict['Mean'].copy()

    # print a summary results table comparing all source files
    if len(summ_dict) > 1:
        tqdm.write(
            f'\n\nSummary over bands:\n\n'
            f'{RasterCompare.stats_table(summ_dict, key_heading="file")}'
        )

    if output is not None:
        stats_dict['Reference'] = str(ref_file)
        with open(output, 'w') as file:
            json.dump(stats_dict, file)


@cli.command(cls=HomonimCommand)
@click.argument(
    'param_files',
    nargs=-1,
    metavar='PARAM...',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@output_option
def stats(param_files: tuple[Path, ...], output: Path):
    """
    Report parameter statistics.

    Report the minimum, maximum, mean etc. values of a parameter image generated with the
    :option:`--param-image <homonim-fuse --param-image>` option of the ``fuse`` command.
    """
    stats_dict = {}
    meta_dict = {}

    # process parameter file(s), storing results
    for param_i, param_filename in enumerate(param_files):
        tqdm.write(
            f'\nProcessing {param_filename.name} ({param_i + 1} of {len(param_files)})'
        )
        try:
            with ParamStats(param_filename) as param_stats:
                stats_dict[str(param_filename)] = param_stats.stats()
                meta_dict[str(param_filename)] = param_stats.metadata
        except (RasterioIOError, HomonimError) as ex:
            raise click.UsageError(str(ex)) from None

    # print a key for the following tables
    tqdm.write(f'\n\n{param_stats.schema_table}')

    # iterate over stored result(s) and print
    for param_filename in stats_dict.keys():
        tqdm.write(f'\n{Path(param_filename).name}:\n')
        tqdm.write(meta_dict[param_filename])
        tqdm.write(f'Stats:\n\n{ParamStats.stats_table(stats_dict[param_filename])}\n')

    if output is not None:
        with open(output, 'w') as file:
            json.dump(stats_dict, file, allow_nan=True)


if __name__ == '__main__':
    cli()
