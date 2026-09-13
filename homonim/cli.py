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
    HomonimWarning,
    Model,
    ParamStats,
    ProcCrs,
    RasterCompare,
    RasterFuse,
)
from homonim.errors import HomonimError
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
src_file_arg = click.argument(
    'src_files',
    nargs=-1,
    metavar='SOURCE...',
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
)
threads_option = click.option(
    '-t',
    '--threads',
    type=click.INT,
    default=RasterFuse._default_config['threads'],
    show_default=True,
    help='Number of image blocks to process concurrently.  If ``0``, the number of '
    'CPUs is used.',
)
max_block_mem_option = click.option(
    '-mbm',
    '--max-block-mem',
    type=click.FLOAT,
    default=RasterFuse._default_config['max_block_mem'],
    show_default=True,
    help='Maximum image block size in megabytes.  If ``0``, a block will correspond '
    'to an image band.',
)
downsampling_option = click.option(
    '-ds',
    '--downsampling',
    type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
    default=KernelModel._default_config['downsampling'].name,
    show_default=True,
    help='Resampling method to use when downsampling.',
)
upsampling_option = click.option(
    '-us',
    '--upsampling',
    type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
    default=KernelModel._default_config['upsampling'].name,
    show_default=True,
    help='Resampling method to use when upsampling.',
)
output_option = click.option(
    '-op',
    '--output',
    type=click.Path(exists=False, dir_okay=False, writable=True, path_type=Path),
    help='Path of a JSON file to write results to.',
)
ref_bands_option = click.option(
    '-rb',
    '--ref-band',
    'ref_bands',
    type=click.INT,
    multiple=True,
    show_default='all center_wavelength tagged or non-alpha bands.',
    help='Indexes of reference bands to match with source bands (1 based).',
)
force_match_option = click.option(
    '-f',
    '--force-match',
    is_flag=True,
    default=False,
    show_default=True,
    help='Bypass wavelength band matching.',
)


@click.group()
@click.option('--verbose', '-v', count=True, help='Increase verbosity.')
@click.option('--quiet', '-q', count=True, help='Decrease verbosity.')
@click.version_option(version=__version__, message='%(version)s')
@click.pass_context
def cli(ctx: click.Context, verbose: int, quiet: int):
    """Surface reflectance correction toolkit."""
    ctx.with_resource(_configure_logging(verbose - quiet))


@cli.command(
    cls=HomonimCommand,
    short_help='Correct images to surface reflectance.',
    epilog='See https://homonim.readthedocs.io/ for more detail on usage.',
)
@src_file_arg
@ref_file_arg
@click.option(
    '-m',
    '--model',
    type=click.Choice(Model, case_sensitive=False),
    default=KernelModel._default_config['model'],
    show_default=True,
    help='Correction model type.',
)
@click.option(
    '-k',
    '--kernel-shape',
    type=click.Tuple([click.INT, click.INT]),
    nargs=2,
    default=KernelModel._default_config['kernel_shape'],
    show_default=True,
    metavar='HEIGHT WIDTH',
    help='Kernel height and width in pixels of the :option:`--proc-crs` image.',
)
@click.option(
    '-sb',
    '--src-band',
    'src_bands',
    type=click.INT,
    multiple=True,
    show_default='all center_wavelength tagged or non-alpha bands.',
    help='Indexes of source bands to be corrected (1 based).',
)
@ref_bands_option
@click.option(
    '-od',
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, writable=True),
    # TODO: change to cwd
    show_default='source image directory.',
    help='Path of the output image directory.',
)
@click.option(
    '-o',
    '--overwrite',
    is_flag=True,
    default=False,
    show_default=True,
    help='Overwrite existing output images.',
)
@click.option(
    '-cmp',
    '--compare',
    'cmp_file',
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Path of an image to compare source and corrected images with. If ``'ref'``,"
    ' source and corrected images are compared with the REFERENCE.',
)
@click.option(
    '-cb',
    '--cmp-band',
    'cmp_bands',
    type=click.INT,
    multiple=True,
    show_default='all center_wavelength tagged or non-alpha bands.',
    help='Indexes of :option:`--compare` bands to match with source / corrected bands '
    "(1 based).  Ignored if :option:`--compare` is ``'ref'``.",
)
@click.option(
    '-bo/-nbo',
    '--build-ovw/--no-build-ovw',
    type=click.BOOL,
    default=True,
    show_default=True,
    help='Build overviews for the output images.',
)
@click.option(
    '-c',
    '--conf',
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    callback=_conf_cb,
    expose_value=False,
    is_eager=True,
    help='Path of a YAML configuration file specifying option values.',
)
@click.option(
    '-pi/-npi',
    '--param-image/--no-param-image',
    type=click.BOOL,
    default=False,
    show_default=True,
    help='Write model parameters and R\N{SUPERSCRIPT TWO} values to parameter images.',
)
@click.option(
    '-mp/-nmp',
    '--mask-partial/--no-mask-partial',
    type=click.BOOL,
    default=KernelModel._default_config['mask_partial'],
    show_default=True,
    help='Mask corrected pixels not produced by full kernel or source / reference '
    'image coverage.  Can help reduce seam-lines.',
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
    # TODO: add callback to convert 0->None, and test for this too
    help='R\N{SUPERSCRIPT TWO} threshold below which to inpaint model offsets from '
    'surrounding values. Valid for the ``gain-offset`` :option:`--model` only.  '
    'If ``0``, no inpainting is done.',
)
@click.option(
    '-pc',
    '--proc-crs',
    type=click.Choice(ProcCrs, case_sensitive=False),
    default=ProcCrs.auto,
    show_default=True,
    help='Which of the source or reference CRS and pixel grids should be used for '
    'estimating correction parameters (``auto`` is recommended).',
)
@click.option(
    '--driver',
    type=click.Choice(Driver, case_sensitive=False),
    default=RasterFuse._default_config['driver'],
    show_default=True,
    help='Corrected image driver.',
)
@click.option(
    '--dtype',
    type=click.Choice(list(dtype_fwd.values())[1:8], case_sensitive=False),
    default=RasterArray.default_dtype,
    show_default=True,
    help='Corrected image data type.',
)
@click.option(
    '--nodata',
    'nodata',
    type=click.STRING,
    callback=_nodata_cb,
    metavar='[NUMBER|null|nan]',
    default=RasterArray.default_nodata,
    show_default=True,
    help='Corrected image nodata value.  If ``null``, an internal mask is written '
    '(recommended for lossy compression).',
)
@click.option(
    '-co',
    '--creation-options',
    metavar='NAME=VALUE',
    type=click.STRING,
    multiple=True,
    callback=_creation_options_cb,
    default=(),
    show_default='auto',
    help='Corrected image :option:`--driver` specific creation options.  If '
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
    src_bands: tuple[int, ...],
    ref_bands: tuple[int, ...],
    out_dir: Path,
    cmp_file: Path,
    cmp_bands: tuple[int, ...],
    proc_crs: ProcCrs,
    param_image: bool,
    force_match: bool,
    **kwargs,
):
    """
    Correct SOURCE images to surface reflectance by fusion with a REFERENCE.

    For best results, source and reference images should be concurrent. Reference
    extents must encompass those of the source.

    The reference should contain bands that are approximate wavelength matches to the
    source bands.  When source and reference bands are RGB, or have
    ``center_wavelength`` tags, bands are matched automatically based on wavelength.
    Otherwise, source and reference bands are assumed to be in matching order.
    Subsets and ordering of bands can be specified with the :option:`--src-band
    <homonim-fuse --src-band>` and :option:`--ref-band <homonim-fuse --ref-band>`
    parameters.

    Option values can be provided via a YAML configuration file with :option:`--conf
    <homonim-fuse --conf>`.  When options are provided on the command line and in the
    configuration file, command line values take precedence.

    Output images are written to the current directory by default.  This can be
    changed with :option:`--out-dir <homonim-fuse --out-dir>`.  Images are named
    based on the source file name and option values.
    """
    cmp_src_files = []

    # iterate over and correct source files
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
                    **kwargs,
                )
        except (RasterioIOError, FileExistsError, HomonimError) as ex:
            raise click.UsageError(str(ex)) from None

        tqdm.write(f'Completed in {timer() - start_time:.2f} secs')
        # build a list of files to pass to compare
        cmp_src_files += [src_file, corr_file]

    # compare source and corrected files with reference (invokes compare command with
    # relevant parameters)
    if cmp_file:
        if str(cmp_file).lower() == 'ref':
            cmp_file = ref_file
            if cmp_bands:
                warnings.warn(
                    "Ignoring --cmp-band as --compare is 'ref'.",
                    category=HomonimWarning,
                    stacklevel=2,
                )
            cmp_bands = ref_bands

        cmp_cfg = {
            k: kwargs[k] for k in RasterCompare._default_config.keys() if k in kwargs
        }
        ctx.invoke(
            compare,
            src_files=cmp_src_files,
            ref_file=cmp_file,
            proc_crs=proc_crs,
            src_bands_list=[src_bands, None] * len(src_files),
            ref_bands=cmp_bands,
            force_match=force_match,
            **cmp_cfg,
        )


@cli.command(
    cls=HomonimCommand,
    short_help='Compare images with a reference.',
    epilog='See https://homonim.readthedocs.io/ for more detail on usage.',
)
@src_file_arg
@ref_file_arg
@click.option(
    '-sb',
    '--src-band',
    'src_bands',
    type=click.INT,
    multiple=True,
    show_default='all center_wavelength tagged or non-alpha bands.',
    help='Indexes of source bands to be compared (1 based).',
)
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
    help='Which of the source or reference CRS and pixel grids should be used for '
    'comparing (``auto`` recommended).',
)
@force_match_option
def compare(
    src_files: tuple[Path, ...],
    ref_file: Path,
    src_bands: tuple[int, ...],
    ref_bands: tuple[int, ...],
    output: Path,
    proc_crs: ProcCrs,
    force_match,
    # non command line option that allows compare() to be invoked with a list of per
    # source file bands
    src_bands_list: list[tuple[int, ...]] | None = None,
    **kwargs,
):
    """
    Report similarity statistics between SOURCE images and a REFERENCE.

    Typically, this is used to compare the before and after accuracy of surface
    reflectance correction, by comparing uncorrected and corrected images with a new
    reference.

    Reference extents must encompass those of the source.

    The reference should contain bands that are approximate wavelength matches to the
    source bands.  When source and reference bands are RGB, or have
    ``center_wavelength`` tags, bands are matched automatically based on wavelength.
    Otherwise, source and reference bands are assumed to be in matching order.
    Subsets and ordering of bands can be specified with the :option:`--src-band
    <homonim-compare --src-band>` and :option:`--ref-band <homonim-compare
    --ref-band>` parameters.
    """
    stats_dict = {}
    # construct src_bands_list if compare() was invoked from the command line
    if src_bands_list is None:
        src_bands_list = [src_bands] * len(src_files)

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


@cli.command(
    cls=HomonimCommand,
    short_help='Report parameter statistics.',
    epilog='See https://homonim.readthedocs.io/ for more detail on usage.',
)
@click.argument(
    'param_files',
    nargs=-1,
    metavar='PARAMETER...',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@output_option
def stats(param_files: tuple[Path, ...], output: Path):
    """
    Report statistics of PARAMETER images.

    Report the minimum, maximum, mean etc. values of parameter images generated with
    the :option:`--param-image <homonim-fuse --param-image>` option of the ``fuse``
    command.
    """
    stats_dict = {}
    meta_dict = {}

    # process parameter files, storing results
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
    tqdm.write(f'\n\n{param_stats.schema_table()}')

    # iterate over stored results and print
    for param_filename in stats_dict.keys():
        tqdm.write(f'\n{Path(param_filename).name}:\n')
        tqdm.write(meta_dict[param_filename])
        tqdm.write(f'Stats:\n\n{ParamStats.stats_table(stats_dict[param_filename])}\n')

    if output is not None:
        with open(output, 'w') as file:
            json.dump(stats_dict, file, allow_nan=True)


if __name__ == '__main__':
    cli()
