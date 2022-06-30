"""
    Homonim: Correction of aerial and satellite imagery to surface relfectance
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

import inspect
import json
import logging
import math
import pathlib
import re
import sys
from timeit import default_timer as timer
from typing import Union, Tuple, Dict

import click
import cloup
import pandas as pd
import rasterio as rio
import yaml
from click.core import ParameterSource
from homonim import utils, version
from homonim.compare import RasterCompare
from homonim.enums import ProcCrs, Method
from homonim.errors import ImageFormatError
from homonim.fuse import RasterFuse
from homonim.kernel_model import KernelModel
from homonim.stats import ParamStats
from homonim.raster_array import RasterArray
from rasterio.warp import SUPPORTED_RESAMPLING

logger = logging.getLogger(__name__)


class PlainInfoFormatter(logging.Formatter):
    """ logging formatter to format INFO logs without the module name etc prefix. """

    def format(self, record: logging.LogRecord):
        if record.levelno == logging.INFO:
            self._style._fmt = '%(message)s'
        else:
            self._style._fmt = '%(levelname)s:%(name)s: %(message)s'
        return super().format(record)


class HomonimCommand(cloup.Command):
    """ cloup.Command sub-class for formatting help with RST markup. """

    def get_help(self, ctx: click.Context):
        """ Strip some RST markup from the help text for CLI display.  Will not work with grid tables. """
        if not hasattr(self, 'wrap_text'):
            self.wrap_text = cloup.formatting._formatter.wrap_text
        sub_strings = {
            '\b\n': '\n\b',  # convert from RST friendly to click literal (unwrapped) block marker
            '\| ': '',  # strip RST literal (unwrapped) marker in e.g. tables and bullet lists
            '\n\.\. _.*:\n': '',  # strip RST ref directive '\n.. <name>:\n'
            '`(.*?) <(.*?)>`_': '\g<1>',  # convert from RST cross-ref '`<name> <<link>>`_' to 'name'
            '::': ':',  # convert from RST '::' to ':'
            '``(.*?)``': '\g<1>',  # convert from RST '``literal``' to 'literal'
            ':option:`(.*?)( <.*?>)?`': '\g<1>',  # convert ':option:`--name <group-command --name>`' to '--name'
            ':option:`(.*?)`': '\g<1>',  # convert ':option:`--name`' to '--name'
        } # yapf: disable

        def reformat_text(text: str, width: int, **kwargs):
            for sub_key, sub_value in sub_strings.items():
                text = re.sub(sub_key, sub_value, text, flags=re.DOTALL)
            return self.wrap_text(text, width, **kwargs)

        cloup.formatting._formatter.wrap_text = reformat_text
        return cloup.Command.get_help(self, ctx)


class FuseCommand(HomonimCommand):
    """ click.Command sub-class for setting fuse command parameters from a yaml config file. """

    def invoke(self, ctx: click.Context):
        """ Merge config file with command line and default parameter values.  """
        # adapted from https://stackoverflow.com/questions/46358797/python-click-supply-arguments-and-options-from-a
        # -configuration-file/46391887
        config_file = ctx.params['conf']
        if config_file is not None:
            # read the config file into a dict
            with open(config_file) as f:
                config_dict = yaml.safe_load(f)

            for conf_key, conf_value in config_dict.items():
                if conf_key not in ctx.params:
                    raise click.BadParameter(f'Unknown config file parameter "{conf_key}"', ctx=ctx, param_hint='conf')
                else:
                    param_src = ctx.get_parameter_source(conf_key)
                    # overwrite parameters not specified on command line with config file values
                    if ctx.params[conf_key] is None or param_src == ParameterSource.DEFAULT:
                        ctx.params[conf_key] = conf_value
                        ctx.set_parameter_source(conf_key, ParameterSource.COMMANDLINE)

        # Set the default creation_options if no other driver or creation_options have been specified.
        # This can't be done in a callback as it depends on --driver.
        if (ctx.get_parameter_source('driver') == ParameterSource.DEFAULT and
                ctx.get_parameter_source('creation_options') == ParameterSource.DEFAULT):
            ctx.params['creation_options'] = RasterFuse.default_out_profile['creation_options']

        return click.Command.invoke(self, ctx)


def _update_existing_keys(default_dict: Dict, **kwargs):
    """ Update values in a dict with args from matching keys in **kwargs. """
    return {k: kwargs.get(k, v) for k, v in default_dict.items()}


def _configure_logging(verbosity: int):
    """ Configure python logging level."""
    # adapted from rasterio https://github.com/rasterio/rasterio
    log_level = max(10, 20 - 10 * verbosity)

    # apply config to package logger, rather than root logger
    pkg_logger = logging.getLogger(__package__)
    formatter = PlainInfoFormatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(log_level)
    logging.captureWarnings(True)


def _threads_cb(ctx: click.Context, param: click.Option, value):
    """ click callback to validate --threads. """
    try:
        threads = utils.validate_threads(value)
    except Exception as ex:
        raise click.BadParameter(str(ex))
    return threads


def _nodata_cb(ctx: click.Context, param: click.Option, value):
    """ click callback to convert --nodata value to None, nan or float. """
    # adapted from rasterio https://github.com/rasterio/rasterio
    if value is None or value.lower() in ['null', 'nil', 'none', 'nada']:
        return None
    else:
        # check value is a number and can be cast to output dtype
        try:
            value = float(value.lower())
            if not rio.dtypes.can_cast_dtype(value, ctx.params['dtype']):
                raise click.BadParameter(
                    f'{value} cannot be cast to the output image data type {ctx.params["dtype"]}', param=param,
                    param_hint='nodata'
                )
        except (TypeError, ValueError):
            raise click.BadParameter(f'{value} is not a number', param=param, param_hint='nodata')

        return value


def _compare_cb(ctx: click.Context, param: click.Option, value):
    """ click callback to check --compare path exists if specified.  """
    if value and str(value) != 'ref':
        if not pathlib.Path(value).exists():
            raise click.BadParameter(f'Comparison image does not exist: {value}')
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
                raise click.BadParameter('Invalid syntax for KEY=VAL arg: {}'.format(pair))
            else:
                k, v = pair.split('=', 1)
                k = k.lower()
                v = v.lower()
                out[k] = None if v.lower() in ['none', 'null', 'nil', 'nada'] else yaml.safe_load(v)
        return out


def _param_file_cb(ctx: click.Context, param: click.Option, value):
    """ click callback to validate parameter image file(s). """
    for filename in value:
        filename = pathlib.Path(filename)
        try:
            utils.validate_param_image(filename)
        except (FileNotFoundError, ImageFormatError):
            raise click.BadParameter(f'{filename.name} is not a valid parameter image.', param=param)
    return value


# define click options and arguments common to more than one command
# use cloup's argument to auto print argument help on command line
ref_file_arg = cloup.argument(
    'ref-file', nargs=1, metavar='REFERENCE', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help='Path to a reference image.'
)
threads_option = click.option(
    '-t', '--threads', type=click.INT, default=RasterFuse.default_homo_config['threads'], show_default=True,
    callback=_threads_cb, help=f'Number of image blocks to process concurrently (0 = use all cpus).'
)
output_option = click.option(
    '-op', '--output', type=click.Path(exists=False, dir_okay=False, writable=True, path_type=pathlib.Path),
    help='Write results to this json file.'
)

""" cloup context settings to print help in 'linear' layout with heading/option emphasis. """
context_settings = cloup.Context.settings(
    formatter_settings=cloup.HelpFormatter.settings(
        col2_min_width=math.inf, theme=cloup.HelpTheme(
            invoked_command=cloup.Style(fg='bright_white', bold=True),
            heading=cloup.Style(fg='bright_white', bold=True),
            col1=cloup.Style(fg='bright_white'),
        )
    )
) # yapf: disable

# define the click CLI
@cloup.group(context_settings=context_settings)
@click.option('--verbose', '-v', count=True, help='Increase verbosity.')
@click.option('--quiet', '-q', count=True, help='Decrease verbosity.')
@click.version_option(version=version.__version__, message='%(version)s')
def cli(verbose, quiet):
    """ Surface reflectance correction and support utilities. """
    verbosity = verbose - quiet
    _configure_logging(verbosity)


# fuse command
@cloup.command(cls=FuseCommand)
# standard options
@cloup.argument(
    'src-file', nargs=-1, metavar='SOURCE...', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help='Path(s) to source image(s) to be corrected.'
)
@ref_file_arg
@cloup.option_group(
    "Standard options",
    # note: either use click.option(...), or cloup.option(..., help=inspect.cleandoc(...)) for RST help strings,
    # if cloup's mutually exclusive etc functionality is needed, it should be the latter.
    click.option(
        '-m', '--method', type=click.Choice([m.value for m in Method], case_sensitive=False),
        default=Method.gain_blk_offset.value, show_default=True,
        help="""Correction method.
        \b
        
        - `gain`: gain-only model.
        - `gain-blk-offset`: gain-only model applied to offset normalised image blocks.
        - `gain-offset`: gain and offset model.
        """,
    ),
    click.option(
        '-k', '--kernel-shape', type=click.Tuple([click.INT, click.INT]), nargs=2, default=(5, 5), show_default=True,
        metavar='HEIGHT WIDTH', help='Kernel height and width in pixels of the :option:`--proc-crs` image.'
    ),
    click.option(
        '-od', '--out-dir', type=click.Path(exists=True, file_okay=False, writable=True),
        show_default='source image directory.', help='Directory in which to place corrected image(s).'
    ),
    click.option(
        '-o', '--overwrite', is_flag=True, type=bool, default=False, show_default=True,
        help='Overwrite existing output file(s).'
    ),
    click.option(
        '-cmp', '--compare', 'comp_ref_file', metavar='FILE', type=click.Path(dir_okay=False,
            path_type=pathlib.Path), is_flag=False,
        flag_value='ref', default=None, callback=_compare_cb,
        help='Compare source and corrected images with this reference image.  If no ``FILE`` value is given, source '
             'and corrected images are compared with :option:`REFERENCE`.'
    ),
    click.option(
        '-bo/-nbo', '--build-ovw/--no-build-ovw', type=click.BOOL, default=True, show_default=True,
        help='Build overviews for the output image(s).'
    ),
    click.option(
        '-c', '--conf', type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
        required=False, default=None, show_default=True,
        help='Path to a yaml configuration file specifying advanced options (as follow below).'
    )
)
# advanced options
@cloup.option_group(
    "Advanced options",
    click.option(
        '-pi/-npi', '--param-image/--no-param-image', type=click.BOOL,
        default=RasterFuse.default_homo_config['param_image'], show_default=True,
        help=f'Write the  model parameters and R\N{SUPERSCRIPT TWO} values for each corrected image into a parameter '
             f'image file.'
    ),
    click.option(
        '-mp/-nmp', '--mask-partial/--no-mask-partial', type=click.BOOL,
        default=KernelModel.default_config['mask_partial'], show_default=True,
        help=f'Mask output pixels produced from partial kernel, or source / reference, image coverage.'
    ),
    threads_option,
    click.option(
        '-mbm', '--max-block-mem', type=click.FLOAT, default=RasterFuse.default_homo_config['max_block_mem'],
        show_default=True, help='Maximum image block size in megabytes (0 = block corresponds to the whole image).'
    ),
    click.option(
        '-ds', '--downsampling', type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
        default=KernelModel.default_config['downsampling'], show_default=True,
        help='Resampling method for re-projecting from high to low resolution.  See the `rasterio docs '
             '<https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_ for '
             'details.'
    ),
    click.option(
        '-us', '--upsampling', type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
        default=KernelModel.default_config['upsampling'], show_default=True,
        help='Resampling method for re-projecting from low to high resolution.  See the `rasterio docs '
             '<https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>`_ for '
             'details.'
    ),
    click.option(
        '-rit', '--r2-inpaint-thresh', type=click.FloatRange(min=0, max=1),
        default=KernelModel.default_config['r2_inpaint_thresh'], show_default=True, metavar='FLOAT 0-1',
        help='R\N{SUPERSCRIPT TWO} threshold below which to inpaint model parameters from surrounding areas '
        '(0 = turn off inpainting). Valid for `gain-offset` :option:`--method` only.'
    ),
    click.option(
        '-pc', '--proc-crs', type=click.Choice([pc.value for pc in ProcCrs], case_sensitive=False),
        default=ProcCrs.auto.value, show_default=True,
        help="""The image CRS in which to estimate correction parameters. 
        \b
    
        - `auto`: lowest resolution of the source and reference CRS's (recommended). 
        - `src`: source image CRS. 
        - `ref`: reference image CRS.
        """
    ),
    click.option(
        '--driver', type=click.Choice(set(rio.drivers.raster_driver_extensions().values()), case_sensitive=False),
        default=RasterFuse.default_out_profile['driver'], show_default=True, metavar='TEXT',
        help='Output image format driver.  See the `GDAL docs <https://gdal.org/drivers/raster/index.html>`_ for '
             'details.'
    ),
    click.option(
        '--dtype', type=click.Choice(list(rio.dtypes.dtype_fwd.values())[1:8], case_sensitive=False),
        default=RasterFuse.default_out_profile['dtype'], show_default=True,
        help=f'Output image data type.  Valid for corrected images only, parameter images always use '
             f'{RasterArray.default_dtype}.'
    ),
    click.option(
        '--nodata', 'nodata', type=click.STRING, callback=_nodata_cb, metavar='[NUMBER|null|nan]',
        default=RasterFuse.default_out_profile['nodata'], show_default=True,
        help=f'Output image nodata value.  Valid for corrected images only, parameter images always use '
             f'{RasterArray.default_nodata}.'
    ),
    click.option(
        '-co', '--creation-options', metavar='NAME=VALUE', multiple=True, default=(),
        callback=_creation_options_cb,
        help='Driver specific image creation option(s) for the output image(s).  See the `GDAL docs '
             '<https://gdal.org/drivers/raster/index.html>`_ for details.'
    ),
)
@click.pass_context
def fuse(
    ctx: click.Context, src_file: Tuple[pathlib.Path,], ref_file: pathlib.Path, method: Method,
    kernel_shape: Tuple[int, int], out_dir: pathlib.Path, overwrite: bool, comp_ref_file: pathlib.Path,
    build_ovw: bool, proc_crs: ProcCrs, conf: pathlib.Path, **kwargs
):
    # @formatter:off
    """
    Correct image(s) to surface reflectance.

    Correct source multi-spectral aerial or satellite imagery to approximate surface reflectance, by fusion with a
    reference satellite image.

    For best results, reference and source image(s) should be concurrent, co-located, and spectrally similar.
    Reference image extents must encompass those of the source image(s), and source / reference band ordering should
    match i.e. reference band 1 corresponds to source band 1, reference band 2 corresponds to source band 2 etc.

    Corrected image(s) are named automatically based on the source file name and option values.
    \b

    Examples:
    ---------

    Correct `source.tif` with `reference.tif` using the default options::

        homonim fuse source.tif reference.tif

    Correct `source.tif` with `reference.tif` using the `gain-blk-offset` method, a kernel of 5 x 5 pixels,
    and placing the corrected images in the `corrected` directory::

        homonim fuse --method gain-blk-offset --kernel-shape 5 5 --out-dir ./corrected source.tif reference.tif

    Correct files matching `source*.tif` with `reference1.tif` using the `gain-offset` method and a kernel of 15 x 15
    pixels.  Produce parameter images, mask partially covered pixels in the corrected images, and statistically
    compare source and corrected images with `reference2.tif`::

        homonim fuse -m gain-offset -k 15 15 --param-image --mask-partial --compare reference2.tif source*.tif reference1.tif
    """
    # @formatter:on

    try:
        kernel_shape = utils.validate_kernel_shape(kernel_shape, method=method)
    except Exception as ex:
        raise click.BadParameter(str(ex))

    # build configuration dictionaries for ImFuse
    config = dict(
        homo_config=_update_existing_keys(RasterFuse.default_homo_config, **kwargs),
        model_config=_update_existing_keys(RasterFuse.default_model_config, **kwargs),
        out_profile=_update_existing_keys(RasterFuse.default_out_profile, **kwargs)
    )
    comp_files = []

    # iterate over and homogenise source file(s)
    try:
        for src_filename in src_file:
            homo_path = pathlib.Path(out_dir) if out_dir is not None else src_filename.parent

            logger.info(f'\nHomogenising {src_filename.name}')
            with RasterFuse(
                src_filename, ref_file, homo_path, method=Method(method), kernel_shape=kernel_shape,
                proc_crs=ProcCrs(proc_crs), overwrite=overwrite, **config
            ) as raster_fuse: # yapf: disable
                start_time = timer()
                raster_fuse.process()
                # build overviews
                if build_ovw:
                    logger.info(f'Building overviews')
                    raster_fuse.build_overviews()

            logger.info(f'Completed in {timer() - start_time:.2f} secs')
            comp_files += [src_filename, raster_fuse.homo_filename]  # build a list of files to pass to compare

        # compare source and corrected files with reference (invokes compare command with relevant parameters)
        if comp_ref_file:
            comp_file = ref_file if str(comp_ref_file) == 'ref' else comp_ref_file
            ctx.invoke(compare, src_file=comp_files, ref_file=comp_file)

    except Exception:
        logger.exception('Exception caught during processing.')  # log exception info
        raise click.Abort()


cli.add_command(fuse)


# compare command
@cloup.command(cls=HomonimCommand)
@cloup.argument(
    'src-file', nargs=-1, metavar='IMAGE...', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help='Path(s) to image(s) to compare with :option:`REFERENCE`.'
)
@ref_file_arg
@output_option
def compare(src_file: Tuple[pathlib.Path,], ref_file: pathlib.Path, output: pathlib.Path):
    """
    Compare image(s) with a reference.

    Report similarity statistics between input image(s) and a reference image.  Typically, this is used to compare the
    before and after accuracy of surface reflectance correction, by comparing source and corrected images with a
    new reference image.

    Reference image extents must encompass those of the input image(s), and input / reference band ordering should
    match i.e. reference band 1 corresponds to input band 1, reference band 2 corresponds to input band 2 etc.

    Images will be re-projected and compared in the lowest resolution of the input and reference image CRS's.
    \b

    Examples:
    ---------

    Compare `source.tif` and `corrected.tif` with `reference.tif`::

        homonim compare source.tif corrected.tif reference.tif
    """

    try:
        res_dict = {}
        # iterate over source files, comparing with reference
        for src_filename in src_file:
            logger.info(f'\nComparing {src_filename.name}')
            start_time = timer()
            cmp = RasterCompare(src_filename, ref_file, proc_crs=ProcCrs.auto)
            res_dict[str(src_filename)] = cmp.compare()
            logger.info(f'Completed in {timer() - start_time:.2f} secs')

        # print a key for the following tables
        logger.info(f'\n\n{cmp.stats_key}')

        # print a results table per source file
        summary_dict = {}
        for src_file, _res_dict in res_dict.items():
            res_df = pd.DataFrame.from_dict(_res_dict, orient='index')
            res_str = res_df.to_string(float_format='{:.2f}'.format, index=True, justify='center', index_names=False)
            logger.info(f'\n\n{src_file}:\n\n{res_str}')
            summary_dict[src_file] = _res_dict['Mean']

        # print a summary results table comparing all source files
        if len(summary_dict) > 1:
            summ_df = pd.DataFrame.from_dict(summary_dict, orient='index')
            summ_df = summ_df.rename(columns=dict(zip(summ_df.columns, ('Mean ' + summ_df.columns))))
            summ_df.insert(0, 'File', [pathlib.Path(fn).name for fn in summ_df.index])
            summ_str = summ_df.to_string(float_format='{:.2f}'.format, index=False, justify='center', index_names=False)
            logger.info(f'\n\nSummary over bands:\n\n{summ_str}')

        if output is not None:
            res_dict['Reference'] = ref_file.stem
            with open(output, 'w') as file:
                json.dump(res_dict, file)

    except Exception:
        logger.exception('Exception caught during processing')
        raise click.Abort()


cli.add_command(compare)


@cloup.command(cls=HomonimCommand)
@cloup.argument(
    'param-file', nargs=-1, metavar='PARAM...', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    callback=_param_file_cb, help='Path(s) to parameter image(s).'
)
@output_option
def stats(param_file: pathlib.Path, output: pathlib.Path):
    """
    Report parameter statistics.

    Report the minimum, maximum, mean etc. values of a parameter image generated with the
    :option:`--param-image <homonim-fuse --param-image>` option of the ``fuse`` command.
    """

    try:
        stats_dict = {}
        meta_dict = {}

        # process parameter file(s), storing results
        for param_filename in param_file:
            logger.info(f'\nProcessing {param_filename.name}')
            param_stats = ParamStats(param_filename)
            stats_dict[str(param_filename)] = param_stats.stats()
            meta_dict[str(param_filename)] = param_stats.metadata

        # iterate over stored result(s) and print
        for param_filename, param_dict in stats_dict.items():
            param_meta = meta_dict[param_filename]

            logger.info(f'\n{pathlib.Path(param_filename).name}:\n')
            logger.info(param_meta)
            # format the statistics as a dataframe to get printable string
            param_df = pd.DataFrame.from_dict(param_dict, orient='index')
            param_str = param_df.to_string(
                float_format='{:.2f}'.format, index=True, justify='center', index_names=False
            )
            logger.info(f'Stats:\n{param_str}')

        if output is not None:
            with open(output, 'w') as file:
                json.dump(stats_dict, file, allow_nan=True)

    except Exception:
        logger.exception('Exception caught during processing')
        raise click.Abort()


cli.add_command(stats)

##
