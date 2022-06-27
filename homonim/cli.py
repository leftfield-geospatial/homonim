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

import json
import logging
import pathlib
import sys
import re
from timeit import default_timer as timer

import click
import cloup
import math
import pandas as pd
import rasterio as rio
import yaml
from click.core import ParameterSource
from rasterio.warp import SUPPORTED_RESAMPLING

from homonim import utils, version
from homonim.compare import RasterCompare
from homonim.enums import ProcCrs, Method
from homonim.errors import ImageFormatError
from homonim.fuse import RasterFuse
from homonim.kernel_model import KernelModel
from homonim.stats import ParamStats

logger = logging.getLogger(__name__)

class PlainInfoFormatter(logging.Formatter):
    """ logging formatter to format INFO logs without the module name etc prefix. """

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = '%(message)s'
        else:
            self._style._fmt = '%(levelname)s:%(name)s: %(message)s'
        return super().format(record)

class HomonimCommand(cloup.Command):
    """ cloup Command sub-class for formatting help with RST markup. """
    def get_help(self, ctx):
        """ Strip some RST markup from the help text for CLI display.  Assumes no grid tables. """
        if not hasattr(self, 'wrap_text'):
            self.wrap_text = cloup.formatting._formatter.wrap_text
        sub_strings = {
            '\b\n': '\n\b',  # convert from RST friendly to click literal (unwrapped) block marker
            ':option:': '',  # strip ':option:'
            '\| ': '',  # strip RST literal (unwrapped) marker in e.g. tables and bullet lists
            '\n\.\. _.*:\n': '',  # strip RST ref directive '\n.. <name>:\n'
            '`(.*)<(.*)>`_': '\g<1>',  # convert from RST cross-ref '`<name> <<link>>`_' to 'name'
            '::': ':'  # convert from RST '::' to ':'
        }

        def reformat_text(text, width, **kwargs):
            for sub_key, sub_value in sub_strings.items():
                text = re.sub(sub_key, sub_value, text, flags=re.DOTALL)
            return self.wrap_text(text, width, **kwargs)

        cloup.formatting._formatter.wrap_text = reformat_text
        return cloup.Command.get_help(self, ctx)

class FuseCommand(HomonimCommand):
    """ click Command sub-class for setting ``fuse`` parameters from a config file. """

    def invoke(self, ctx):
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
                    # overwrite default parameters with values from config file
                    if ctx.params[conf_key] is None or param_src == ParameterSource.DEFAULT:
                        ctx.params[conf_key] = conf_value
                        ctx.set_parameter_source(conf_key, ParameterSource.COMMANDLINE)

        # set the default creation_options if no other driver or creation_options have been specified
        # (this can't be done in a callback as it depends on 'driver')
        if (ctx.get_parameter_source('driver') == ParameterSource.DEFAULT and
                ctx.get_parameter_source('creation_options') == ParameterSource.DEFAULT):
            ctx.params['creation_options'] = RasterFuse.default_out_profile['creation_options']

        return click.Command.invoke(self, ctx)

def _update_existing_keys(default_dict, **kwargs):
    """ Update values in a dict with args from matching keys in **kwargs. """
    return {k: kwargs.get(k, v) for k, v in default_dict.items()}


def _configure_logging(verbosity):
    """ configure python logging level."""
    # adapted from rasterio https://github.com/rasterio/rasterio
    log_level = max(10, 20 - 10 * verbosity)

    # limit logging config to homonim by applying to package logger, rather than root logger
    # pkg_logger level etc are then 'inherited' by logger = getLogger(__name__) in the modules
    pkg_logger = logging.getLogger(__package__)
    formatter = PlainInfoFormatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(log_level)
    logging.captureWarnings(True)


def _threads_cb(ctx, param, value):
    """ click callback to validate threads. """
    try:
        threads = utils.validate_threads(value)
    except Exception as ex:
        raise click.BadParameter(str(ex))
    return threads


def _nodata_cb(ctx, param, value):
    """ click callback to convert nodata value to None, nan or float. """
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


def _compare_cb(ctx, param, value):
    if value and str(value) != 'ref':
        if not pathlib.Path(value).exists():
            raise click.BadParameter(f'Comparison image does not exist: {value}')
    return value


def _creation_options_cb(ctx, param, value):
    """
    click callback to validate `--opt KEY1=VAL1 --opt KEY2=VAL2` and collect
    in a dictionary like the one below, which is what the CLI function receives.
    If no value or `None` is received then an empty dictionary is returned.
        {
            'KEY1': 'VAL1',
            'KEY2': 'VAL2'
        }
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


def _param_file_cb(ctx, param, value):
    """ click callback to validate parameter image file(s). """
    for filename in value:
        filename = pathlib.Path(filename)
        try:
            utils.validate_param_image(filename)
        except (FileNotFoundError, ImageFormatError):
            raise click.BadParameter(f'{filename.name} is not a valid parameter image.', param=param)
    return value

context_settings = cloup.Context.settings(
    formatter_settings=cloup.HelpFormatter.settings(col2_min_width=math.inf, theme=cloup.HelpTheme.dark())
)

# define click options and arguments common to more than one command
src_file_arg = cloup.argument(
    'src-file', nargs=-1, metavar='INPUTS...', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help='Path(s) to source image(s) to be corrected.'
)
ref_file_arg = cloup.argument(
    'ref-file', nargs=1, metavar='REFERENCE', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help='Path to a surface reflectance reference image.'
)

threads_option = cloup.option(
    '-t', '--threads', type=click.INT, default=RasterFuse.default_homo_config['threads'], show_default=True,
    callback=_threads_cb, help=f'Number of image blocks to process concurrently (0 = use all cpus).'
)
output_option = cloup.option(
    '-o', '--output',
    type=click.Path(exists=False, dir_okay=False, writable=True, path_type=pathlib.Path),
    help='Write results to this json file.'
)


# define the click CLI
@cloup.group(context_settings=context_settings)
@cloup.option('--verbose', '-v', count=True, help='Increase verbosity.')
@cloup.option('--quiet', '-q', count=True, help='Decrease verbosity.')
@click.version_option(version=version.__version__, message='%(version)s')
def cli(verbose, quiet):
    """ Surface reflectance correction and comparison of aerial and satellite imagery. """
    verbosity = verbose - quiet
    _configure_logging(verbosity)


# fuse command
@cloup.command(cls=FuseCommand)
# standard options
@src_file_arg
@ref_file_arg
@cloup.option_group(
    "Standard options",
    cloup.option(
        '-m', '--method', type=click.Choice([m.value for m in Method], case_sensitive=False),
        default=Method.gain_blk_offset.value, help='Correction method.'
    ),
    cloup.option(
        '-k', '--kernel-shape', type=click.Tuple([click.INT, click.INT]), nargs=2, default=(5, 5), show_default=True,
        metavar='<HEIGHT WIDTH>', help='Kernel height and width in pixels of the ``--proc-crs`` image.'
    ),
    cloup.option(
        '-od', '--output-dir', type=click.Path(exists=True, file_okay=False, writable=True),
        show_default='Source image directory.', help='Directory in which to place corrected image(s).'
    ),
    cloup.option(
        '-ovw', '--overwrite', 'overwrite', is_flag=True, type=bool, default=False, show_default=True,
        help='Overwrite existing output file(s).'
    ),
    cloup.option(
        '-cmp', '--compare', 'comp_file', type=click.Path(dir_okay=False, path_type=pathlib.Path), is_flag=False,
        flag_value='ref', default=None, callback=_compare_cb,
        help='Statistically compare source and corrected images with this image.  If specified without an '
        'image file, source and corrected images will be compared with the reference.'
    ),
    cloup.option(
        '-bo/-nbo', '--build-ovw/--no-build-ovw', type=click.BOOL, default=True, show_default=True,
        help='Build overviews for the corrected image(s).'
    ),
    cloup.option(
        '-pc', '--proc-crs', type=click.Choice([pc.value for pc in ProcCrs], case_sensitive=False),
        default=ProcCrs.auto.value, help='The image CRS in which to estimate correction parameters.'
    ),
    cloup.option(
        '-c', '--conf', type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
        required=False, default=None, show_default=True,
        help='Path to a yaml configuration file specifying the options below.'
    )
)

# advanced options
@cloup.option_group(
    "Advanced options",
    cloup.option(
        '-pi/-npi', '--param-image/--no-param-image', type=click.BOOL,
        default=RasterFuse.default_homo_config['param_image'], show_default=True,
        help=f'Create a debug image, containing model parameters and R\N{SUPERSCRIPT TWO} values for each '
        'corrected image.'
    ),
    cloup.option(
        '-mp/-nmp', '--mask-partial/--no-mask-partial', type=click.BOOL,
        default=KernelModel.default_config['mask_partial'], show_default=True,
        help=f'Mask biased corrected pixels produced from partial kernel or source / reference image coverage.'
    ),
    threads_option,
    cloup.option(
        '-mbm', '--max-block-mem', type=click.FLOAT, default=RasterFuse.default_homo_config['max_block_mem'],
        show_default=True, help='Maximum image block size in megabytes (0 = block size is the image size).'
    ),
    cloup.option(
        '-ds', '--downsampling', type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
        default=KernelModel.default_config['downsampling'], show_default=True,
        help='Resampling method for re-projecting from high to low resolution.'
    ),
    cloup.option(
        '-us', '--upsampling', type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
        default=KernelModel.default_config['upsampling'], show_default=True,
        help='Resampling method for re-projecting from low to high resolution.'
    ),
    cloup.option(
        '-rit', '--r2-inpaint-thresh', type=click.FloatRange(min=0, max=1),
        default=KernelModel.default_config['r2_inpaint_thresh'], show_default=True, metavar='FLOAT 0-1',
        help='R\N{SUPERSCRIPT TWO} threshold below which to inpaint model parameters from surrounding areas '
             '(0 = turn off inpainting). For "gain-offset" method only.'
    ),
    cloup.option(
        '--out-driver', 'driver',
        type=click.Choice(list(rio.drivers.raster_driver_extensions().values()), case_sensitive=False),
        default=RasterFuse.default_out_profile['driver'], show_default=True, metavar='TEXT',
        help='Output image format driver.  See the GDAL docs for options.'
    ),
    cloup.option(
        '--out-dtype', 'dtype', type=click.Choice(list(rio.dtypes.dtype_fwd.values())[1:8], case_sensitive=False),
        default=RasterFuse.default_out_profile['dtype'], show_default=True, help='Output image data type.'
    ),
    cloup.option(
        '--out-nodata', 'nodata', type=click.STRING, callback=_nodata_cb, metavar='[NUMBER|null|nan]',
        default=RasterFuse.default_out_profile['nodata'], show_default=True, help='Output image nodata value.'
    ),
    cloup.option(
        '-co', '--out-profile', 'creation_options', metavar='NAME=VALUE', multiple=True, default=(),
        callback=_creation_options_cb,
        help='Driver specific image creation options for the output image(s).  See the GDAL docs for details.'
    ),
)
@click.pass_context
def fuse(
    ctx, src_file, ref_file, method, kernel_shape, output_dir, overwrite, comp_file, build_ovw, proc_crs, conf, **kwargs
):
    # @formatter:on
    """
    Correct image(s) to surface reflectance, by fusion with a reference.

    The *reference* image bounds should contain those of the *source* image(s), and *source* / *reference* bands should
    correspond i.e. reference band 1 corresponds to source band 1, reference band 2 corresponds to source band 2 etc.

    For best results, the reference and source image(s) should be concurrent, co-located, and spectrally similar.

    The following options for ``method`` are available:
    \b

        * `gain`: Gain-only model.
        * | `gain-blk-offset`: Gain-only model applied to offset normalised image
          | blocks.
        * `gain-offset`: Full gain and offset model.

    The following options ``--proc-crs`` :
    \b

        * | `auto`: Estimate in the lowest resolution of the source and reference
          | image CRS's (recommended). \r
        * `src`: Estimate in the source image CRS.
        * `ref`: Estimate in the reference image CRS.


    \b

    Examples:
    ---------

    Correct 'source.tif' with 'reference.tif', using the 'gain-blk-offset' method, and a kernel of 5 x 5 pixels::

        homonim fuse -m gain-blk-offset -k 5 5 source.tif reference.tif


    Correct files matching 'source*.tif' with 'reference.tif', using the 'gain-offset' method and a kernel of 15 x 15
    pixels. Place corrected files in the './homog' directory, produce parameter images, and mask
    partially covered pixels in the corrected images::

        homonim fuse --method gain-offset --kernel-shape 15 15 -od ./homog --param-image --mask-partial source*.tif reference.tif

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
    compare_files = []

    # iterate over and homogenise source file(s)
    try:
        for src_filename in src_file:
            homo_path = pathlib.Path(output_dir) if output_dir is not None else src_filename.parent

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
            compare_files += [src_filename, raster_fuse.homo_filename]  # build a list of files to pass to compare

            # compare source and corrected files with reference (invokes compare command with relevant parameters)
            if comp_file:
                comp_file = ref_file if str(comp_file) == 'ref' else comp_file
                ctx.invoke(compare, src_file=compare_files, ref_file=comp_file, proc_crs=proc_crs)
    except Exception:
        logger.exception('Exception caught during processing.')  # log exception info
        raise click.Abort()


cli.add_command(fuse)


# compare command
@cloup.command(cls=HomonimCommand)
@src_file_arg
@ref_file_arg
@cloup.option(
    '-pc', '--proc-crs', type=click.Choice([pc.value for pc in ProcCrs], case_sensitive=False),
    default=ProcCrs.auto.value, show_default=True, help='The image CRS in which to compare.'
)
@output_option
def compare(src_file, ref_file, proc_crs, output):
    """
    Report similarity statistics between image(s) and a reference.

    Reference image extents should encompass those of the input image(s), and input / reference bands should correspond
    (i.e. reference band 1 corresponds to input band 1, reference band 2 corresponds to input band 2 etc).
    \b

    Examples:
    ---------

    Compare 'source.tif' and 'corrected.tif with 'reference.tif'::

        homonim compare source.tif corrected.tif reference.tif
    """

    try:
        res_dict = {}
        # iterate over source files, comparing with reference
        for src_filename in src_file:
            logger.info(f'\nComparing {src_filename.name}')
            start_time = timer()
            cmp = RasterCompare(src_filename, ref_file, proc_crs=ProcCrs(proc_crs))
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
    'param-file', nargs=-1, metavar='INPUTS...', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    callback=_param_file_cb, help='Path(s) to parameter image(s).'
)
@output_option
def stats(param_file, output):
    """ Report parameter image statistics. """

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
