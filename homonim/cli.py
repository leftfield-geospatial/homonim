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

import datetime
import json
import logging
import pathlib
import sys

import click
import pandas as pd
import rasterio as rio
import yaml
from click.core import ParameterSource
from rasterio.warp import SUPPORTED_RESAMPLING
import numpy as np

from homonim.compare import RasterCompare
from homonim.fuse import RasterFuse
from homonim.kernel_model import KernelModel
from homonim.enums import ProcCrs, Method

logger = logging.getLogger(__name__)


def _create_homo_postfix(proc_crs, method, kernel_shape, driver='GTiff'):
    """Create a postfix string, including extension, for the homogenised image file"""
    ext_dict = rio.drivers.raster_driver_extensions()
    ext_idx = list(ext_dict.values()).index(driver)
    ext = list(ext_dict.keys())[ext_idx]
    post_fix = f'_HOMO_c{proc_crs.name.upper()}_m{method.upper()}_k{kernel_shape[0]}_{kernel_shape[1]}.{ext}'
    return post_fix


def _update_existing_keys(default_dict, **kwargs):
    """Update values in a dict with args from matching keys in **kwargs"""
    return {k: kwargs.get(k, v) for k, v in default_dict.items()}


def _configure_logging(verbosity):
    """configure python logging level"""
    # adapted from rasterio https://github.com/rasterio/rasterio
    log_level = max(10, 20 - 10 * verbosity)
    formatter = _PlainInfoFormatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    logging.root.setLevel(log_level)
    logging.captureWarnings(True)


def _nodata_cb(ctx, param, value):
    """click callback to convert nodata value to None, nan or float"""
    # adapted from rasterio https://github.com/rasterio/rasterio
    if value is None or value.lower() in ["null", "nil", "none", "nada"]:
        return None
    elif value.lower() == "nan":
        return float("nan")
    else:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise click.BadParameter("{!r} is not a number".format(value), param=param, param_hint="nodata")

def _kernel_shape_cb(ctx, param, value):
    """click callback to error check kernel_shape"""
    kernel_shape = np.array(value)
    if np.any(kernel_shape < 1):
        raise click.BadParameter(f'Invalid kernel shape: {kernel_shape}, should be a minimum of one in both dimensions')
    if not np.all(np.mod(kernel_shape, 2) == 1):
        raise click.BadParameter(f'Invalid kernel shape: {kernel_shape}, should be odd in both dimensions')
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
                raise click.BadParameter(
                    "Invalid syntax for KEY=VAL arg: {}".format(pair))
            else:
                k, v = pair.split('=', 1)
                k = k.lower()
                v = v.lower()
                out[k] = None if v.lower() in ['none', 'null', 'nil', 'nada'] else yaml.safe_load(v)
        return out


class _PlainInfoFormatter(logging.Formatter):
    """logging formatter to format INFO logs without the module name etc prefix"""
    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "%(levelname)s:%(name)s: %(message)s"
        return super().format(record)


class _ConfigFileCommand(click.Command):
    """
    click Command to read config file and combine with CLI parameters.

    User-supplied CLI values are given priority, followed by the config file values.
    Where neither user supplied CLI, or config file values, for a click parameter exist,
    it will retain its default value.
    """

    # adapted from https://stackoverflow.com/questions/46358797/python-click-supply-arguments-and-options-from-a-configuration-file/46391887
    def invoke(self, ctx):
        config_file = ctx.params['conf']
        if config_file is not None:

            # read the config file into a dict
            with open(config_file) as f:
                config_dict = yaml.safe_load(f)

            # Replace the click context default value parameters with any config file values
            # Where parameter values have been specified by the user on the command line, they are left as is.
            for conf_key, conf_value in config_dict.items():
                if not conf_key in ctx.params:
                    raise click.BadParameter(f"Unknown config file parameter '{conf_key}'", param="conf",
                                             param_hint="conf")
                else:
                    param_src = ctx.get_parameter_source(conf_key)
                    # overwrite default parameters with values from config file
                    if (ctx.params[conf_key] is None or param_src == ParameterSource.DEFAULT):
                        ctx.params[conf_key] = conf_value
                        ctx.set_parameter_source(conf_key, ParameterSource.COMMANDLINE)
        return click.Command.invoke(self, ctx)


# define click options and arguments common to more than one command
proc_crs_option = click.option("-pc", "--proc-crs", type=click.Choice(ProcCrs, case_sensitive=False),
                               default=ProcCrs.auto.name, show_default=True,
                               help="The image CRS in which to perform processing.")
multithread_option = click.option("-nmt", "--no-multithread", "multithread", type=click.BOOL, is_flag=True,
                                  default=RasterFuse.default_homo_config['multithread'],
                                  help=f"Process image blocks consecutively.")
src_file_arg = click.argument("src-file", nargs=-1, metavar="INPUTS...",
                              type=click.Path(exists=False, dir_okay=True, readable=False, path_type=pathlib.Path))
ref_file_arg = click.argument("ref-file", nargs=1, metavar="REFERENCE",
                              type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path))

# define the click CLI
@click.group()
@click.option(
    '--verbose', '-v',
    count=True,
    help="Increase verbosity.")
@click.option(
    '--quiet', '-q',
    count=True,
    help="Decrease verbosity.")
def cli(verbose, quiet):
    verbosity = verbose - quiet
    _configure_logging(verbosity)


@click.command(cls=_ConfigFileCommand)
@src_file_arg
@ref_file_arg
@click.option("-k", "--kernel-shape", type=click.Tuple([click.INT, click.INT]), nargs=2, default=(5, 5),
              show_default=True, callback=_kernel_shape_cb, metavar='<HEIGHT WIDTH>',
              help="Kernel height and width in pixels (of the the lowest resolution of the source and reference "
                   "images).")
@click.option("-m", "--method", type=click.Choice(Method, case_sensitive=False),
              default=Method.gain_im_offset.name, show_default=True,
              help="Homogenisation method.")
@click.option("-od", "--output-dir", type=click.Path(exists=True, file_okay=False, writable=True),
              help="Directory to create homogenised image(s) in. [default: use source image directory]")
@click.option("-cmp", "--compare", "do_cmp", type=click.BOOL, is_flag=True, default=False,
              help=f"Statistically compare source and homogenised images with the reference.")
@click.option("-nbo", "--no-build-ovw", "build_ovw", type=click.BOOL, is_flag=True, default=True,
              help="Don't build overviews for the created image(s).")
@proc_crs_option
@click.option("-c", "--conf", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
              required=False, default=None, help="Path to a configuration file.")
@click.option("-di", "--debug-image", type=click.BOOL, is_flag=True,  # TODO: normally on and flag to switch off?
              default=RasterFuse.default_homo_config['debug_image'],
              help=f"Create a debug image for each source file containing parameter and R\N{SUPERSCRIPT TWO} values.")
@click.option("-mp", "--mask-partial", type=click.BOOL, is_flag=True,
              default=RasterFuse.default_homo_config['mask_partial'],
              help=f"Mask output pixels produced from partial kernel, or source image coverage.")
@multithread_option
@click.option("-mbm", "--max-block-mem", type=click.INT, help="Maximum image block size for processing (MB)",
              default=RasterFuse.default_homo_config['max_block_mem'], show_default=True)
@click.option("-ds", "--downsampling", type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
              default=KernelModel.default_config['downsampling'], show_default=True,
              help="Resampling method for re-projecting from reference to source CRS.")
@click.option("-us", "--upsampling", type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
              help="Resampling method for re-projecting from source to reference CRS.",
              default=KernelModel.default_config['upsampling'], show_default=True)
@click.option("-rit", "--r2-inpaint-thresh", type=click.FloatRange(min=0, max=1),
              default=KernelModel.default_config['r2_inpaint_thresh'], show_default=True, metavar="FLOAT 0-1",
              help="R\N{SUPERSCRIPT TWO} threshold below which to inpaint (interpolate) model parameters from "
                   "surrounding areas. For 'gain-offset' method only.")
@click.option("--out-driver", "driver",
              type=click.Choice(set(rio.drivers.raster_driver_extensions().values()), case_sensitive=False),
              default=RasterFuse.default_out_profile['driver'], show_default=True, metavar="TEXT",
              help="Output format driver.")
@click.option("--out-dtype", "dtype", type=click.Choice(list(rio.dtypes.dtype_fwd.values())[1:8], case_sensitive=False),
              default=RasterFuse.default_out_profile['dtype'], show_default=True, help="Output image data type.")
@click.option("--out-nodata", "nodata", type=click.STRING, callback=_nodata_cb, metavar="[NUMBER|null|nan]",
              default=RasterFuse.default_out_profile['nodata'], show_default=True,
              help="Output image nodata value.")
# @click.option('--co', '--out-profile', 'creation_options', metavar='NAME=VALUE', multiple=True,
#               default=tuple(f'{k}={v}' for k,v in ImFuse.default_out_profile['creation_options'].items()),
#               show_default=True, callback=_creation_options_cb,
#               help="Driver specific creation options.  See the rasterio documentation for more information: "
#                    "https://rasterio.readthedocs.io/en/latest/topics/image_options.html.")
@click.option('--co', '--out-profile', 'creation_options', metavar='NAME=VALUE', multiple=True,
              default=(), callback=_creation_options_cb,
              help="Driver specific creation options.  See the rasterio documentation for more information.")
@click.pass_context
def fuse(ctx, src_file, ref_file, kernel_shape, method, output_dir, do_cmp, build_ovw, proc_crs, conf, **kwargs):
    """Radiometrically homogenise image(s) by fusion with a reference"""
    compare_files = []
    config = {}
    config['homo_config'] = _update_existing_keys(RasterFuse.default_homo_config, **kwargs)
    config['model_config'] = _update_existing_keys(RasterFuse.default_model_config, **kwargs)
    # if (kwargs['driver'] != ImFuse.default_out_profile['driver'] and
    #         ctx.get_parameter_source('creation_options') == ParameterSource.DEFAULT):
    #     kwargs['creation_options'] = {}
    if (ctx.get_parameter_source('driver') == ParameterSource.DEFAULT and
            ctx.get_parameter_source('creation_options') == ParameterSource.DEFAULT):
        # if no other driver or creation_options have been specified, use the defaults
        kwargs['creation_options'] = RasterFuse.default_out_profile['creation_options']
    config['out_profile'] = _update_existing_keys(RasterFuse.default_out_profile, **kwargs)

    # iterate over and homogenise source file(s)
    try:
        for src_file_spec in src_file:
            src_file_path = pathlib.Path(src_file_spec)
            if len(list(src_file_path.parent.glob(src_file_path.name))) == 0:
                raise click.BadParameter(f'Could not find any source image(s) matching {src_file_path.name}',
                                         param_hint='src_file')

            for src_filename in src_file_path.parent.glob(src_file_path.name):
                if output_dir is not None:
                    homo_root = pathlib.Path(output_dir)
                else:
                    homo_root = src_filename.parent

                logger.info(f'\nHomogenising {src_filename.name}')
                start_ttl = datetime.datetime.now()
                him = RasterFuse(src_filename, ref_file, method=method, kernel_shape=kernel_shape, proc_crs=proc_crs,
                                 **config)

                # create output image filename and homogenise
                post_fix = _create_homo_postfix(proc_crs=proc_crs, method=method, kernel_shape=kernel_shape,
                                                driver=config['out_profile']['driver'])
                homo_filename = homo_root.joinpath(src_filename.stem + post_fix)
                him.homogenise(homo_filename)

                # ttl_time = (datetime.datetime.now() - start_ttl)
                # logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')
                if build_ovw:
                    # build overviews
                    # start_ttl = datetime.datetime.now()
                    logger.info(f'Building overviews for {homo_filename.name}')
                    him.build_overviews(homo_filename)

                    if config['homo_config']['debug_image']:
                        param_out_filename = him._create_debug_filename(homo_filename)
                        logger.info(f'Building overviews for {param_out_filename.name}')
                        him.build_overviews(param_out_filename)

                ttl_time = (datetime.datetime.now() - start_ttl)
                logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')
                # copy individual src and homogenised filenames into context for chained downstream commands to re-use
                # ctx.obj['src_file'] += (src_filename, homo_filename)
                compare_files += (src_filename, homo_filename)
        if do_cmp:
            ctx.invoke(compare, src_file=compare_files, ref_file=ref_file, proc_crs=proc_crs,
                       multithread=kwargs['multithread'])
    except Exception:
        logger.exception("Exception caught during processing")
        raise click.Abort()


cli.add_command(fuse)


@click.command()
@src_file_arg
@ref_file_arg
@proc_crs_option
@multithread_option
@click.option("-o", "--output",
              type=click.Path(exists=False, dir_okay=False, writable=True, resolve_path=True, path_type=pathlib.Path),
              help="Write comparison results in json format.")
def compare(src_file, ref_file, proc_crs, multithread, output):
    """Statistically compare image(s) with a reference"""
    try:
        res_dict = {}
        for src_file_spec in src_file:
            src_file_path = pathlib.Path(src_file_spec)
            if len(list(src_file_path.parent.glob(src_file_path.name))) == 0:
                raise click.BadParameter(f'Could not find any source image(s) matching {src_file_path.name}',
                                         param='src_file', param_hint='src_file')

            for src_filename in src_file_path.parent.glob(src_file_path.name):
                logger.info(f'\nComparing {src_filename.name}')
                start_ttl = datetime.datetime.now()
                cmp = RasterCompare(src_filename, ref_file, proc_crs=proc_crs, multithread=multithread)
                # TODO: what if file stems are identical
                res_dict[src_filename.stem] = cmp.compare()
                ttl_time = (datetime.datetime.now() - start_ttl)
                logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')
        # print results
        summary_dict = {}
        for src_file, _res_dict in res_dict.items():
            res_df = pd.DataFrame.from_dict(_res_dict, orient='index')
            res_str = res_df.to_string(float_format="{:.2f}".format, index=True, justify="center",
                                                        index_names=False)
            logger.info(f'\n\n{src_file}:\n\n{res_str}')
            summary_dict[src_file] = _res_dict['Mean']

        if len(summary_dict) > 1:
            summ_df = pd.DataFrame.from_dict(summary_dict, orient='index')
            summ_df = summ_df.rename(columns=dict(zip(summ_df.columns, ('Mean ' + summ_df.columns))))
            summ_str = summ_df.to_string(float_format="{:.2f}".format, index=True, justify="center",
                                                          index_names=False)
            logger.info(f'\n\nSummary:\n\n{summ_str}')

        if output is not None:
            res_dict['Reference'] = ref_file.stem
            with open(output, 'w') as file:
                json.dump(res_dict, file)

    except Exception:
        logger.exception("Exception caught during processing")
        raise click.Abort()


cli.add_command(compare)

##
