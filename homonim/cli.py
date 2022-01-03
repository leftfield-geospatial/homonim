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

from homonim.compare import ImCompare
from homonim.fuse import ImFuse
from homonim.kernel_model import KernelModel

# print formatting
# np.set_printoptions(precision=4)
# np.set_printoptions(suppress=True)
logger = logging.getLogger(__name__)


def _create_homo_postfix(proc_crs, method, kernel_shape, driver='GTiff'):
    """Create a postfix string for the homogenised image file"""
    ext_dict = rio.drivers.raster_driver_extensions()
    ext_idx = list(ext_dict.values()).index(driver)
    ext = list(ext_dict.keys())[ext_idx]
    post_fix = f'_HOMO_c{proc_crs.upper()}_m{method.upper()}_k{kernel_shape[0]}_{kernel_shape[1]}.{ext}'
    return post_fix


def _update_existing_keys(default_dict, **kwargs):
    """Update values in a dict with args from matching keys in **kwargs"""
    return {k: kwargs.get(k, v) for k, v in default_dict.items()}


def _parse_nodata(ctx, param, value):
    """Return a float or None"""
    if value is None or value.lower() in ["null", "nil", "none", "nada"]:
        return None
    elif value.lower() == "nan":
        return float("nan")
    else:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise click.BadParameter("{!r} is not a number".format(value), param=param, param_hint="nodata")


def _configure_logging(verbosity):
    # adapted from rasterio
    log_level = max(10, 20 - 10 * verbosity)
    formatter = _CustomFormatter()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    logging.root.setLevel(log_level)
    logging.captureWarnings(True)


def _get_chained_param(ctx, param, value, required=False):
    param_source = ctx.get_parameter_source(param.name)
    if (value is None or param_source == ParameterSource.DEFAULT):
        if param.name in ctx.obj:
            return ctx.obj[param.name]
        elif required:
            raise click.MissingParameter(param=param)
    return value

def _str_to_path(ctx, param, value):
    return (pathlib.Path(file_str) for file_str in value)

def _src_file_callback(ctx, param, value):
    print(f'src_file: {value}')
    return value

def _cb_key_val(ctx, param, value):

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

    if not value:
        return {}
    elif isinstance(value, dict):
        return value
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
                out[k] = None if v.lower() in ['none', 'null', 'nil', 'nada'] else v
        return out

class _CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "%(levelname)s:%(name)s: %(message)s"
        return super().format(record)


class _ConfigFileCommand(click.Command):
    """Class to combine config file with command line parameters"""
    # adapted from https://stackoverflow.com/questions/46358797/python-click-supply-arguments-and-options-from-a-configuration-file/46391887
    def invoke(self, ctx):
        config_file = ctx.params['conf']
        if config_file is not None:
            # overwrite context parameters with values from config file
            with open(config_file) as f:
                config_dict = yaml.safe_load(f)
            for conf_key, conf_value in config_dict.items():
                if not conf_key in ctx.params:
                    raise click.BadParameter(f"Unknown config file parameter '{conf_key}'", param="conf",
                                             param_hint="conf")
                else:
                    param_src = ctx.get_parameter_source(conf_key)
                    if (ctx.params[conf_key] is None or param_src == ParameterSource.DEFAULT):
                        # overwrite default or None parameters with values from config file
                        ctx.params[conf_key] = conf_value
        return click.Command.invoke(self, ctx)


class _ChainedCommand(click.Command):
    """Class to override parameter defaults with their corresponding values from chained commands"""
    def invoke(self, ctx):
        for param_key, param_val in ctx.params.items():
            if param_key in ctx.obj:
                param_src = ctx.get_parameter_source(param_key)
                if (ctx.params[param_key] is None or param_src == ParameterSource.DEFAULT):
                    ctx.params[param_key] = ctx.obj[param_key]
        return click.Command.invoke(self, ctx)


proc_crs_option = click.option("-pc", "--proc-crs", type=click.Choice(['ref', 'src', 'auto'], case_sensitive=False),
                               default='auto', show_default=True, help="The image CRS in which to perform processing.")
multithread_option = click.option("-nmt", "--no-multithread", "multithread", type=click.BOOL, is_flag=True,
                                  default=ImFuse.default_homo_config['multithread'],
                                  help=f"Process image blocks consecutively.")
src_file_arg = click.argument("src-file", nargs=-1, metavar="INPUTS...",
                              type=click.Path(exists=False, dir_okay=True, readable=False, path_type=pathlib.Path))
ref_file_arg = click.argument("ref-file", nargs=1, metavar="REFERENCE",
                              type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path))


@click.group()
@click.option(
    '--verbose', '-v',
    count=True,
    help="Increase verbosity.")
@click.option(
    '--quiet', '-q',
    count=True,
    help="Decrease verbosity.")
@click.pass_context
def cli(ctx, verbose, quiet):
    verbosity = verbose - quiet
    _configure_logging(verbosity)
    ctx.obj = {}
    ctx.obj["verbosity"] = verbosity  # TODO - necessary?


@click.command(cls=_ConfigFileCommand)
# @click.option("-s", "--src-file", type=click.STRING,
#               required=True, multiple=True, callback=_src_file_callback,
#               help="Path(s) or wildcard pattern(s)'' specifying the source image file(s).")
# @click.option("-r", "--ref-file",
#               type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
#               required=True, help="Path to the reference image file.")
@src_file_arg
@ref_file_arg
# @files_in_arg
@click.option("-k", "--kernel-shape", type=click.Tuple([click.INT, click.INT]), nargs=2, default=(5, 5),
              show_default=True, help="Sliding kernel (window) width and height (in reference pixels).")
@click.option("-m", "--method", type=click.Choice(['gain', 'gain-im-offset', 'gain-offset'], case_sensitive=False),
              default='gain-im-offset', show_default=True, help="Homogenisation method.")
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
              default=ImFuse.default_homo_config['debug_image'],
              help=f"Create a debug image for each source file containing parameter and R\N{SUPERSCRIPT TWO} values.")
@click.option("-mp", "--mask-partial", type=click.BOOL, is_flag=True,
              default=ImFuse.default_homo_config['mask_partial'],
              help=f"Mask output pixels produced from partial kernel, or source image coverage.")
@multithread_option
@click.option("-mbm", "--max-block-mem", type=click.INT, help="Maximum image block size for processing (MB)",
              default=ImFuse.default_homo_config['max_block_mem'], show_default=True)
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
              default=ImFuse.default_out_profile['driver'], show_default=True, metavar="TEXT",
              help="Output format driver.")
@click.option("--out-dtype", "dtype", type=click.Choice(list(rio.dtypes.dtype_fwd.values())[1:8], case_sensitive=False),
              default=ImFuse.default_out_profile['dtype'], show_default=True, help="Output image data type.")
@click.option("--out-nodata", "nodata", type=click.STRING, callback=_parse_nodata, metavar="[NUMBER|null|nan]",
              default=ImFuse.default_out_profile['nodata'], show_default=True,
              help="Output image nodata value.")
@click.option('--co', '--out-profile', 'creation_options', metavar='NAME=VALUE', multiple=True, callback=_cb_key_val,
              default=tuple(f'{k}={v}' for k,v in ImFuse.default_out_profile['creation_options'].items()),
              show_default=True,
              help="Driver specific creation options.  See the rasterio documentation for more information: "
                   "https://rasterio.readthedocs.io/en/latest/topics/image_options.html.")

# @click.option("--out-blockxsize", "blockxsize", type=click.INT, default=ImFuse.default_out_profile['blockxsize'],
#               show_default=True, help="Output image block width.")
# @click.option("--out-blockysize", "blockysize", type=click.INT, default=ImFuse.default_out_profile['blockysize'],
#               show_default=True, help="Output image block height.")
# @click.option("--out-compress", "compress",
#               type=click.Choice([item.value for item in rio.enums.Compression], case_sensitive=False),
#               default=ImFuse.default_out_profile['compress'], show_default=True,  # metavar="TEXT",
#               help="Output image compression.")
# @click.option("--out-interleave", "interleave", type=click.Choice(["pixel", "band"], case_sensitive=False),
#               default=ImFuse.default_out_profile['interleave'], show_default=True,
#               help="Output image data interleaving.")
# @click.option("--out-photometric", "photometric",
#               type=click.Choice([item.value for item in rio.enums.PhotometricInterp], case_sensitive=False),
#               default=ImFuse.default_out_profile['photometric'], show_default=True,
#               help="Output image photometric interpretation.")
@click.pass_context
def fuse(ctx, src_file, ref_file, kernel_shape, method, output_dir, do_cmp, build_ovw, proc_crs, conf, **kwargs):
    """Radiometrically homogenise image(s) by fusion with a reference"""
    # ctx.obj.update(**ctx.params)  # copy all parameters for chained commands to use
    # ctx.obj['src_file'] = ()  # filled below
    # logger.info(f'files: {files}')
    # src_file = files[:-1]
    # ref_file = files[-1]
    compare_files = []
    config = {}
    config['homo_config'] = _update_existing_keys(ImFuse.default_homo_config, **kwargs)
    config['model_config'] = _update_existing_keys(ImFuse.default_model_config, **kwargs)
    config['out_profile'] = _update_existing_keys(ImFuse.default_out_profile, **kwargs)

    # iterate over and homogenise source file(s)
    try:
        for src_file_spec in src_file:
            src_file_path = pathlib.Path(src_file_spec)
            if len(list(src_file_path.parent.glob(src_file_path.name))) == 0:
                raise click.BadParameter(f'Could not find any source image(s) matching {src_file_path.name}',
                                         param='src_file', param_hint='src_file')

            for src_filename in src_file_path.parent.glob(src_file_path.name):
                if output_dir is not None:
                    homo_root = pathlib.Path(output_dir)
                else:
                    homo_root = src_filename.parent

                logger.info(f'\nHomogenising {src_filename.name}')
                start_ttl = datetime.datetime.now()
                him = ImFuse(src_filename, ref_file, method=method, kernel_shape=kernel_shape, proc_crs=proc_crs,
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
# @click.option("-s", "--src-file", type=click.STRING,
#             required=False, multiple=True, help="Path(s) or wildcard pattern(s) specifying the source image file(s).")
# @click.option("-r", "--ref-file",
#               type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
#               required=False, help="Path to the reference image file.")
@src_file_arg
@ref_file_arg
@proc_crs_option
@multithread_option
@click.option("-o", "--output",
              type=click.Path(exists=False, dir_okay=False, writable=True, resolve_path=True, path_type=pathlib.Path),
              help="Write comparison results in json format.")
@click.pass_context
def compare(ctx, src_file, ref_file, proc_crs, multithread, output):
    """Statistically compare image(s) with a reference"""
    # check the src_file and ref_file have been set either via command line or previous command in the chain
    # for param, name in zip([src_file, ref_file], ['-s / --src-file', '-r / --ref-file']):
    #     if param is None:
    #         raise click.BadOptionUsage(option_name=name, ctx=ctx,
    #                                    message=f'Missing option `{name}`, either pass this on the command line '
    #                                            f'or chain this command with `fuse`.')

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
                cmp = ImCompare(src_filename, ref_file, proc_crs=proc_crs, multithread=multithread)
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
