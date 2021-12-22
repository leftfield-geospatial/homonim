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
import pathlib

import click
import numpy as np
import rasterio as rio
import yaml
from click.core import ParameterSource
from rasterio.warp import SUPPORTED_RESAMPLING

from homonim import get_logger
from homonim.homonim import HomonIm
from homonim.kernel_model import KernelModel

# print formatting
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
logger = get_logger(__name__)


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


def _create_homo_postfix(homo_crs=None, method=None, kernel_shape=None):
    """Create a postfix string for the homogenised raster file"""
    post_fix = f'_HOMO_c{homo_crs.upper()}_m{method.upper()}_k{kernel_shape[0]}_{kernel_shape[1]}.tif'
    return post_fix


def _update_existing_keys(default_dict, **kwargs):
    """Update values in a dict with args from matching keys in **kwargs"""
    return {k: kwargs.get(k, v) for k, v in default_dict.items()}


def _parse_output_profile(ctx, param, value):
    if ctx.get_parameter_source(param.name) == ParameterSource.DEFAULT:  # defaults
        return dict(value)
    else:
        out = {}
        for k, v in value:
            out[k] = None if v.lower() in ['none', 'null', 'nil', 'nada'] else yaml.safe_load(v)
        return out


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


@click.command(cls=_ConfigFileCommand)
@click.option("-s", "--src-file", type=click.Path(), required=True, multiple=True,
              help="Path(s) or wildcard pattern(s) specifying the source image file(s).")
@click.option("-r", "--ref-file", type=click.Path(exists=True, dir_okay=False, readable=True), required=True,
              help="Path to the reference image file.")
@click.option("-k", "--kernel-shape", type=click.Tuple([click.INT, click.INT]), nargs=2, default=(5, 5),
              show_default=True, help="Sliding kernel (window) width and height.")
@click.option("-m", "--method", type=click.Choice(['gain', 'gain-im-offset', 'gain-offset'], case_sensitive=False),
              default='gain-im-offset', show_default=True, help="Homogenisation method.")
@click.option("-mc", "--model-crs", type=click.Choice(['ref', 'src'], case_sensitive=False), default='ref',
              show_default=True, help="Derive homogenisation model in ref (reference) or src (source) image CRS.")
@click.option("-od", "--output-dir", type=click.Path(exists=True, file_okay=False, writable=True),
              help="Directory to create homogenised image(s) in. [default: use --src-file directory]")
@click.option("-nbo", "--no-build-ovw", "build_ovw", type=click.BOOL, is_flag=True, default=True,
              help="Don't build overviews for the created image(s).")
@click.option("-c", "--conf", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=pathlib.Path),
              required=False, default=None, help="Path to a configuration file.")
@click.option("-dr", "--debug-raster", type=click.BOOL, is_flag=True,
              default=HomonIm.default_homo_config['debug_raster'],
              help=f"Create a debug raster for each source file containing parameter and R\N{SUPERSCRIPT TWO} values.")
@click.option("-mp", "--mask-partial", type=click.BOOL, is_flag=True,
              default=HomonIm.default_homo_config['mask_partial'],
              help=f"Mask output pixels produced from partial kernel, or source image coverage.")
@click.option("-nmt", "--no-mutlithread", "multithread", type=click.BOOL, is_flag=True,
              default=HomonIm.default_homo_config['multithread'],
              help=f"Process image blocks consecutively.")
@click.option("-mbm", "--max-block-mem", type=click.INT, help="Maximum image block size for processing (MB)",
              default=HomonIm.default_homo_config['max_block_mem'], show_default=True)
@click.option("-sri", "--src2ref-interp", type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
              default=KernelModel.default_config['src2ref_interp'], show_default=True,
              help="Resampling method for re-projecting from reference to source CRS.")
@click.option("-rsi", "--ref2src-interp", type=click.Choice([r.name for r in rio.warp.SUPPORTED_RESAMPLING]),
              help="Resampling method for re-projecting from source to reference CRS.",
              default=KernelModel.default_config['ref2src_interp'], show_default=True)
@click.option("-rit", "--r2-inpaint-thresh", type=click.FloatRange(min=0, max=1),
              default=KernelModel.default_config['r2_inpaint_thresh'], show_default=True, metavar="FLOAT 0-1",
              help="R\N{SUPERSCRIPT TWO} threshold below which to inpaint (interpolate) model parameters from "
                   "surrounding areas. For 'gain-offset' method only.")
@click.option("--out-driver", "driver",
              type=click.Choice(set(rio.drivers.raster_driver_extensions().values()), case_sensitive=False),
              default=HomonIm.default_out_profile['driver'], show_default=True, metavar="TEXT",
              help="Output format driver.")
@click.option("--out-dtype", "dtype", type=click.Choice(list(rio.dtypes.dtype_fwd.values())[1:8], case_sensitive=False),
              default=HomonIm.default_out_profile['dtype'], show_default=True, help="Output raster data type.")
@click.option("--out-blockxsize", "blockxsize", type=click.INT, default=HomonIm.default_out_profile['blockxsize'],
              show_default=True, help="Output raster block width.")
@click.option("--out-blockysize", "blockysize", type=click.INT, default=HomonIm.default_out_profile['blockysize'],
              show_default=True, help="Output raster block height.")
@click.option("--out-compress", "compress",
              type=click.Choice([item.value for item in rio.enums.Compression], case_sensitive=False),
              default=HomonIm.default_out_profile['compress'], show_default=True,  # metavar="TEXT",
              help="Output raster compression.")
@click.option("--out-interleave", "interleave", type=click.Choice(["pixel", "band"], case_sensitive=False),
              default=HomonIm.default_out_profile['interleave'], show_default=True,
              help="Output raster data interleaving.")
@click.option("--out-photometric", "photometric",
              type=click.Choice([item.value for item in rio.enums.PhotometricInterp], case_sensitive=False),
              default=HomonIm.default_out_profile['photometric'], show_default=True,
              help="Output raster photometric interpretation.")
@click.option("--out-nodata", "nodata", type=click.STRING, callback=_parse_nodata, metavar="[NUMBER|null|nan]",
              default=HomonIm.default_out_profile['nodata'], show_default=True,
              help="Output raster nodata value.")
def cli(src_file, ref_file, kernel_shape, method, model_crs, output_dir, build_ovw, conf, **kwargs):
    """Radiometrically homogenise image(s) by fusion with reference satellite data"""

    config = {}
    config['homo_config'] = _update_existing_keys(HomonIm.default_homo_config, **kwargs)
    config['model_config'] = _update_existing_keys(HomonIm.default_model_config, **kwargs)
    config['out_profile'] = _update_existing_keys(HomonIm.default_out_profile, **kwargs)

    # iterate over and homogenise source file(s)
    for src_file_spec in src_file:
        src_file_path = pathlib.Path(src_file_spec)
        if len(list(src_file_path.parent.glob(src_file_path.name))) == 0:
            raise Exception(f'Could not find any source image(s) matching {src_file_spec}')

        for src_filename in src_file_path.parent.glob(src_file_path.name):
            if output_dir is not None:
                homo_root = pathlib.Path(output_dir)
            else:
                homo_root = src_filename.parent

            logger.info(f'Homogenising {src_filename.name}')
            start_ttl = datetime.datetime.now()
            him = HomonIm(src_filename, ref_file, method=method, kernel_shape=kernel_shape, model_crs=model_crs,
                          **config)

            # create output raster filename and homogenise
            post_fix = _create_homo_postfix(homo_crs=model_crs, method=method, kernel_shape=kernel_shape)
            homo_filename = homo_root.joinpath(src_filename.stem + post_fix)
            him.homogenise(homo_filename)

            # set metadata in output file
            him.set_homo_metadata(homo_filename)

            if config['homo_config']['debug_raster']:
                param_out_filename = him._create_debug_filename(homo_filename)
                him.set_debug_metadata(param_out_filename)

            ttl_time = (datetime.datetime.now() - start_ttl)
            logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')

            if build_ovw:
                # build overviews
                start_ttl = datetime.datetime.now()
                logger.info(f'Building overviews for {homo_filename.name}')
                him.build_overviews(homo_filename)

                if config['homo_config']['debug_raster']:
                    logger.info(f'Building overviews for {param_out_filename.name}')
                    him.build_overviews(param_out_filename)

                ttl_time = (datetime.datetime.now() - start_ttl)
                logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')
