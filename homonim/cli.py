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
import yaml


from homonim import homonim
from homonim import root_path, get_logger


# print formatting
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
logger = get_logger(__name__)

def _create_homo_postfix(space=None, method=None, win_size=None, normalise=None):
    if space == 'ref-space':
        post_fix = f'_HOMO_sREF_m{method.upper()}_n{"ON" if normalise else "OFF"}_w{win_size[0]}_{win_size[1]}.tif'
    else:
        post_fix = f'_HOMO_sSRC_m{method.upper()}_n{"ON" if normalise else "OFF"}_w{win_size[0]}_{win_size[1]}.tif'
    return post_fix

@click.command()
@click.option(
    "-s",
    "--src-file",
    type=click.Path(exists=False),  # check below
    help="path(s) or wildcard pattern(s) specifying the source image file(s)",
    required=True,
    multiple=True
)
@click.option(
    "-r",
    "--ref-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="path to the reference image file",
    required=True,
)
@click.option(
    "-w",
    "--win-size",
    type=click.Tuple([click.INT, click.INT]),
    nargs=2,
    help="sliding window width and height (e.g. -w 3 3) [default: use source directory]",
    required=False,
    default=(3, 3),
    show_default=True,
)
@click.option(
    "-m",
    "--method",
    type=click.Choice(['gain_only', 'gain_offset'], case_sensitive=False),
    help="homogenisation method",
    default='gain_only',
    show_default=True,
)
@click.option(
    "-n/-nn",
    "--norm/--no-norm",
    default=False,
    help="Do/don't pre-process with image-wide normalisation.  [default: --no-norm]",
    required=False,
)
@click.option(
    "-rs",
    "--ref-space",
    'homo_space',
    help="Homogenise in source or reference image space.  [default: --ref-space]",
    flag_value='ref-space',
    default=True,
    required=False,
)
@click.option(
    "-ss",
    "--src-space",
    'homo_space',
    help="Homogenise in source or reference image space.  [default: --ref-space]",
    flag_value='src-space',
    default=False,
    required=False,
)
@click.option(
    "-n/-nn",
    "--norm/--no-norm",
    default=False,
    help="Do/don't pre-process with image-wide normalisation.  [default: --no-norm]",
    required=False,
)
@click.option(
    "-od",
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    help="directory to write homogenised image(s) in [default: use src-file directory]",
    required=False,
)
@click.option(
    "-bo/-nbo",
    "--build-ovw/--no-build-ovw",
    default=True,
    help="Do/don't build overviews for the homogenised image.  [default: --build-ovw]",
    required=False,
)
@click.option(
    "-c",
    "--conf",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="path to a configuration file",
    required=False
)
def cli(src_file=None, ref_file=None, win_size=(3, 3), method="gain_only", norm=False, homo_space='ref-space',
        output_dir=None, build_ovw=True, conf=None):
    """Radiometrically homogenise image(s) by fusion with reference satellite data"""

    # read configuration
    if conf is None:
        conf_filename = root_path.joinpath('config.yaml')
    else:
        conf_filename = pathlib.Path(conf)

    if not conf_filename.exists():
        raise Exception(f'Config file {conf_filename} does not exist')

    with open(conf_filename, 'r') as f:
        config = yaml.safe_load(f)

    method = method.lower()
    # check src_file points to exisiting file(s)
    for src_file_spec in src_file:
        src_file_path = pathlib.Path(src_file_spec)
        if len(list(src_file_path.parent.glob(src_file_path.name))) == 0:
            raise Exception(f'Could not find any source image(s) matching {src_file_spec}')

        for src_filename in src_file_path.parent.glob(src_file_path.name):
            # try:
            # set homogenised filename
            if output_dir is not None:
                homo_root = pathlib.Path(output_dir)
            else:
                homo_root = src_filename.parent

            logger.info(f'Homogenising {src_filename.name}')
            start_ttl = datetime.datetime.now()
            if homo_space == 'ref-space':
                him = homonim.HomonimRefSpace(src_filename, ref_file, homo_config=config['homogenisation'], out_config=config['output'])
            else:
                him = homonim.HomonimSrcSpace(src_filename, ref_file, homo_config=config['homogenisation'], out_config=config['output'])

            post_fix = _create_homo_postfix(space=homo_space, method=method, win_size=win_size, normalise=norm)
            homo_filename = homo_root.joinpath(src_filename.stem + post_fix)
            him.homogenise(homo_filename, method=method, win_size=win_size, normalise=norm)

            # set metadata in output file
            meta_dict = dict(HOMO_SRC_FILE=src_filename.name, HOMO_REF_FILE=pathlib.Path(ref_file).name,
                             HOMO_SPACE=homo_space, HOMO_METHOD=method, HOMO_WIN_SIZE=win_size, HOMO_NORM=norm,
                             HOMO_CONF=str(config['homogenisation']))
            him.set_metadata(homo_filename, **meta_dict)

            ttl_time = (datetime.datetime.now() - start_ttl)
            logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')

            if build_ovw:
                start_ttl = datetime.datetime.now()
                logger.info(f'Building overviews for {homo_filename.name}')
                him.build_overviews(homo_filename)

                if config['homogenisation']['debug']:
                    param_out_filename = him._create_param_filename(homo_filename)
                    logger.info(f'Building overviews for {param_out_filename.name}')
                    him.build_overviews(param_out_filename)

                ttl_time = (datetime.datetime.now() - start_ttl)
                logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')

            # except Exception as ex:
            #     # catch exceptions so that problem image(s) don't prevent processing of a batch
            #     logger.error('Exception: ' + str(ex))
