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

import argparse
import datetime
import os
import pathlib

import numpy as np
import pandas as pd
import rasterio as rio
import yaml
from homonim import homonim
from homonim import root_path, get_logger

# print formatting
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
logger = get_logger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Radiometrically homogenise an image by fusion with satellite data.')
    parser.add_argument("src_file", help="path or wildcard specifying the source image file(s)", type=str,
                        metavar='src_file', nargs='+')
    parser.add_argument("ref_file", help="path to the satellite reference image", type=str)
    parser.add_argument("-w", "--win-size", help="sliding window width and height (e.g. -w 3 3)", nargs='+', type=int)
    parser.add_argument("-hd", "--homo-dir",
                        help="write homogenised image(s) to this directory (default: write to source directory)",
                        type=str)
    parser.add_argument("-rc", "--readconf",
                        help="read custom config from this path (default: use config.yaml in homonim root)",
                        type=str)
    parser.add_argument("-wc", "--writeconf", help="write default config to this path and exit", type=str)
    parser.add_argument("-v", "--verbosity", choices=[1, 2, 3, 4],
                        help="logging level: 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR (default: 2)", type=int)
    return parser.parse_args()


def _process_args(args):
    # set logging level
    if args.verbosity is not None:
        logger.setLevel(10 * args.verbosity)
        homonim.logger.setLevel(10 * args.verbosity)

    # read configuration
    if args.readconf is None:
        config_filename = root_path.joinpath('config.yaml')
    else:
        config_filename = pathlib.Path(args.readconf)

    if not config_filename.exists():
        raise Exception(f'Config file {config_filename} does not exist')

    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)

    # write configuration if requested and exit
    if args.writeconf is not None:
        out_config_filename = pathlib.Path(args.writeconf)
        with open(out_config_filename, 'w') as f:
            yaml.dump(config, stream=f)
        logger.info(f'Wrote config to {out_config_filename}')
        exit(0)

    # check files exist
    for src_im_file_spec in args.src_file:
        src_im_file_path = pathlib.Path(src_im_file_spec)
        if len(list(src_im_file_path.parent.glob(src_im_file_path.name))) == 0:
            raise Exception(f'Could not find any source image(s) matching {src_im_file_spec}')

    if not pathlib.Path(args.ref_file).exists():
        raise Exception(f'Reference file {args.ref_file} does not exist')

    if args.homo_dir is not None:
        homo_dir = pathlib.Path(args.homo_dir)
        if not homo_dir.is_dir():
            raise Exception(f'Ortho directory {args.homo_dir} is not a valid directory')
        if not homo_dir.exists():
            logger.warning(f'Creating output directory {args.homo_dir}')
            os.mkdir(str(homo_dir))

    if args.win_size is None:
        args.win_size = [3, 3]

    return config


def main(args):
    """
    Homogenise an image

    Parameters
    ----------
    args :  ArgumentParser.parse_args()
            Run `python homonim.py -h` to see help on arguments
    """

    try:
        # check args and get config
        config = _process_args(args)

        # loop through image file(s) or wildcard(s), or combinations thereof
        for src_file_spec in args.src_file:
            src_file_path = pathlib.Path(src_file_spec)
            for src_filename in src_file_path.parent.glob(src_file_path.name):
                try:
                    # set homogenised filename
                    if args.homo_dir is not None:
                        homo_filename = pathlib.Path(args.homo_dir).joinpath(src_filename.stem + '_HOMO_SRC_w33.tif')
                    else:
                        homo_filename = src_filename.parent.joinpath(src_filename.stem + '_HOMO_SRC_w33.tif')

                    # create OrthoIm  and orthorectify
                    logger.info(f'Homogenising {src_filename.name}')
                    start_ttl = datetime.datetime.now()
                    him = homonim.HomonimSrcSpace(src_filename, args.ref_file, win_size=args.win_size)
                    him.homogenise(homo_filename)

                    ttl_time = (datetime.datetime.now() - start_ttl)
                    logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')

                    if config['homogenisation']['build_ovw']:
                        start_ttl = datetime.datetime.now()
                        logger.info(f'Building overviews for {src_filename.name}')
                        him.build_ortho_overviews(homo_filename)
                        ttl_time = (datetime.datetime.now() - start_ttl)
                        logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')

                except Exception as ex:
                    # catch exceptions so that problem image(s) don't prevent processing of a batch
                    logger.error('Exception: ' + str(ex))

    except Exception as ex:
        logger.error('Exception: ' + str(ex))
        raise ex


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
