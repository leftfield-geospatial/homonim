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
import click

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

if False:
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Radiometrically homogenise an image by fusion with satellite data.')
        parser.add_argument("src_file", help="path or wildcard specifying the source image file(s)", type=str,
                            metavar='src_file', nargs='+')
        parser.add_argument("ref_file", help="path to the satellite reference image", type=str)
        parser.add_argument("-m", "--method", help="homogenisation method [default: gain-only]", nargs=1, type=str)
        parser.add_argument("-w", "--win-size", help="sliding window width and height (e.g. -w 3 3) [default: 3 x 3]", nargs=2, type=int)
        parser.add_argument("-od", "--output-dir",
                            help="write homogenised image(s) to this directory [default: write to source directory]",
                            type=str)
        # parser.add_argument("-rc", "--readconf",
        #                     help="read custom config from this path (default: use config.yaml in homonim root)",
        #                     type=str)
        # parser.add_argument("-wc", "--writeconf", help="write default config to this path and exit", type=str)
        parser.add_argument("-v", "--verbosity", choices=[1, 2, 3, 4],
                            help="logging level: 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR (default: 2)", type=int)
        return parser.parse_args()


    def _process_args(args):
        # set logging level
        if args.verbosity is not None:
            logger.setLevel(10 * args.verbosity)
            homonim.logger.setLevel(10 * args.verbosity)

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
                raise Exception(f'Output directory {args.homo_dir} is not a valid directory')
            if not homo_dir.exists():
                logger.warning(f'Creating output directory {args.homo_dir}')
                os.mkdir(str(homo_dir))

        if args.win_size is None:
            args.win_size = [3, 3]


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
            _process_args(args)

            # loop through image file(s) or wildcard(s), or combinations thereof
            for src_file_spec in args.src_file:
                src_file_path = pathlib.Path(src_file_spec)
                for src_filename in src_file_path.parent.glob(src_file_path.name):
                    try:
                        # set homogenised filename
                        if args.homo_dir is not None:
                            homo_filename = pathlib.Path(args.homo_dir).joinpath(src_filename.stem + f'_HOMO_REF_w{args.win_size[0]}{args.win_size[1]}.tif')
                        else:
                            homo_filename = src_filename.parent.joinpath(src_filename.stem + f'_HOMO_REF_w{args.win_size[0]}{args.win_size[1]}.tif')

                        # create OrthoIm  and orthorectify
                        logger.info(f'Homogenising {src_filename.name}')
                        start_ttl = datetime.datetime.now()
                        him = homonim.HomonimRefSpace(src_filename, args.ref_file, win_size=args.win_size)
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

    def main_entry():
        args = parse_arguments()
        main(args)

    if __name__ == "__main__":
        main_entry()



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
    required=True
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
    type=click.Choice(['GAIN_ONLY', 'GAIN_OFFSET', 'GAIN_BANDOFFSET'], case_sensitive=False),
    help="homogenisation method",
    default='GAIN_ONLY',
    show_default=True,
)
@click.option(
    "-od",
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    help="directory to write homogenised image(s) in [default: use src-file directory]",
    required=False,
)
def cli(src_file=None, ref_file=None, win_size=(3, 3), method="gain_only", output_dir=None):
    """Radiometrically homogenise image(s) by fusion with reference satellite data"""

    # check src_file points to exisiting file(s)
    for src_file_spec in src_file:
        src_file_path = pathlib.Path(src_file_spec)
        if len(list(src_file_path.parent.glob(src_file_path.name))) == 0:
            raise Exception(f'Could not find any source image(s) matching {src_file_spec}')
        for src_filename in src_file_path.parent.glob(src_file_path.name):
            try:
                # set homogenised filename
                if output_dir is not None:
                    homo_root = pathlib.Path(output_dir)
                else:
                    homo_root = src_filename.parent
                homo_filename = homo_root.joinpath(src_filename.stem +
                                                   f'_HOMO_REF_m{method.upper()}_w{win_size[0]}{win_size[1]}.tif')

                logger.info(f'Homogenising {src_filename.name}')
                start_ttl = datetime.datetime.now()
                him = homonim.HomonimRefSpace(src_filename, ref_file, win_size=win_size)
                him.homogenise(homo_filename)
                ttl_time = (datetime.datetime.now() - start_ttl)
                logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')

                if True:
                    start_ttl = datetime.datetime.now()
                    logger.info(f'Building overviews for {homo_filename.name}')
                    him.build_ortho_overviews(homo_filename)
                    ttl_time = (datetime.datetime.now() - start_ttl)
                    logger.info(f'Completed in {ttl_time.total_seconds():.2f} secs')

            except Exception as ex:
                # catch exceptions so that problem image(s) don't prevent processing of a batch
                logger.error('Exception: ' + str(ex))

