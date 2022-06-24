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
import glob
import pathlib

import numpy as np
import rasterio as rio
from rasterio.warp import Resampling
"""
Downsample an image by an integer factor, keeping the grid alignment and bounds.  
Scale to uint8 and compress with DEFLATE.

Intended to convert NGI images into a small enough file to keep in git repo and use for unit tests.  
"""

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Downsample an image by an integer factor, keeping the grid alignment and bounds. '
        'Intended for NGI unrectified imagery.'
    )
    parser.add_argument(
        "src_im_wildcard", help="source image wildcard pattern or directory (e.g. '.' or '*_CMP.TIF')", type=str
    )
    parser.add_argument(
        "-s", "--scale", help="scale factor to downsample by (default=10)", default=10, choices=range(1, 1000), type=int
    )
    return parser.parse_args()


def process_args(args):
    # check files exist
    if len(glob.glob(args.src_im_wildcard)) == 0:
        if len(glob.glob(args.src_im_wildcard + '*.tif')) == 0:
            raise Exception(f'Could not find any files matching {args.src_im_wildcard}')
        else:
            args.src_im_wildcard = args.src_im_wildcard + '*.tif'

    return


def main(args):
    # parse the command line
    process_args(args)
    ds_factor = args.scale
    bands = [0, 1, 2]  # use first 3 bands

    src_im_list = glob.glob(args.src_im_wildcard)
    print(f'Batch downsampling {len(src_im_list)} file(s) matching {args.src_im_wildcard} by factor of {ds_factor}')
    for src_i, src_im_filename in enumerate(src_im_list):
        src_im_filename = pathlib.Path(src_im_filename)
        ds_im_filename = src_im_filename.parent.joinpath(src_im_filename.stem + '_DS' + src_im_filename.suffix)

        print(f'Processing {src_im_filename.stem} - file {src_i + 1} of {len(src_im_list)}:')
        try:
            with rio.Env():
                with rio.open(src_im_filename, 'r', num_threads='all_cpus') as src_im:
                    # construct output profile from src profile
                    ds_profile = src_im.profile.copy()
                    ds_transform = src_im.transform
                    ds_transform = ds_transform * rio.Affine.scale(ds_factor)
                    ds_shape = np.int32(np.array(src_im.shape) / ds_factor)

                    ds_profile.update(
                        count=3, nodata=0, dtype='uint8', compress='deflate', tiled=True, blockxsize=256,
                        blockysize=256, transform=ds_transform, width=ds_shape[1], height=ds_shape[0],
                        num_threads='all_cpus', interleave='band', photometric="MINISBLACK"
                    )

                    # read and downsample
                    src_array = src_im.read(out_shape=(4, ds_shape[0], ds_shape[1]), resampling=Resampling.average)

                    # scale, clip and cast to uint8
                    ds_array = np.clip(src_array[bands, :, :] * (255 / 3000), 0, 255).astype('uint8')

                    with rio.open(ds_im_filename, 'w', **ds_profile) as ds_im:
                        ds_im.write(ds_array)

        except Exception as ex:
            print(f'Exception: {str(ex)}')


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
