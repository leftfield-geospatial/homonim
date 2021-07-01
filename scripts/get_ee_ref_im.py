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
from datetime import datetime, timedelta
import dateutil
import os
import pathlib
import ee
from shapely import geometry
import fiona
import rasterio as rio
from rasterio.warp import reproject, Resampling, transform_geom, transform_bounds
from homonim import root_path, get_logger

# conda install -c conda-forge earthengine-api
# conda install -c conda-forge folium
# rio bounds NGI_3322A_2010_Subsection_Source.vrt > NGI_3322A_2010_Subsection_Source_Bounds.geojson


# src_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3322A_2010_HotSpotSeamLineEg\Source\NGI_3322A_2010_HotSpotSeamLineEg_Source.vrt")    #2010-01-22 - 2010-02-01
# src_date = '2010-01-26'
# src_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3323D_2015_GefSite\Source\NGI_3323DA_2015_GefSite_Source.vrt")    #2010-01-22 - 2010-02-01
# src_date = '2015-05-08'
##
# TODO: update this to an "intelligent" version that chooses the best of L8, L7 etc based on image date,
#  then searches and expands the date range until it finds something, and chooses appropriate cloud masking, perhaps reducing criteria etc
# TODO: write cloud masking and band matching backend based on my GEE code

logger = get_logger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Download reference Landsat and Sentinel imagery from Google Earth Engine (GEE).\n')
    parser.add_argument('extent_file', help='path specifying source image/vector file whose spatial extent should be covered', type=str,
                        metavar='extent_file', nargs='+')
    parser.add_argument('-d', '--date', help='capture date of source image e.g. \'2015-01-28\' '
                                             '(default: use creation time of the <extent> file)', type=str)
    parser.add_argument('-s', '--satellite',
                        help='name of the data collection: \'L5\'=Landsat5, \'L7\'=LANDSAT/LE07/C02/T1_L2, \'L8\'=LANDSAT/LC08/C02/T1_L2, '
                             '\'S2\'=Sentinel2, ...=any name of a valid GEE image collection (default: \'L8\')',
                        choices=['L5', 'L7', 'L8', 'S2'], type=str)
    parser.add_argument('-od', '--output_dir',
                        help='save downloaded image in this directory (default: save to current directory)',
                        type=str)
    return parser.parse_args()

# ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')  # Landsat 8 SR
# ee.ImageCollection("LANDSAT/LE07/C01/T1_SR")  # Landsat 7 SR
# ee.ImageCollection('COPERNICUS/S2')           # Sentinel-2 SR
def cloud_mask_landsat(image):
    # adapted from https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C01_T1_SR?hl=en
    qa = image.select('pixel_qa')
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    mask_bit = (1 << 5) | (1 << 3)
    qa = image.select('pixel_qa')
    return image.updateMask(qa.bitwiseAnd(mask_bit).eq(0))

    # # If the cloud bit (5) is set and the cloud confidence (7) is high
    # # or the cloud shadow bit is set (3), then it's a bad pixel.
    # cloud = qa.bitwiseAnd(1 << 5) and (qa.bitwiseAnd(1 << 7)) or (qa.bitwiseAnd(1 << 3))
    # # Remove edge pixels that don't occur in all bands
    # mask2 = image.mask().reduce(ee.Reducer.min())
    # return image.updateMask(not cloud).updateMask(mask2)

def s2_cloud_mask(image):
  qa = image.select('QA60');
  bit_mask = (1 << 11) | (1 << 10)
  return image.updateMask(qa.bitwiseAnd(bit_mask).eq(0).focal_min(10))


def main(args):
    """
    Download reference Landsat and Sentinel imagery from Google Earth Engine (GEE)

    Parameters
    ----------
    args :  ArgumentParser.parse_args()
            Run `python get_gee_ref_im.py -h` to see help on arguments
    """

    ## check arguments
    for extent_file_spec in args.extent_file:
        extent_file_path = pathlib.Path(extent_file_spec)
        if len(list(extent_file_path.parent.glob(extent_file_path.name))) == 0:
            raise Exception(f'Could not find any files matching {extent_file_spec}')

    if pathlib.Path(args.output_file).exists():
        raise Exception(f'Reference file {args.ref_file} exists already')

    src_date = dateutil.parse(args.date)

    ## initialise GEE
    ee.Authenticate()
    ee.Initialize()

    if args.satellite == 'L5':
        im_collection = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').map(cloud_mask_landsat)    #LANDSAT/LT05/C01/T1_SR
    elif args.satellite == 'L7':
        im_collection = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").map(cloud_mask_landsat)    #.filter(ee.Filter.lt('CLOUD_COVER', 10))    #LANDSAT/LE07/C01/T1_SR LANDSAT/LE07/C02/T1_L2
        # im_collection = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR").map(cloud_mask_landsat)
    elif args.satellite == 'L8':
        im_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').map(cloud_mask_landsat)    #LANDSAT/LC08/C02/T1_L2 #LANDSAT/LC08/C01/T1_SR
    elif args.satellite == 'S2':
        # im_collection = ee.ImageCollection("COPERNICUS/S2_SR").map(s2_cloud_mask)
        im_collection = ee.ImageCollection("COPERNICUS/S2").map(s2_cloud_mask) #TOA
    else:
        im_collection = ee.ImageCollection(args.satellite)

    # for extent_file_spec in args.extent_file:
    #     extent_file_path = pathlib.Path(extent_file_spec)
    #     if len(list(extent_file_path.parent.glob(extent_file_path.name))) == 0:
    #         raise Exception(f'Could not find any files matching {extent_file_spec}')

    with rio.open(args.extent) as src_im:
        src_bbox_m = geometry.box(*src_im.bounds)
        src_bbox_m = src_bbox_m.buffer(2000)
        src_bbox = geometry.shape(transform_geom(src_im.crs, 'WGS84', src_bbox_m))

    start_date = src_date - timedelta(days=21)
    end_date = src_date + timedelta(days=21)

    images = im_collection.filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')).\
        filterBounds(geometry.mapping(src_bbox))

    logger.info(f'Found {images.size().getInfo()} images')

    image = images.median()   #.select(['B4', 'B3', 'B2', 'B1'])

    link = image.getDownloadURL({
        'scale': 30,
        'crs': src_im.crs.to_wkt(),
        'fileFormat': 'GeoTIFF',
        'bands': [{'id': 'B3'}, {'id': 'B2'}, {'id': 'B1'}, {'id': 'B4'}],
        'filePerBand': False,
        'region': geometry.mapping(src_bbox)})

    print(f'Downloading from {link}')

# print(l7_sr_images.getInfo())

