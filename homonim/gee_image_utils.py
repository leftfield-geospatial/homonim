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
from datetime import datetime, timedelta
from urllib import request
from rasterio.warp import transform_geom
import rasterio as rio
from shapely import geometry
import pathlib
import sys
import ee
from homonim import get_logger, root_path
import pandas as pd
import numpy as np

logger = get_logger(__name__)

def load_collection_info():
    """
    Loads the satellite band etc information from json file into a dict
    """
    with open(root_path.joinpath('data/inputs/satellite_info.json')) as f:
        satellite_info = json.load(f)
    return satellite_info

def cloud_mask_landsat_c2(image):
    """
    Cloud mask collection 2 landsat 7/8 image
    i.e. images from ee.ImageCollection("LANDSAT/LE07/C02/T1_L2") and ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

    Parameters
    ----------
    image:  ee.Image
            the image to mask

    Returns
    -------
    ee.Image
    The masked image
    """
    # Bits 1-4 are dilated cloud, cirrus, cloud and cloud shadow respectively
    mask_bit = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
    qa = image.select('QA_PIXEL')
    return image.updateMask(qa.bitwiseAnd(mask_bit).eq(0))

def cloud_mask_landsat_c1(image):
    """
    Cloud mask collection 1 landsat 7/8 image
    i.e. images from ee.ImageCollection("LANDSAT/LE07/C01/T1_L2") and ee.ImageCollection("LANDSAT/LC08/C01/T1_L2")

    Parameters
    ----------
    image:  ee.Image
            the GEE image to mask

    Returns
    -------
    ee.Image
    The masked image
    """
    # Bits 3 and 5 are cloud and cloud shadow respectively
    mask_bit = (1 << 5) | (1 << 3)
    qa = image.select('pixel_qa')
    return image.updateMask(qa.bitwiseAnd(mask_bit).eq(0))

def cloud_mask_sentinel2(image):
    """
    Cloud mask sentinel-2 TOA and SR image
    i.e. (ee.ImageCollection("COPERNICUS/S2") and ee.ImageCollection("COPERNICUS/S2_SR"))

    Parameters
    ----------
    image:  ee.Image
            the GEE image to mask

    Returns
    -------
    ee.Image
    The masked image
    """
    # TODO: add cloud shadow mask from SCL band
    qa = image.select('QA60');
    # bits 10 and 11 are opaque and cirrus clouds respectively
    bit_mask = (1 << 11) | (1 << 10)
    return image.updateMask(qa.bitwiseAnd(bit_mask).eq(0).focal_min(10))

def cloud_mask_modis(image):
    """
    Dummy function to cloud mask MODIS NBAR data
    i.e. (ee.ImageCollection("MODIS/006/MCD43A4")))

    Parameters
    ----------
    image:  ee.Image
            the GEE image to mask

    Returns
    -------
    ee.Image
    The masked image
    """
    return image

def get_image_bounds(filename, expand=10):
    """
    Get the WGS84 geojson bounds of an image

    Parameters
    ----------
    filename :  str
                name of the image file whose bounds to find
    expand :  int
              percentage (0-100) by which to dilate the bounds (default: 10)

    Returns
    -------
    bounds : geojson
             polygon of bounds in WGS84
    crs: str
         WKT CRS of image file
    """
    with rio.open(filename) as im:
        src_bbox = geometry.box(*im.bounds)
        if (im.crs.linear_units == 'metre') and (expand > 0):
            expand_m = np.sqrt(src_bbox.area) * expand / 100.
            src_bbox = src_bbox.buffer(expand_m)    # expand the bounding box 2km beyond the extents (for visualisation)
        src_bbox_wgs84 = geometry.shape(transform_geom(im.crs, 'WGS84', src_bbox))
    return geometry.mapping(src_bbox_wgs84), im.crs.to_wkt()


def search_image_collection(collection, bounds, date, cloud_mask=None, min_images=1, crs=None, bands=[]):
    """
    Search a GEE image collection, returning an image download link where possible

    Parameters
    ----------
    collection : str
                Name of GEE image collection to search ('landsat7', 'landsat8', 'sentinel2_toa', 'sentinel2_sr',
                'modis') (default: landsat8)
    cloud_mask : function
                 function to cloud mask ee.Image (default: None = select automatically)
    bounds : geojson polygon
             bounds of the source image(s) to be covered
    date : datetime.datetime
           capture date of source image
    min_images : int
                 minimum number of images allowed for composite formation (default=1)
    crs : str
          WKT or EPSG string specifying the CRS of download image (default: WGS84)
    bands : list
            list of dicts specifying bands to download (default: None = all)
            e.g. bands=[{'id': 'B3'}, {'id': 'B2'}, {'id': 'B1'}, {'id': 'B4'}]

    Returns
    -------
    link :  str
            a download link for the image
    image : ee.Image
            a composite image matching the search criteria, or None if they could not be satisfied
    """

    # parse arguments
    if cloud_mask is None:
        if (collection == 'landsat7') or (collection == 'landsat8'):
            cloud_mask = cloud_mask_landsat_c2
        elif (collection == 'sentinel2_toa') or (collection == 'sentinel2_sr'):
            cloud_mask = cloud_mask_sentinel2()
        elif (collection == 'modis'):
            cloud_mask = None  # MODIS NBAR has already been cloud masked
        else:
            raise ValueError(f'Unknown GEE image collection: {collection}')
    collection_info = load_collection_info()
    if not collection in collection_info.keys():
        raise ValueError(f'Unknown collection: {collection}')
    collection_info = collection_info[collection]
    band_df = pd.DataFrame(collection_info['bands'])

    if crs is None:
        crs = 'WGS84'

    if bands is None:
        bands = []

    # expand the date range in powers of 2 until we find an image
    start_date = date - timedelta(hours=12)
    end_date  = date + timedelta(hours=12)
    num_images = 0
    expand_pow = 0
    while (num_images < min_images) and (expand_pow <= 5):
        logger.info(f'Searching for images between {start_date.strftime("%Y-%m-%d")} and {end_date.strftime("%Y-%m-%d")}')
        im_collection = ee.ImageCollection(collection_info['ee_collection']).\
            filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')).\
            filterBounds(bounds).map(cloud_mask)

        num_images = im_collection.size().getInfo()
        start_date -= timedelta(days=2**expand_pow)
        end_date += timedelta(days=2**expand_pow)
        expand_pow += 1

    if num_images == 0:
        logger.info(f'Could not find any images within {2**5} days of {start_date.strftime("%Y-%m-%d")}')
        return None

    logger.info(f'Found {num_images} images')

    # form composite image and get download link
    image = im_collection.median()
    link = image.getDownloadURL({
        'scale': int(band_df['res'].min()),
        'crs': crs,
        'fileFormat': 'GeoTIFF',
        'bands': bands,
        'filePerBand': False,
        'region': bounds})

    return link, image

def download_image(link, filename):
    """
    Download a GEE image link to a zip file

    Parameters
    ----------
    link :  str
            link to GEE image to download
    filename :  str
                Path of the zip file to download to.  Extension will be renamed to .zip if necessary.
    """
    logger.info(f'Opening link: {link}')

    file_link = request.urlopen(link)
    meta = file_link.info()
    file_size = int(meta['Content-Length'])
    logger.info(f'Download size: {file_size / (1024 ** 2):.2f} MB')

    filename = pathlib.Path(filename)
    filename = filename.joinpath(filename.parent, filename.stem + '.zip')    # force to zip file

    if filename.exists():
        logger.warning(f'{filename} exists, overwriting...')
    logger.info(f'Downloading to {filename}')

    with open(filename, 'wb') as f:
        file_size_dl = 0
        block_size = 8192
        while (file_size_dl <= file_size):
            buffer = file_link.read(block_size)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)

            progress = (file_size_dl / file_size)
            sys.stdout.write('\r')
            sys.stdout.write('[%-50s] %d%%' % ('=' * int(50 * progress), 100 * progress))
            sys.stdout.flush()
        sys.stdout.write('\n')
