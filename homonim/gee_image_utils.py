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
import pathlib
import sys
from datetime import timedelta
from urllib import request

import ee
import numpy as np
import pandas as pd
import rasterio as rio
from homonim import get_logger, root_path
from rasterio.warp import transform_geom
from shapely import geometry

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
    # TODO .focal_min(10) ?
    # mask_bit = (1 << 3) | (1 << 4)
    qa = image.select('QA_PIXEL')
    return image.updateMask(qa.bitwiseAnd(mask_bit).eq(0).focal_min(200, units='meters'))

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
    qa = image.select('QA60')
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


def search_image_collection(collection, bounds, date, cloud_mask=True, min_images=1, crs=None, bands=[]):
    """
    Search a GEE image collection, returning an image download link where possible

    Parameters
    ----------
    collection : str
                Name of GEE image collection to search ('landsat7', 'landsat8', 'sentinel2_toa', 'sentinel2_sr',
                'modis') (default: landsat8)
    cloud_mask : bool
                 do per-pixel cloud masking for compositing (default: True)
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
    # TODO: add .filterMetadata('CLOUD_COVER','less_than',50); for landsat and similar for s2 and perhaps .filterMetadata("GEOMETRIC_RMSE_MODEL", "less_than", 10)
    #  note that CLOUD_COVER may be for whole granule and not just the bounds filtered image.

    # parse arguments
    collection_info = load_collection_info()
    if not collection in collection_info.keys():
        raise ValueError(f'Unknown collection: {collection}')
    collection_info = collection_info[collection]
    band_df = pd.DataFrame(collection_info['bands'])

    if crs is None:
        crs = 'WGS84'

    if bands is None:
        bands = []

    im_collection_init = ee.ImageCollection(collection_info['ee_collection']).filterBounds(bounds)
    if (collection == 'landsat7') or (collection == 'landsat8'):
        # im_collection_init = im_collection_init.filterMetadata('CLOUD_COVER', 'less_than', 10)   # rather require
        if cloud_mask:
            im_collection_init = im_collection_init.map(cloud_mask_landsat_c2)
    elif (collection == 'sentinel2_toa') or (collection == 'sentinel2_sr'):
        # im_collection_init = im_collection_init.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 10)
        if cloud_mask:
            im_collection_init = im_collection_init.map(cloud_mask_sentinel2)
        # note that CLOUDY_PIXEL_PERCENTAGE is for whole granule, not just this sub-image, a granule is 100 x 100 km
    elif (collection == 'modis'):
        if cloud_mask:
            logger.info('MODIS NBAR does not require cloud masking, setting cloud_mask=False')
            cloud_mask = False
    else:
        raise ValueError(f'Unknown GEE image collection: {collection}')

    # expand the date range in powers of 2 until we find an image
    start_date = date - timedelta(hours=12)
    end_date  = date + timedelta(hours=12)
    num_images = 0
    expand_pow = 0
    while (num_images < min_images) and (expand_pow <= 5):
        logger.info(f'Searching for {collection_info["ee_collection"]} images between {start_date.strftime("%Y-%m-%d")} and {end_date.strftime("%Y-%m-%d")}')

        im_collection = im_collection_init.filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        num_images = im_collection.size().getInfo()
        start_date -= timedelta(days=2**expand_pow)
        end_date += timedelta(days=2**expand_pow)
        expand_pow += 1

    if num_images == 0:
        logger.info(f'Could not find any images within {2**5} days of {start_date.strftime("%Y-%m-%d")}')
        return None, None

    logger.info(f'Found {num_images} images')

    # form composite image and get download link
    image = im_collection.median()
    image = image.select(band_df['id'].tolist(), band_df['abbrev'].tolist())

    # TODO: can we rename bands here according to satellite_info? I think we need to use raterio on downloaded file to rename them https://rasterio.readthedocs.io/en/latest/topics/tags.html
    link = image.getDownloadURL({
        'scale': int(band_df['res'].min()),
        'crs': crs,
        'fileFormat': 'GeoTIFF',
        'bands': band_df['abbrev'].tolist(),
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



class EeRefImage:
    def __init__(self, source_image_filename, collection=''):
        self.collection = collection
        self.bounds, self.crs = get_image_bounds(source_image_filename, expand=10)
        collection_info = load_collection_info()
        if not collection in collection_info.keys():
            raise ValueError(f'Unknown collection: {collection}')
        self.collection_info = collection_info[collection]
        self.band_df = pd.DataFrame(self.collection_info['bands'])
        self.scale = int(self.band_df['res'].min())

    def _get_init_collection(self):
        return ee.ImageCollection(self.collection_info['ee_collection']).filterBounds(self.bounds)

    def search(self, date, min_images=1):
        im_collection_init = self._get_init_collection()

        # expand the date range in powers of 2 until we find an image
        start_date = date - timedelta(hours=12)
        end_date = date + timedelta(hours=12)
        num_images = 0
        expand_pow = 0
        while (num_images < min_images) and (expand_pow <= 5):
            logger.info(
                f'Searching for {self.collection_info["ee_collection"]} images between {start_date.strftime("%Y-%m-%d")} and {end_date.strftime("%Y-%m-%d")}')

            im_collection = im_collection_init.filterDate(start_date.strftime('%Y-%m-%d'),
                                                          end_date.strftime('%Y-%m-%d'))

            num_images = im_collection.size().getInfo()
            start_date -= timedelta(days=2 ** expand_pow)
            end_date += timedelta(days=2 ** expand_pow)
            expand_pow += 1

        if num_images == 0:
            logger.info(f'Could not find any images within {2 ** 5} days of {start_date.strftime("%Y-%m-%d")}')
            return None, None

        logger.info(f'Found {num_images} images')

        # form composite image and get download link
        image = im_collection.median()
        # image.select(self.band_df['id'].tolist(), self.band_df['abbrev'].tolist())
        # TODO: can we rename bands here according to satellite_info?
        link = image.getDownloadURL({
            'scale': self.scale,
            'crs': self.crs,
            'fileFormat': 'GeoTIFF',
            'bands':  [],
            'filePerBand': False,
            'region': self.bounds})

        return link, image

class EeLandsatRefImage(EeRefImage):
    def __init__(self, source_image_filename, collection='landsat8'):
        EeRefImage.__init__(self, source_image_filename, collection=collection)

    def _add_cloudscore(self, image):
        mask_bit = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
        qa = image.select('QA_PIXEL')
        qa = qa.rename('CLOUD_PORTION')     # ??
        cloud_mask = qa.bitwiseAnd(mask_bit).neq(0)
        cloud_portion = cloud_mask.reduceRegion(reducer='mean', geometry=self.bounds, scale=self.scale)
        return image.set(cloud_portion)

    def _get_init_collection(self):
        return ee.ImageCollection(self.collection_info['ee_collection']).\
            filterBounds(self.bounds).\
            map(self._add_cloudscore).\
            filter(ee.Filter.lt('CLOUD_PORTION', 0.05))

class EeSentinelRefImage(EeRefImage):
    def __init__(self, source_image_filename, collection='sentinel2_toa'):
        EeRefImage.__init__(self, source_image_filename, collection=collection)

    def _add_cloudscore(self, image):
        bit_mask = (1 << 11) | (1 << 10)
        qa = image.select('QA60')
        cloud_mask = qa.bitwiseAnd(bit_mask).neq(0)
        cloud_portion = cloud_mask.reduceRegion(reducer='mean', geometry=self.bounds, scale=self.scale)
        return image.set('CLOUD_PORTION', cloud_portion)

    def _get_init_collection(self):
        return ee.ImageCollection(self.collection_info['ee_collection']).\
            filterBounds(self.bounds).\
            map(self._add_cloudscore).\
            filter(ee.Filter.lt('CLOUD_PORTION', 0.05))
