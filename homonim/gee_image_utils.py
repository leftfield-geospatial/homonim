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
from datetime import datetime, timedelta
from urllib import request
from rasterio.warp import transform_geom
import rasterio as rio
from shapely import geometry
import pathlib
import sys
import ee
from homonim import get_logger
logger = get_logger(__name__)

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

def cloud_mask_s2(image):
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
    bounds : geojson polygon of bounds in WGS84
    """
    with rio.open(filename) as im:
        src_bbox = geometry.box(*im.bounds)
        if (im.crs.linear_units == 'metre') and (expand > 0):
            expand_m = np.sqrt(src_bbox.area) * expand / 100.
            src_bbox = src_bbox.buffer(expand_m)    # expand the bounding box 2km beyond the extents (for visualisation)
        src_bbox_wgs84 = geometry.shape(transform_geom(im.crs, 'WGS84', src_bbox))
    return geometry.mapping(src_bbox_wgs84)

def download_link(link, filename):
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
    filename = filename.joinpath(filename.parent, filename.stem, '.zip')    # force to zip file

    if filename.exists():
        logger.warning()
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

class GeeRefImage:
    def __init__(self, collection_str='', scale=30, bands=None):
        """
        Class to assist the search and download of GEE Landsat 7/8 and Sentinel-2 surface reflectance imagery

        Parameters
        ----------
        im_collection_str : str
                            Name of a valid GEE image collection e.g. "LANDSAT/LC08/C01/T1_L2"
        """
        self.im_collection_str = collection_str
        self.im_scale = scale   # the native spatial resolution of the image in meters
        self.im_bands = bands   # subset of bands to retrieve

    @staticmethod
    def cloud_mask(image):
        """
        Mask cloudy pixels in image

        Parameters
        ----------
        image:  ee.Image
                the GEE image to mask

        Returns
        -------
        image:  ee.Image
                the masked image
        """
        raise NotImplementedError

    def get_image_link(self, bounds, date, min_images=1, crs=None, bands=None):
        """
        Search for and return a link to download a GEE reference image

        Parameters
        ----------
        bounds : geojson polygon
                 bounds of the source image(s) to be covered
        date : datetime.datetime
               capture date of source image
        min_images : int
                     minimum number of images to form composite of
        crs : str
              WKT or EPSG string specifying the CRS of reference image (default: WGS84)
        bands : list
                bands of reference image to download (default: all)
                e.g. 'bands': [{'id': 'B3'}, {'id': 'B2'}, {'id': 'B1'}, {'id': 'B4'}],

        Returns
        -------
        link :  str
                a download link for the image
        image : ee.Image
                a composite image matching the search criteria, or None if they could not be satisfied
        """
        if crs is None:
            crs = 'WGS84'
        # expand the date range in powers of 2 until we find an image
        start_date = date
        end_date  = date
        num_images = 0
        expand_pow = 0
        while (num_images < min_images) and (expand_pow <= 5):
            logger.info(f'Searching for images between {start_date.strftime("%Y-%m-%d")} and {end_date.strftime("%Y-%m-%d")}')
            im_collection = ee.ImageCollection(self.im_collection_str).\
                filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')).\
                filterBounds(bounds).map(self.cloud_mask)

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
            'scale': self.scale,
            'crs': crs,
            'fileFormat': 'GeoTIFF',
            'bands': bands,
            'filePerBand': False,
            'region': bounds})

        return link, image

class GeeLandsat7Image(GeeRefImage):
    def __init__(self):
        """
        Class to assist the search and download of GEE Landsat 7/8 and Sentinel-2 surface reflectance imagery

        Parameters
        ----------
        im_collection_str : str
                            Name of a valid GEE image collection e.g. "LANDSAT/LC08/C01/T1_L2"
        """

        GeeRefImage.__init__('LANDSAT/LT05/C02/T1_L2', scale=30, bands=)

