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
import os
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
import zipfile

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
    Download a GEE image download.zip file and extract to a geotiff file

    Parameters
    ----------
    link :  str
            link to GEE image to download
    filename :  str
                Path of the geotiff file to extract to.  Extension will be renamed to .tif if necessary.
    band_labels: list
                 Optional list of band labels to apply to the destination geotiff
    """
    logger.info(f'Opening link: {link}')

    file_link = request.urlopen(link)
    meta = file_link.info()
    file_size = int(meta['Content-Length'])
    logger.info(f'Download size: {file_size / (1024 ** 2):.2f} MB')

    tif_filename = pathlib.Path(filename)
    tif_filename = tif_filename.joinpath(tif_filename.parent, tif_filename.stem + '.tif')    # force to zip file
    zip_filename = tif_filename.parent.joinpath('gee_image_download.zip')

    for fn in (zip_filename, tif_filename):
        if fn.exists():
            logger.warning(f'{fn} exists, overwriting...')

    logger.info(f'Downloading to {zip_filename}')

    with open(zip_filename, 'wb') as f:
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

    # extract download.zip -> download.tif and rename to tif_filename
    logger.info(f'Extracting {zip_filename} to {zip_filename.parent}')
    with zipfile.ZipFile(zip_filename,"r") as zip_file:
        zip_file.extractall(zip_filename.parent)

    os.rename(zip_filename.parent.joinpath('download.tif'), tif_filename)


class EeRefImage:
    def __init__(self, source_image_filename, collection=''):
        self.collection = collection
        self._bounds, self._crs = get_image_bounds(source_image_filename, expand=10)
        collection_info = load_collection_info()
        if not collection in collection_info.keys():
            raise ValueError(f'Unknown collection: {collection}')
        self.collection_info = collection_info[collection]
        self._band_df = pd.DataFrame(self.collection_info['bands'])
        self.scale = int(self._band_df['res'].min())
        self._im_collection = None
        self._composite_im = None

    def _get_init_collection(self):
        return ee.ImageCollection(self.collection_info['ee_collection']).filterBounds(self._bounds)

    def search(self, date, min_images=1):
        im_collection_init = self._get_init_collection()

        # expand the date range in powers of 2 until we find an image
        start_date = date - timedelta(hours=12)
        end_date = date + timedelta(hours=12)
        num_images = 0
        expand_pow = 0
        while (num_images < min_images) and (expand_pow <= 6):
            logger.info(
                f'Searching for {self.collection_info["ee_collection"]} images between {start_date.strftime("%Y-%m-%d")} and {end_date.strftime("%Y-%m-%d")}')

            self._im_collection = im_collection_init.filterDate(start_date.strftime('%Y-%m-%d'),
                                                                end_date.strftime('%Y-%m-%d'))

            num_images = self._im_collection.size().getInfo()
            start_date -= timedelta(days=2 ** expand_pow)
            end_date += timedelta(days=2 ** expand_pow)
            expand_pow += 1

        if num_images == 0:
            logger.info(f'Could not find any images within {2 ** 6} days of {start_date.strftime("%Y-%m-%d")}')
            return None, None

        # print image dates
        logger.info(f'Found {num_images} images in {self.collection_info["ee_collection"]}:')
        im_list = self._im_collection.toList(num_images)
        for i in range(num_images):
            im_i = ee.Image(im_list.get(i))
            date_str = ee.Date(im_i.get("system:time_start")).format().getInfo()    # UTC/GMT
            index_str = im_i.get("system:index").getInfo()
            # cloud_portion = im_i.get('CLOUD_PORTION').getInfo()
            # SUN_AZIMUTH
            # SUN_ELEVATION
            # GEOMETRIC_RMSE_MODEL_Y & X
            # 'system:index'
            # 'CLOUD_PORTION' or similar
            logger.info(f'{index_str} captured on {date_str}')

        return self._im_collection

    def create_composite_image(self):
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')

        # form composite image
        self._composite_im = self._im_collection.mosaic()

        # set some metadata
        # image.set('system:time_start', ee.Date(date_range.get('min')))
        # image.set('system:time_end', ee.Date(date_range.get('max')))
        # image.set('n_components', num_images)

        return self._composite_im

    def _download_im(self, image, filename, is_composite=False):
        # create download link
        link = image.getDownloadURL({
            'scale': self.scale,
            'crs': self._crs,
            'fileFormat': 'GeoTIFF',
            'bands':  [],
            'filePerBand': False,
            'region': self._bounds})

        logger.info(f'Opening link: {link}')

        file_link = request.urlopen(link)
        meta = file_link.info()
        file_size = int(meta['Content-Length'])
        logger.info(f'Download size: {file_size / (1024 ** 2):.2f} MB')

        tif_filename = pathlib.Path(filename)
        tif_filename = tif_filename.joinpath(tif_filename.parent, tif_filename.stem + '.tif')    # force to zip file
        zip_filename = tif_filename.parent.joinpath('gee_image_download.zip')

        if zip_filename.exists():
            logger.warning(f'{zip_filename} exists, overwriting...')

        logger.info(f'Downloading to {zip_filename}')

        with open(zip_filename, 'wb') as f:
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

        # extract download.zip -> download.tif and rename to tif_filename
        logger.info(f'Extracting {zip_filename}')
        with zipfile.ZipFile(zip_filename, "r") as zip_file:
            zip_file.extractall(zip_filename.parent)

        if tif_filename.exists():
            logger.warning(f'{tif_filename} exists, overwriting...')
            os.remove(tif_filename)

        _tif_filename = zipfile.ZipFile(zip_filename, "r").namelist()[0]
        os.rename(zip_filename.parent.joinpath(_tif_filename), tif_filename)

        # write metadata to geotiff
        date_range = self._im_collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
        start_date_str = ee.Date(date_range.get('min')).format('YYYY-MM-dd').getInfo()
        end_date_str = ee.Date(date_range.get('max')).format('YYYY-MM-dd').getInfo()
        with rio.open(tif_filename, 'r+') as im:
            # TODO: add capture date range, and composite type?
            im.update_tags(EE_COLLECTION=self.collection_info['ee_collection'])
            if is_composite:
                im.update_tags(START_CAPTURE_DATE=start_date_str)
                im.update_tags(END_CAPTURE_DATE=end_date_str)
                im.update_tags(N_COMPONENT_IMS=self._im_collection.size().getInfo())

            for band_i, band_row in self._band_df.iterrows():
                im.update_tags(band_i+1, ID=band_row['id'])
                im.update_tags(band_i+1, NAME=band_row['name'])
                im.update_tags(band_i+1, ABBREV=band_row['abbrev'])

        return link

    def download_composite_im(self, filename):
        """
        Download a GEE image download.zip file and extract to a geotiff file

        Parameters
        ----------
        filename :  str
                    Path of the geotiff file to extract to.  Extension will be renamed to .tif if necessary.

        Returns
        -------
        link :  str
                link of the downloaded image
        """
        if self._composite_im is None:
            raise Exception('First generate a valid image collection with search(...) method')

        return self._download_im(self._composite_im, filename, is_composite=True)

    def _download_im_collection(self, filename):
        """
        Download all images in the collection

        Parameters
        ----------
        filename :  str
                    Base filename to use.

        Returns
        -------
        """
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')

        num_images = self._im_collection.size().getInfo()
        ee_im_list = self._im_collection.toList(num_images)
        filename = pathlib.Path(filename)

        for i in range(num_images):
            im_i = ee.Image(ee_im_list.get(i))
            im_date_i = ee.Date(im_i.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            filename_i = filename.parent.joinpath(f'{filename.stem}_{i}_{im_date_i}.tif')
            logger.info(f'Downloading {filename_i.stem}')
            self._download_im(im_i, filename_i, is_composite=False)



class Landsat8EeImage(EeRefImage):
    def __init__(self, source_image_filename, apply_mask=True):
        self.apply_mask = apply_mask
        EeRefImage.__init__(self, source_image_filename, apply_mask=apply_mask, collection='landsat8')

    def _add_validmask(self, image):
        # Notes
        # - QA_PIXEL The *conf bit pairs (8-9,10-11,12-13,14-15) will always be 1 or more, unless it is a fill pixel -
        # i.e. the fill bit 0 is set.  Values are higher where there are cloud, cloud shadow etc.  The water bit 7, is
        # seems to be set incorrectly quite often, but with the rest of the bits ok/sensible.
        # - SR_QA_AEROSOL bits 6-7 can have a value of 0, and this 0 can occur in e.g. an area of QA_PIXEL=cloud shadow,
        # NB this is not band 9 as on GEE site, but band 8.
        # - The behaviour of updateMask in combination with ImageCollection qualityMosaic (and perhaps median and mosaic
        # ) is weird: updateMask always masks bands added with addBands, but only masks the original SR etc bands after
        # a call to qualityMosaic (or perhaps median etc)


        # create a mask of valid (non cloud, shadow and aerosol) pixels
        # bits 1-4 of QA_PIXEL are dilated cloud, cirrus, cloud & cloud shadow, respectively
        qa_pixel_bitmask = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
        qa_pixel = image.select('QA_PIXEL')
        cloud_mask = qa_pixel.bitwiseAnd(qa_pixel_bitmask).eq(0)

        # bits 6-7 of SR_QA_AEROSOL, are aerosol level where 3 = high, 2=medium, 1=low
        sr_qa_aerosol_bitmask = (1 << 6) | (1 << 7)
        sr_qa_aerosol = image.select('SR_QA_AEROSOL')
        # aerosol_prob = sr_qa_aerosol.bitwiseAnd(sr_qa_aerosol_bitmask).rightShift(6)
        aerosol_prob = sr_qa_aerosol.rightShift(6).bitwiseAnd(3)
        aerosol_mask = aerosol_prob.lt(3)

        # TODO: figure out why we need this weird rename here
        # TODO: is aerosol_mask helpful in general, it looks suspect for GEF NGI ims
        valid_mask = cloud_mask.And(aerosol_mask).rename('VALID_PORTION')
        valid_portion = valid_mask.Not().reduceRegion(reducer='mean', geometry=self._bounds, scale=self.scale)

        # create a pixel quallity score (higher is better)
        cloud_conf = qa_pixel.rightShift(8).bitwiseAnd(3)
        cloud_shadow_conf = qa_pixel.rightShift(10).bitwiseAnd(3)
        cirrus_conf = qa_pixel.rightShift(14).bitwiseAnd(3)
        q_score = cloud_conf.add(cloud_shadow_conf).add(cirrus_conf).add(aerosol_prob).multiply(-1).rename('Q_SCORE')

        if False:
            image = image.addBands(sr_qa_aerosol)
            image = image.addBands(cloud_conf)
            image = image.addBands(cloud_shadow_conf)
            image = image.addBands(cirrus_conf)
            image = image.addBands(aerosol_prob)

        if self.apply_mask:
            return image.set(valid_portion).addBands(q_score).updateMask(valid_mask)
        else:
            return image.set(valid_portion).addBands(q_score)

    def _get_init_collection(self):
        return ee.ImageCollection(self.collection_info['ee_collection']).\
            filterBounds(self._bounds).\
            map(self._add_validmask).\
            filter(ee.Filter.lt('VALID_PORTION', 0.05))

    def create_composite_image(self):
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')

        # form composite image
        # self._composite_im = self._im_collection.mosaic()
        self._composite_im = self._im_collection.qualityMosaic('Q_SCORE')

        # set some metadata
        # image.set('system:time_start', ee.Date(date_range.get('min')))
        # image.set('system:time_end', ee.Date(date_range.get('max')))
        # image.set('n_components', num_images)
        # qualityMosaic(qualityBand)
        return self._composite_im

class Landsat7EeImage(EeRefImage):
    def __init__(self, source_image_filename):
        EeRefImage.__init__(self, source_image_filename, collection='landsat7')

    def _add_cloudmask(self, image):
        mask_bit = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
        qa_pixel = image.select('QA_PIXEL')
        qa_pixel = qa_pixel.rename('CLOUD_PORTION')     # ??
        # qa.bitwiseAnd(mask_bit).eq(0)
        cloud_mask = qa_pixel.bitwiseAnd(mask_bit).eq(0)
        cloud_portion = cloud_mask.Not().reduceRegion(reducer='mean', geometry=self._bounds, scale=self.scale)
        return image.set(cloud_portion).updateMask(cloud_mask)

    def _get_init_collection(self):
        return ee.ImageCollection(self.collection_info['ee_collection']).\
            filterBounds(self._bounds).\
            map(self._add_cloudmask).\
            filter(ee.Filter.lt('CLOUD_PORTION', 0.05))

    def create_composite_image(self):
        if self._im_collection is None:
            raise Exception('First generate a valid image collection with search(...) method')

        # form composite image
        self._composite_im = self._im_collection.mosaic()

        # set some metadata
        # image.set('system:time_start', ee.Date(date_range.get('min')))
        # image.set('system:time_end', ee.Date(date_range.get('max')))
        # image.set('n_components', num_images)

        return self._composite_im


class Sentinel2EeImage(EeRefImage):
    def __init__(self, source_image_filename, collection='sentinel2_toa'):
        EeRefImage.__init__(self, source_image_filename, collection=collection)

    def _add_cloudmask(self, image):
        bit_mask = (1 << 11) | (1 << 10)
        qa = image.select('QA60')
        # qa.bitwiseAnd(mask_bit).eq(0)
        cloud_mask = qa.bitwiseAnd(bit_mask).eq(0)
        cloud_portion = cloud_mask.Not().reduceRegion(reducer='mean', geometry=self._bounds, scale=self.scale)
        return image.set('CLOUD_PORTION', cloud_portion).updateMask(cloud_mask)

    def _get_init_collection(self):
        return ee.ImageCollection(self.collection_info['ee_collection']).\
            filterBounds(self._bounds).\
            map(self._add_cloudmask).\
            filter(ee.Filter.lt('CLOUD_PORTION', 0.05))
