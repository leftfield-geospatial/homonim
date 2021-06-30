import pathlib
import ee
from shapely import geometry
import fiona
import rasterio as rio
from rasterio.warp import reproject, Resampling, transform_geom, transform_bounds
# conda install -c conda-forge earthengine-api
# conda install -c conda-forge folium
# rio bounds NGI_3322A_2010_Subsection_Source.vrt > NGI_3322A_2010_Subsection_Source_Bounds.geojson


src_filename = pathlib.Path(r"V:\Data\HomonimEgs\NGI_3322A_2010_Subsection\Source\NGI_3322A_2010_Subsection_Source.vrt")    #2010-01-22 - 2010-02-01

ee.Authenticate()
ee.Initialize()

with rio.open(src_filename) as src_im:
    src_bbox_m = geometry.box(*src_im.bounds)
    src_bbox_m = src_bbox_m.buffer(2000)
    src_bbox = geometry.shape(transform_geom(src_im.crs, 'WGS84', src_bbox_m))

l7_sr_images = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR").\
    filterDate('2010-01-02', '2010-02-21').\
    filter(ee.Filter.lt('CLOUD_COVER', 10)).\
    filterBounds(geometry.mapping(src_bbox))
print(f'Number of images: {l7_sr_images.size().getInfo()}')

image = l7_sr_images.median()   #.select(['B4', 'B3', 'B2', 'B1'])

link = image.getDownloadURL({
    'scale': 30,
    'crs': src_im.crs.to_wkt(),
    'fileFormat': 'GeoTIFF',
    'bands': [{'id': 'B3'}, {'id': 'B2'}, {'id': 'B1'}, {'id': 'B4'}],
    'filePerBand': False,
    'region': geometry.mapping(src_bbox)})

print(f'Download link: {link}')

# print(l7_sr_images.getInfo())

