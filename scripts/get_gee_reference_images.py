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
    src_bbox = geometry.shape(transform_geom(src_im.crs, 'WGS84', geometry.box(*src_im.bounds)))

l7_sr_images = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR").filterDate('2010-01-12', '2010-02-11').filterBounds(geometry.mapping(src_bbox))
print(f'Number of images: {l7_sr_images.size().getInfo()}')

link = l7_sr_images.median().getDownloadURL({
    'scale': 30,
    'crs': 'EPSG:4326',
    'fileFormat': 'GeoTIFF',
    'region': geometry.mapping(src_bbox)})

print(f'Download link: {link}')

# print(l7_sr_images.getInfo())

