import rasterio as rio

# copy band metadata from landsat tif to vrt
with rio.open('LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_Byte.tif', 'r') as src_im:
    with rio.open('LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_Byte.vrt', 'r+') as dest_im:
        for bi in range(1, src_im.count+1):
            dest_im.update_tags(bi, **src_im.tags(bi))
