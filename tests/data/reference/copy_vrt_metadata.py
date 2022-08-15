import rasterio as rio

# copy band metadata from landsat tif to vrt
with rio.open('landsat8_byte.tif', 'r') as src_im:
    with rio.open('landsat8_byte.vrt', 'r+') as dest_im:
        for bi in range(1, src_im.count+1):
            dest_im.update_tags(bi, **src_im.tags(bi))
