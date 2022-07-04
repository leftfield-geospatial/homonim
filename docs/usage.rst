Background
==========

``homonim`` uses a *reference* surface reflectance image to which a *source* image is corrected.  The *reference* image is usually a satellite image at a coarser resolution that the *source* image.  The surface reflectance relationship between *source*  and *reference* images is approximated with localised linear models.  Models are estimated for each pixel location inside a small rectangular *kernel* (window), using a fast DFT approach.  The model parameters are applied to the *source* image to produce the corrected output.

The theoretical basis for using local linear models is explained in the paper_.  Broadly speaking, the gain term compensates for atmospheric absorption and BRDF effects, while the offset compensates for atmospheric reflectance and haze.

Image preparation
=================

Before correcting, a *reference* image needs to be acquired.  Examples of suitable surface reflectance image
collections for the *reference* image are those produced by Landsat, Sentinel-2 and MODIS.  There are a number of platforms providing these images including the Google_ and `Amazon <https://aws.amazon.com/earth/>`_ repositories.  See other options `here <https://eos.com/blog/free-satellite-imagery-sources/>`_.

|geedim|_ can be used as a companion tool to ``homonim`` for acquiring *reference* imagery.  It provides command line search, cloud/shadow-free compositing, and download of `Google Earth Engine`_ surface reflectance imagery.

For best results, the *reference* and *source* image(s) should be:

* **Concurrent**:  Capture dates are similar.
* **Co-located**:  Accurately co-registered / orthorectified.
* **Spectrally similar**:  Band spectral responses overlap.

The *reference* image bounds should contain those of the *source* image(s), and *source* / *reference* bands should correspond i.e. *reference* band 1 corresponds to *source* band 1, *reference* band 2 corresponds to *source* band 2 etc.  |rasterio|_ and |gdal|_ provide command line tools for re-ordering bands etc. ``rasterio`` is included in the ``homonim`` installation.

The `method formulation <https://www.researchgate
.net/publication/328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data>`_ assumes
*source* images are raw i.e. without colour-balancing, gamma-correction etc adjustments.  Where possible, this
assumption should be adhered to.  Adjusted *source* images will still benefit from correction, however.


.. |rasterio| replace:: ``rasterio``
.. |gdal| replace:: ``gdal``
.. |geedim| replace:: ``geedim``
.. |gain| replace:: ``gain``
.. |gain-blk-offset| replace:: ``gain-blk-offset``
.. |gain-offset| replace:: ``gain-offset``
.. |kernel-shape| replace:: ``kernel-shape``
.. |proc-crs| replace:: ``proc-crs``
.. |param-image| replace:: ``param-image``
.. |max-block-mem| replace:: ``max-block-mem``
.. |compare| replace:: ``compare``
.. |fuse| replace:: ``fuse``
.. |stats| replace:: ``stats``
.. _rasterio: https://rasterio.readthedocs.io/en/latest/cli.html
.. _`rasterio docs`: <https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling>
.. _gdal: https://gdal.org/programs/index.html
.. _geedim: https://github.com/dugalh/geedim
.. _Google: https://developers.google.com/earth-engine/datasets
.. _config.yaml: https://github.com/dugalh/homonim/blob/main/config.yaml
.. _`gdal driver`: https://gdal.org/drivers/raster/index.html
.. _`method formulation`: https://www.researchgate.net/publication/328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data
.. _methods: `method formulation`_
.. _`Google Earth Engine`: Google_
.. _paper: `method formulation`_