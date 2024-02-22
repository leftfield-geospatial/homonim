Command line interface
======================

Getting started
---------------

.. include:: ../README.rst
    :start-after: cli_start
    :end-before: cli_end

.. _cli_running_examples:

Running examples
^^^^^^^^^^^^^^^^

The examples that follow use the ``homonim`` test images.  You can get these by downloading the repository directly:

.. code:: shell

    curl -LO# "https://github.com/leftfield-geospatial/homonim/archive/refs/heads/main.zip"
    tar -xf main.zip

Alternatively, you can clone the repository with `git <https://git-scm.com/downloads>`_:

.. code:: shell

    git clone https://github.com/leftfield-geospatial/homonim.git

After you have the repository, navigate to *<homonim root>/tests/data*, and create a *corrected* sub-directory to contain processed images:

.. code:: shell

    cd <homonim root>/tests/data
    mkdir corrected

The commands that follow use relative paths, and should be run from *<homonim root>/tests/data* (*<homonim root>* will be one of *homonim-main* or *homonim*, depending on your download method).


Basic fusion and comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *gain-blk-offset* model and kernel shape of 5 x 5 pixels are the default fusion settings and work reasonably well for a variety of problems.  Here we specify these settings to correct the test aerial images with the Sentinel-2 reference.

.. code:: shell

   homonim fuse -m gain-blk-offset -k 5 5 -od ./corrected ./source/*rgb_byte*.tif ./reference/sentinel2_b432_byte.tif

The corrected images are placed in the *corrected* sub-directory, and named with a *FUSE_cREF_mGAIN-BLK-OFFSET_k5_5* postfix describing the fusion parameters.

To investigate the change in surface reflectance accuracy, we compare source and corrected images with a second reference image not used for fusion, i.e. a Landsat-8 reference.

.. code:: shell

   homonim compare ./source/*rgb_byte*.tif ./corrected/*FUSE*.tif ./reference/landsat8_byte.tif

This command prints a series of tables describing the per-band similarity between each source/corrected and reference image pair.  The last table summarises these results per-image:

.. code:: text

    ...
    Summary over bands:

                                                  File    r²   RMSE   rRMSE     N
    -------------------------------------------------- ----- ------ ------- -----
                                    ngi_rgb_byte_1.tif 0.390 93.517   2.454 28383
                                    ngi_rgb_byte_2.tif 0.488 94.049   2.380 28166
                                    ngi_rgb_byte_3.tif 0.386 88.610   2.323 27676
                                    ngi_rgb_byte_4.tif 0.607 89.409   2.412 27342
    ngi_rgb_byte_1_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif 0.924 16.603   0.489 28383
    ngi_rgb_byte_2_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif 0.906 15.590   0.445 28166
    ngi_rgb_byte_3_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif 0.881 15.531   0.456 27676
    ngi_rgb_byte_4_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif 0.897 15.702   0.474 27342

The first four rows list the source images, and the last four, the corrected images.

It is possible to combine the above *fuse* and *compare* commands, using the :option:`--compare <homonim-fuse --compare>` option as follows:

.. code:: shell

    homonim fuse -m gain-blk-offset -k 5 5 -od ./corrected --compare ./reference/landsat8_byte.tif --overwrite  ./source/*rgb_byte*.tif ./reference/sentinel2_b432_byte.tif


Band matching
^^^^^^^^^^^^^

``homonim`` automatically matches *source* to *reference* spectral bands when these images are either RGB or have ``center_wavelength`` metadata (as is the case with the ``homonim`` test data).  Subsets of *source* and/or *reference* bands to use for matching can be specified with the :option:`--src-band <homonim-fuse --src-band>` and :option:`--ref-band <homonim-fuse --ref-band>` options.

Let's correct the red, green and blue bands of the Landsat-8 reference with the MODIS NBAR reference.  The :option:`--src-band <homonim-fuse --src-band>` option is used to specify the Landsat-8 band numbers corresponding to red, green and blue.  ``homonim`` then finds the matching MODIS NBAR bands.

.. code:: shell

    homonim --verbose fuse --src-band 4 --src-band 3 --src-band 2 -od ./corrected ./reference/landsat8_byte.tif ./reference/modis_nbar.tif

With the :option:`--verbose <homonim --verbose>` option specified above, ``homonim`` prints a table showing which *source* and *reference* bands have been matched:

.. code:: text

    Source    Source                   Source  Ref                   Ref                        Ref
    Name      Description             Wavelen  Name                  Description            Wavelen
    --------  --------------------  ---------  --------------------  -------------------  ---------
    SR_B2     Band 2 (blue)             0.482  Nadir_Reflectance_Ba  NBAR at local solar      0.469
              surface reflectance              nd3                   noon for band 3
    SR_B3     Band 3 (green)            0.562  Nadir_Reflectance_Ba  NBAR at local solar      0.555
              surface reflectance              nd4                   noon for band 4
    SR_B4     Band 4 (red) surface      0.655  Nadir_Reflectance_Ba  NBAR at local solar      0.645
              reflectance                      nd1                   noon for band 1

In the case where *source* and *reference* are not RGB, and don't have ``center_wavelength`` metadata, it is up to the user to specify matching bands.  This can be done simply by providing *source* and *reference* files with the same number of bands in the matching order (i.e. source bands 1, 2, .., N correspond to reference bands 1, 2, ..., N).  Or, by specifying matching order subsets of *source* and *reference* bands with the :option:`--src-band <homonim-fuse --src-band>` and :option:`--ref-band <homonim-fuse --ref-band>` options.

Let's repeat the previous example to see how this would look.  Here, we also specify the matching reference bands with the :option:`--ref-band <homonim-fuse --ref-band>` option (source bands 4, 3, 2 match reference bands 1, 4, 3 - in that order).

.. code:: shell

    homonim --verbose fuse --src-band 4 --src-band 3 --src-band 2 --ref-band 1 --ref-band 4 --ref-band 3 -od ./corrected --overwrite ./reference/landsat8_byte.tif ./reference/modis_nbar.tif

.. note::
    Images downloaded with `geedim <https://github.com/leftfield-geospatial/geedim>`_ have ``center_wavelength`` metadata compatible with ``homonim``.

    You can use ``gdalinfo`` (from the `gdal <https://github.com/OSGeo/gdal>`_ package) to inspect the ``center_wavelength``, and other metadata of an image, e.g::

        gdalinfo ./reference/sentinel2_b432_byte.tif


Output file format
^^^^^^^^^^^^^^^^^^

By default ``homonim`` writes output files as *GeoTIFF*\s with *DEFLATE* compression, *float32* data type and *nan* nodata value.  These options are all configurable.

Here we create a *JPEG* compressed image in *GeoTIFF* format, with *uint8* data type:

.. code:: shell

    homonim fuse -od ./corrected --driver GTiff --dtype uint8 --nodata null -co COMPRESS=JPEG -co INTERLEAVE=PIXEL -co PHOTOMETRIC=YCBCR ./source/ngi_rgb_byte_4.tif ./reference/sentinel2_b432_byte.tif

Setting nodata to *null* forces the writing of an internal mask.  This avoids lossy compression `transparency artifacts <https://gis.stackexchange.com/questions/114370/compression-artifacts-and-gdal>`__.  JPEG compression is configured to sub-sample YCbCr colour space values with the ``-co INTERLEAVE=PIXEL -co PHOTOMETRIC=YCBCR`` creation options.

Next, the corrected image is formatted as a 12 bit JPEG compressed GeoTIFF.  A *null* nodata value is used again to write an internal mask, and the *uint16* data type gets truncated to 12 bits:

.. code:: shell

    homonim fuse -od ./corrected --driver GTiff --dtype uint16 --nodata null -co COMPRESS=JPEG -co NBITS=12 ./source/ngi_rgb_byte_4.tif ./reference/sentinel2_b432_byte.tif

The ``-co INTERLEAVE=PIXEL -co PHOTOMETRIC=YCBCR`` creation options could also be used with this example, to further compress the RGB image.

.. note::

    Support for 12 bit JPEG compression is `rasterio <https://rasterio.readthedocs.io/en/stable>`__ build / package dependent.

See the `GDAL docs <https://gdal.org/drivers/raster/index.html>`__ for available drivers and their parameters.

Usage
-----

.. click:: homonim.cli:cli
  :prog: homonim

.. click:: homonim.cli:fuse
  :prog: homonim fuse

.. click:: homonim.cli:compare
  :prog: homonim compare

.. click:: homonim.cli:stats
  :prog: homonim stats
