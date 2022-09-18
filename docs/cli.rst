Command line interface
----------------------

Getting started
^^^^^^^^^^^^^^^

.. include:: ../README.rst
    :start-after: cli_start
    :end-before: cli_end

.. _cli_running_examples:

Running examples
~~~~~~~~~~~~~~~~

The examples that follow use the ``homonim`` test data.  You can get this by doing a partial clone with `git <https://git-scm.com/downloads>`_:

.. code:: shell

    git clone --filter=blob:none https://github.com/dugalh/homonim.git

Change directories to the data root and make a *corrected* folder:

.. code:: shell

    cd homonim/tests/data
    mkdir corrected

..
        cd homonim/tests/data
        mkdir corrected

    Alternatively, by you can download the repository directly, and set downloading download and extract the repository as follows:

    .. code:: shell

        curl -LO# "https://github.com/dugalh/homonim/archive/refs/heads/main.zip"
        tar -xf main.zip
        cd homonim-main/tests/data
        mkdir corrected


Basic fusion and comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``gain-blk-offset`` model with a kernel shape of 5 x 5 pixels are the default fusion settings and work reasonably well for a variety of problems.  Here we specify these settings to correct the test aerial images with the Sentinel-2 reference.  The corrected images are placed in the *corrected* sub-directory.

.. code:: shell

   homonim fuse -m gain-blk-offset -k 5 5 -od ./corrected ./source/*rgb_byte*.tif ./reference/sentinel2_b432_byte.tif

To investigate the improvement in surface reflectance "accuracy" we compare the raw and corrected images with a second reference image not used in the fusion above, i.e. a Landsat-8 reference.

.. code:: shell

   homonim compare ./source/*rgb_byte*.tif ./corrected/*FUSE*.tif ./reference/landsat8_byte.tif

The last table printed by this command is:

.. code:: text

    ...
    Summary over bands:

                                                  File    r²    RMSE   rRMSE     N
    -------------------------------------------------- ----- ------ ------- -----
                                    ngi_rgb_byte_1.tif 0.390 93.517   2.454 28383
                                    ngi_rgb_byte_2.tif 0.488 94.049   2.380 28166
                                    ngi_rgb_byte_3.tif 0.386 88.610   2.323 27676
                                    ngi_rgb_byte_4.tif 0.607 89.409   2.412 27342
    ngi_rgb_byte_1_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif 0.924 16.603   0.489 28383
    ngi_rgb_byte_2_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif 0.906 15.590   0.445 28166
    ngi_rgb_byte_3_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif 0.881 15.531   0.456 27676
    ngi_rgb_byte_4_FUSE_cREF_mGAIN-BLK-OFFSET_k5_5.tif 0.897 15.702   0.474 27342

The correlation (r²) between the corrected and reference images is higher than the correlation between the source and reference images, indicating an improvement in similarity with the Landsat-8 reference.

It is possible to combine the above two commands, using the :option:`--compare <homonim-fuse --compare>` option as follows:

.. code:: shell

    homonim fuse -m gain-blk-offset -k 5 5 -od ./corrected --compare ./reference/landsat8_byte.tif --overwrite  ./source/*rgb_byte*.tif ./reference/sentinel2_b432_byte.tif


Band matching
~~~~~~~~~~~~~

``homonim`` automatically matches *source* to *reference* spectral bands when these images are either RGB or have *center_wavelength* metadata (as is the case with the ``homonim`` test data).  Subsets of *source* and/or *reference* bands to use for matching can be specified with the :option:`--src-band <homonim-fuse --src-band>` and :option:`--ref-band <homonim-fuse --ref-band>` options.

Let's *fuse* (harmonise) the red, green and blue bands of the Landsat-8 reference with the MODIS NBAR reference.  The :option:`--src-band <homonim-fuse --src-band>` option is used to specify the Landsat-8 band numbers corresponding to red, green and blue.  ``homonim`` then finds the matching MODIS NBAR bands.

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

In the case where *source* and *reference* are not RGB, and don't have *center_wavelength* metadata, it is up to the user to specify matching bands.  This can be done, simply by providing *source* and *reference* files with the same number of bands in the matching order (i.e. source bands 1, 2, .., N correspond to reference bands 1, 2, ..., N).  Or, matching and ordered subsets of *source* and *reference* bands can be specified with the :option:`--src-band <homonim-fuse --src-band>` and :option:`--ref-band <homonim-fuse --ref-band>` options.

Let's repeat the previous example to see how this would look.  Here, we also specify the matching reference bands with the :option:`--ref-band <homonim-fuse --ref-band>` option.

.. code:: shell

    homonim --verbose fuse --src-band 4 --src-band 3 --src-band 2 --ref-band 1 --ref-band 4 --ref-band 3 -od ./corrected --overwrite ./reference/landsat8_byte.tif ./reference/modis_nbar.tif

.. note::

    You can use ``gdalinfo`` to inspect the *center_wavelength*, and other metadata of an image.  E.g::

        gdalinfo ./reference/sentinel2_b432_byte.tif


Output file format
~~~~~~~~~~~~~~~~~~

By default ``homonim`` writes output files as GeoTIFFS with *DEFLATE* compression, *float32* data type and *nan* nodata value.  These options are all configurable.

Here we create a corrected image in JPEG format, with *uint8* data type, *0* nodata value, and a *QUALITY* setting of *85*.

.. code:: shell

    homonim fuse -od ./corrected  --driver JPEG --dtype uint8 --nodata 0 -co QUALITY=85 ./source/ngi_rgb_byte_4.tif ./reference/sentinel2_b432_byte.tif

Usage
^^^^^

.. click:: homonim.cli:cli
  :prog: homonim

.. click:: homonim.cli:fuse
  :prog: homonim fuse

.. click:: homonim.cli:compare
  :prog: homonim compare

.. click:: homonim.cli:stats
  :prog: homonim stats
