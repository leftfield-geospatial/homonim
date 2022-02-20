=====
Usage
=====

----------
Background
----------
``homonim`` uses a *reference* surface reflectance image to which a *source* image is homogenised.  The *reference* image is usually a satellite image at a coarser resolution that the *source* image.  The surface reflectance relationship between *source*  and *reference* images is approximated with localised linear models.  Models are estimated for each pixel location inside a small rectangular *kernel* (window), using a fast DFT approach.  The model parameters are applied to the *source* image to produce the homogenised output.  

The theoretical basis for using local linear models is explained in the paper_.  Broadly speaking, the gain term compensates for atmospheric absorption and BRDF effects, while the offset compensates for atmospheric reflectance and haze.

-----------------
Image preparation
-----------------
Before homogenising, a *reference* image needs to be acquired.  Examples of suitable surface reflectance image collections for the *reference* image are those produced by Landsat, Sentinel-2 and MODIS.  There are a number of platforms providing these images including the Google_ and `Amazon <https://aws.amazon.com/earth/>`_ repositories.  See other options `here <https://eos.com/blog/free-satellite-imagery-sources/>`_.

|geedim|_ can be used as a companion tool to ``homonim`` for acquiring *reference* imagery.  It provides command line search, cloud/shadow-free compositing, and download of `Google Earth Engine`_ surface reflectance imagery.

For best results, the *reference* and *source* image(s) should be:

* **Concurrent**:  Capture dates are similar.
* **Co-located**:  Accurately co-registered / orthorectified.
* **Spectrally similar**:  Band spectral responses overlap.

The *reference* image bounds should contain those of the *source* image(s), and *source* / *reference* bands should correspond i.e. *reference* band 1 corresponds to *source* band 1, *reference* band 2 corresponds to *source* band 2 etc.  |rasterio|_ and |gdal|_ provide command line tools for re-ordering bands etc. ``rasterio`` is included in the ``homonim`` installation.

The `method formulation <https://www.researchgate.net/publication/328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data>`_ assumes *source* images are raw i.e. without colour-balancing, gamma-correction etc adjustments.  Where possible, this assumption should be adhered to.  Adjusted *source* images will still benefit from homogenisation, however.  


----------------------
Command line interface
----------------------
All homogenisation functionality is accessed via the ``homonim`` command and its sub-commands.  Use ``homonim <command> --help`` for command line help.

homonim
========

``homonim [OPTIONS] COMMAND [ARGS]...``

Options
-------
``-v``, ``--verbose``:
    Increase verbosity.
``-q``, ``--quiet``:
    Decrease verbosity.
``--help``:
    Show the command line help message and exit.

Commands
--------

|fuse|_ :
    Radiometrically homogenise image(s) by fusion with a reference.
|compare|_ :
    Report similarity statistics between image(s) and a reference.
|stats|_ :
    Report parameter image statistics.


fuse
====

``homonim fuse [OPTIONS] INPUTS... REFERENCE``

Radiometrically homogenise image(s) by fusion with a reference.  

Examples 
--------
Homogenise *source.tif* with *reference.tif*, using the ``gain-blk-offset`` method_, and a kernel_ of 5 x 5 pixels.

.. code-block:: console

    homonim fuse -m gain-blk-offset -k 5 5 source.tif reference.tif

Homogenise files matching *source\*.tif* with *reference.tif*, using the ``gain-offset`` method_ and a kernel_ of 15 x 15 pixels. Place homogenised files in the *./homog* directory, produce parameter images, and mask partially covered pixels in the homogenised images.

.. code-block:: console

    homonim fuse --method gain-offset --kernel-shape 15 15 -od ./homog --param-image --mask-partial source*.tif reference.tif

Required arguments
------------------
``INPUTS`` : 
    Path(s) to *source* image(s) to be homogenised.
``REFERENCE`` : 
    Path to a surface reflectance *reference* image.  

Standard options
----------------
.. _method:

``-m``, ``--method``:  ``gain``, ``gain-blk-offset`` or ``gain-offset``
    Homogenisation method:
    
    * ``gain``: A gain-only model, suitable for haze-free and zero offset images (i.e. images where a surface reflectance of zero corresponds to a pixel value of ~zero).
    * ``gain-blk-offset``: A gain-only model applied to offset normalised image blocks.  The image block size is determined by max-block-mem_.  It is the default method and is suitable for most image combinations.  
    * ``gain-offset``: A gain and offset model.  The most accurate method, but sensitive to differences between *source* and *reference*, such as shadowing and land cover changes.  Suitable for well-matched *source* / *reference* combinations.  (See also the associated r2-inpaint-thresh_ option.)

.. _kernel-shape:

``-k``, ``--kernel-shape``: <HEIGHT WIDTH> as odd *integers*
    The kernel height and width in pixels of the |proc-crs|_ image.  Larger kernels are less susceptible to over-fitting on noisy data, while smaller kernels provide higher spatial resolution homogenisation parameters. The minimum ``kernel-shape`` is 1 x 1 for the ``gain`` and ``gain-blk-offset`` methods_, and 5 x 5 for the ``gain-offset`` method_. The default is a 5 x 5 pixel kernel.

.. _output-dir:

``-od``, ``--output-dir``: DIRECTORY
   The directory in which to create homogenised image(s).  Homogenised image(s) are named automatically based on the *source* file name and option values. The default ``output-dir`` is the source image directory. 

.. _overwrite:

``-ovw``, ``--overwrite``:
    If specified, existing output file(s) are overwritten.  The default is to raise an exception when the output file already exists.

.. _compare_option:

``-cmp``, ``--compare``:
    Report statistics describing the similarity of the *source* and *reference*, and homogenised and *reference* image pairs.  Useful for comparing the effects of differerent ``method``, ``kernel-shape`` etc. options.

.. _no-build-ovw:

``-nbo``, ``--no-build-ovw``:
    If specified, overview building is turned off.  The default is to build overviews for all output files.

.. _proc-crs:

``-pc``, ``--proc-crs``: ``auto``, ``src`` or ``ref``
    The image CRS in which to perform parameter estimation.
    
    * ``auto``: Estimate parameters in the lowest resolution of the *source* and *reference* image CRS's. This is the default, and recommended setting.
    * ``src``: Estimate parameters in the *source* image CRS.
    * ``ref``: Estimate parameters in the *referemce* image CRS.

.. _conf:

``-c``, ``--conf`` : FILE
    Path to a yaml configuration file specifying the `advanced options`_.  Command line options take precedence over those from the config file, which take precedence over those derived from the defaults.  See `config.yaml`_ for an example.

.. _help:

``--help``
    Show the command line help message and exit.


Advanced options
----------------

.. _param-image:

``-pi``, ``--param-image``:
    Create a debug image containing the model parameters and R² values for each homogenised image.

.. _mask-partial:

``-mp``, ``--mask-partial``:
    Mask biased homogenised pixels produced from partial kernel or *source* / *reference* image coverage.  This option reduces seamlines in mosaics of overlapping images.

.. _threads:

``-t``, ``--threads``: INTEGER
    The number of image blocks to process concurrently (0 = process as many blocks as there are cpus).  Note that the amount of memory used by ``homonim`` increases with this number.  The default is 0.  

.. _max-block-mem:

``-mbm``, ``--max-block-mem``: FLOAT
    The maximum image block size in megabytes (0 = block size is the image size).  ``homonim`` processes images in blocks to reduce memory usage, and allow concurrency.   The image block size is determined automatically, using this option as an upper limit.  The default is 100.  

.. _downsampling:

``-ds``, ``--downsampling``: ``nearest``, ``bilinear``, ``cubic``, ``cubic_spline``, ``lanczos``, ``average``, ``mode``, ``max``, ``min``, ``med``, ``q1``, ``q3``, ``sum`` or ``rms``
    The resampling method for re-projecting from high to low resolution. See the `rasterio docs`_ for details on the available options.  ``average`` is the default (recommended).

.. _upsampling:

``-us``, ``--upsampling``: ``nearest``, ``bilinear``, ``cubic``, ``cubic_spline``, ``lanczos``, ``average``, ``mode``, ``max``, ``min``, ``med``, ``q1``, ``q3``, ``sum`` or ``rms``
    The resampling method for re-projecting from low to high resolution. See the `rasterio docs`_ for details on the available options.  ``cubic_spline`` is the default (recommended).

.. _r2-inpaint-thresh:

``-rit``, ``--r2-inpaint-thresh``: FLOAT 0-1
    The kernel model R² (coefficient of determination) threshold below which to inpaint the offset parameter from surrounding areas (0 = turn off inpainting). The gain parameter is re-fitted with the inpainted offsets.  This option applies only to ``gain-offset``, and can improve the stability of this method in noisy areas.  The default is 0.25.

.. _out-driver:

``--out-driver``: TEXT
    The output image format driver.  See the `GDAL driver`_ documentation for options.  ``GTiff`` is the default (recommended).

.. _out-dtype:

``--out-dtype``: ``uint8``, ``uint16``, ``int16``, ``uint32``, ``int32``, ``float32`` or ``float64``
    The output image data type.  ``float32`` is the default.

.. _out-nodata:

``--out-nodata``: NUMBER, ``null`` or ``nan``
    The output image nodata value (``null`` = no nodata value).  ``nan`` is the default.

.. _out-profile:

``-co``, ``--out-profile``: NAME=VALUE
    Driver specific image creation options for the output image(s).  For details of available options for a particular driver, see the `GDAL driver`_ documentation.  This option can be repeated e.g. ``-co COMPRESS=DEFLATE -co TILED=YES ...``.  The default ``GTiff`` creations options are: ``TILED=YES``, ``BLOCKXSIZE=512``, ``BLOCKYSIZE=512``, ``COMPRESS=DEFLATE`` and ``INTERLEAVE=BAND``.  Other format drivers have no defaults.  If out-driver_ matches the format of the *source* image, output creation options are copied from the *source* image, and overridden with any equivalent command line out-profile specifications or defaults.


compare
=======

``homonim compare [OPTIONS] INPUTS... REFERENCE``

Report similarity statistics between image(s) and a reference.  

Example
-------
Compare *source.tif* and *homogenised.tif* with *reference.tif*.

.. code-block:: console

    homonim compare source.tif homogenised.tif reference.tif


Required arguments
------------------
``INPUTS`` :
    Path(s) to image(s) to be compared.

``REFERENCE`` :
    Path to a surface reflectance *reference* image.  

Options
-------

.. _proc_crs_compare:

``-pc``, ``--proc-crs``: ``auto``, ``src`` or ``ref``
    The image CRS in which to perform the comparison.
    
    * ``auto``: Compare images in the lowest resolution of the *source* and *reference* image CRS's. This is the default, and recommended setting.
    * ``src``: Compare images in the *source* image CRS.
    * ``ref``: Compare images in the *reference* image CRS.

.. _output_compare:

``-o``, ``--output``: FILE
    Write results to a json file.

.. _help_compare:

``--help``
    Show the command line help message and exit.

stats
=====

``homonim stats [OPTIONS] INPUTS...``

Report parameter image statistics.

Example
-------
Report statistics for *param.tif*.

.. code-block:: console

    homonim stats param.tif


Required arguments
------------------

``INPUTS``:
    Path(s) to parameter image(s).  These are images produced by ``homonim`` |fuse|_ with the --|param-image|_ option.

Options
-------

.. _output_stats:

``-o``, ``--output``: FILE
    Write results to a json file.

.. _help_stats:

``--help``
    Show the command line help message and exit.



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
.. _kernel: `kernel-shape`_
.. _`Google Earth Engine`: Google_
.. _paper: `method formulation`_