=====
Usage
=====

----------
Background
----------
``homonim`` uses a *reference* surface reflectance image to adjust a *source* image.  Typically the *reference* image is a satellite image at a coarser resolution that the *source* image. The surface reflectance relationship between *source*  and *reference* images is approximated with localised linear models.  Models are estimated for each pixel location inside a small rectangular *kernel* (window), using a fast DFT approach.  The homogenised output is produced by applying the model parameters to the source image.  [Mention what offset and gain compensate for]

-----------------
Image preparation
-----------------
Before homogenising, a suitable *reference* image needs to be acquired.  For best results, the *reference* and *source* image(s) should be concurrent, co-located (accurately co-registered / orthorectified), and spectrally similar (with overlapping band spectral responses).

The *reference* image bounds should encompass those of the *source* image(s), and *source* / *reference* band ordering should match (i.e. reference band 1 corresponds to source band 1, reference band 2 corresponds to source band 2 etc).  |rasterio|_ or |gdal|_ command line tools can be used to re-order bands etc. as necessary.  These packages are included in the ``homonim`` installation.  

Examples of suitable surface reflectance image collections for the *reference* image are those produced by Landsat, Sentinel-2 and MODIS.  There are a number of platforms, and associated tools, for acquiring these images, including the `Google <https://developers.google.com/earth-engine/datasets>`_ and `Amazon <https://aws.amazon.com/earth/>`_ repositories.

|geedim|_ can be used as a companion tool to ``homonim``, and allows search, basic cloud/shadow-free compositing, and download of Google Earth Engine surface reflectance imagery.  More details `here <https://github.com/dugalh/geedim>`_.

Where possible, ``homonim`` should be applied to raw *source* imagery i.e. without colour-balancing or gamma-correction etc.  This will help satisfy the assumptions of the method, but is not strictly necessary, and adjusted *source* images will still benefit from homogenisation.  

----------------------
Command line interface
----------------------
All functionality can be accessed via the command line.  There are three sub-commands: `compare`, `fuse` and `stats`.   

Fuse
====
``homonim fuse [OPTIONS] INPUTS... REFERENCE``

Required arguments
------------------
``INPUTS`` : 
    Path(s) to *source* image(s) to be homogenised.
``REFERENCE`` : 
    Path to a surface reflectance *reference* image.  The *reference* image bounds need to contain the *source* images(s), and *source* / *referennce* band ordering should match (i.e. reference band 1 corresponds to source band 1, reference band 2 corresponds to source band 2 etc).

Standard options
----------------
.. _method:

``-m, --method`` :  '``gain``', '``gain-blk-offset``' or '``gain-offset``'
    Homogenisation method:
    
    * ``gain`` : A gain-only model, suitable for haze-free and 0 offset images (i.e. images where a surface reflectance of 0, corresponds to a pixel value of ~0).
    * ``gain-blk-offset`` : A gain-only model applied to offset normalised image blocks.  The image block size is determined by max-block-mem_.  Effectively this applies a fine-scale gain and coarse-scale offset.  This is the default method - it is robust and suitable for most image combinations.
    * ``gain-offset`` : A gain and offset model.  The most accurate method, but sensitive to differences between *source* and *reference*, such as shadowing and land cover changes.  ``gain-offset`` requires a larger |kernel-shape|_ than the other methods as it has i.e. a minimum of 5 x 5 pixels.  Suitable for well-matched *source*/*reference* combinations.

.. _kernel-shape:

``-k, --kernel-shape``: <HEIGHT WIDTH> as odd *integers*
    The kernel height and width in pixels (of the |proc-crs|_ image).  Larger kernels are less susceptible to over-fitting on noisy data, while smaller kernels provide higher spatial resolution homogenisation parameters. The minimum ``kernel-shape`` is 1 x 1 for the ``gain`` and ``gain-blk-offset`` methods_, and 5 x 5 for the ``gain-offset`` method_. The default is a 5 x 5 pixel kernel.

``-od, --output-dir``: DIRECTORY
   The directory in which to create homogenised image(s).  Homogenised image(s) are named automatically based on the *source* file name and option values. The default ``output-dir`` is the source image directory.

``-ovw, --overwrite``:
    If specified, existing output file(s) will be overwritten.  The default is to raise an exception when the output file already exists.

``-cmp, --compare``:
    Report statistics describing the similarity of the *source* and *reference*, and homogenised and *reference* images.  These statitics give an indication of the improvement in surface reflectance homogeneity, and can be used for comparing performance of differerent options.   

``-nbo, --no-build-ovw``:
    If specified, overview builidng is turned off.  The default is to build overviews for all output files.

.. _proc-crs:

``-pc, --proc-crs`` : '``auto``', '``src``' or '``ref``'
    The image CRS in which to perform processing.  [default: auto]

``-c, --conf`` : FILE
    Path to an optional configuration file. 


Advanced options
----------------


.. |rasterio| replace:: ``rasterio``
.. |gdal| replace:: ``gdal``
.. |geedim| replace:: ``geedim``
.. |gain| replace:: ``gain``
.. |gain-blk-offset| replace:: ``gain-blk-offset``
.. |gain-offset| replace:: ``gain-offset``
.. |kernel-shape| replace:: ``kernel-shape``
.. |proc-crs| replace:: ``proc-crs``
.. |max-block-mem| replace:: ``max-block-mem``
.. _rasterio: https://rasterio.readthedocs.io/en/latest/cli.html
.. _gdal: https://gdal.org/programs/index.html
.. _geedim: https://github.com/dugalh/geedim
.. _methods: method_