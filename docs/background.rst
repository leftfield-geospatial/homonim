Background
==========

Reference image
---------------

``homonim`` requires a *reference* surface reflectance image with which a *source* image is fused to produce the *corrected* image.  The *reference* is usually a satellite image at a coarser resolution that the *source*.  For best results, the *reference* should be chosen to satisfy the following criteria:

- Co-location: bounds of the *reference* image should cover those of the *source*, and *source* and *reference* should be ortho-rectified / co-registered.
- Concurrency: *source* and *reference* capture dates should be close in time, so there is minimal land cover change between them.
- Spectral similarity: the *reference* should contain bands whose spectral responses overlap with those of the *source*.

..
    While some care should be taken in selecting a *reference*, it is seldom difficult to satisfy these criteria in practice.

A number of satellite programs, including Landsat, Sentinel-2, and MODIS, provide suitable *reference* surface reflectance imagery freely to the public.  geedim_ is recommended as a companion tool to ``homonim`` for acquiring cloud/shadow-free imagery from these, and other programs.  geedim_ acquired imagery includes metadata that is used by ``homonim`` for automatic matching of spectral bands.  Alternatively, satellite imagery is available from a number of sources, including the `Google <https://developers.google.com/earth-engine/datasets>`_, `Amazon <https://aws.amazon.com/earth/>`_ and `Microsoft <https://planetarycomputer.microsoft.com/catalog>`_ repositories.

Source image
------------

Any orthorectified, multi-spectral *source* imagery can be used with ``homonim``, including drone, aerial and satellite imagery. *Source* images should  preferably be provided to ``homonim`` without gamma correction or colour balancing type adjustments.  If this is not possible, ``homonim`` will still improve surface reflectance accuracy.


Fusion
------

``homonim`` uses spatially varying localised *models* to describe the surface reflectance relationship between *source* and *reference*.  These *models* are fitted at each pixel location, inside a small *kernel* (window), using a fast DFT approach.  After fitting, ``homonim`` produces the *corrected* image by applying the models to the *source* (i.e. "fusing" the *source* with the *reference*.).

From the user perspective, the *kernel shape* (pixel dimensions) and *model* are the main parameters for configuring *fusion*.  When not specified, ``homonim`` uses default values that will provide reasonable results for most use cases.

More details on the theoretical basis for the method can be found in the `paper <https://raw.githubusercontent.com/dugalh/homonim/main/docs/radiometric_homogenisation_preprint.pdf>`_.

Example
-------

Mosaics of 0.5 m resolution aerial imagery before and after correction with a Landsat-7 reference image.

.. image:: https://raw.githubusercontent.com/dugalh/homonim/update_docs/docs/background_eg.png
   :alt: example

.. |geedim| replace:: ``geedim``
.. _geedim: https://github.com/dugalh/geedim
