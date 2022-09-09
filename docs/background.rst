Background
==========

Image preparation
-----------------

``homonim`` requires a *reference* surface reflectance image with which a *source* image is fused to produce the *corrected* image.  The *reference* is usually a satellite image at a coarser resolution that the *source*.  For best results, the *reference* should be chosen so that the *source* - *reference* pair meet the following the criteria:

- Co-location: bounds of the *reference* image should cover those of the *source*, and *source* and *reference* should be ortho-rectified, so that they are co-registered (spatially aligned).
- Concurrency: *source* and *reference* capture dates should be close in time, so there is minimal land cover change between them.
- Spectral similarity: the *reference* should contain bands that overlap spectrally with those of the *source*.

While some care should be taken in selecting a *reference*, it is seldom difficult to satisfy these criteria in practice.  A number of satellite programs, including Landsat, Sentinel-2, and MODIS, provide suitable surface reflectance imagery freely to the public.  geedim_ is recommended as a companion tool to ``homonim`` for acquiring *reference* imagery from these, and other programs.  Imagery acquired with geedim_ includes metadata that is used by ``homonim`` for automatic matching of spectral bands.  Alternatively, there are many sources of satellite imagery, and any *reference* which satisfies the criteria above will work.

Any orthorectified, multi-spectral *source* imagery can be used with ``homonim``, including drone, aerial and satellite imagery. *Source* images should be preferably be provided to ``homonim`` without gamma correction or colour balancing type adjustments.  If this is not possible, ``homonim`` will still improve surface reflectance accuracy.

Fusion concepts
---------------

``homonim`` uses spatially varying localised *models* to describe the surface reflectance relationship between *source* and *reference*.  *Models* are fitted at each pixel location, inside a small *kernel* (window), using a fast DFT approach.  Larger kernels are less susceptible to over-fitting on noisy data, while smaller kernels provide higher spatial resolution correction parameters.  A *model* form and fitting approach can be chosen from the available options in :class:`~homonim.enums.Model`.  After fitting, ``homonim`` produces the *corrected* image by applying the models to the *source*.  More details on the theoretical basis for the method can be found in the `paper <https://raw.githubusercontent.com/dugalh/homonim/main/docs/radiometric_homogenisation_preprint.pdf>`_.

From the user the perspective, the *kernel shape* (pixel dimensions) and *model* are the main parameters for configuring *fusion*.  ``homonim`` uses sensible defaults for these parameters, if they are not specified.


.. |geedim| replace:: ``geedim``
.. _geedim: https://github.com/dugalh/geedim
