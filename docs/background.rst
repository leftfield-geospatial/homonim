Background
==========

``homonim`` uses a *reference* surface reflectance image to which a *source* image is corrected.  The *reference* image is usually a satellite image at a coarser resolution that the *source* image.  The surface reflectance relationship between *source*  and *reference* images is approximated with localised linear models.  Models are estimated for each pixel location inside a small rectangular *kernel* (window), using a fast DFT approach.  The model parameters are applied to the *source* image to produce the corrected output.

The theoretical basis for using local linear models is explained in the paper_.  Broadly speaking, the gain term compensates for atmospheric absorption and BRDF effects, while the offset compensates for atmospheric reflectance and haze.

.. _`method formulation`: https://www.researchgate.net/publication/328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data
.. _paper: `method formulation`_