Advanced usage
==============

Block processing
----------------

``homonim`` splits and processes *source* - *reference* image pairs in blocks.  This is done to limit memory usage and allow the processing of large images.  In addition, processing speed is increased by processing blocks concurrently.  The block size (MB) and number of concurrent blocks can be set by the user with the :option:`--max-block-mem <homonim fuse --max-block-mem>` and :option:`--threads <homonim fuse --threads>` command line parameters respectively; and the ``block_config`` parameter of the :meth:`homonim.RasterFuse.process` API.

Processing CRS and re-projections
---------------------------------

Processing (i.e. model fitting or image comparison) takes place in one of the *source* or *reference* image CRS's (coordinate reference system) and pixel grid.  This CRS and pixel grid is termed the *processing CRS*.  By default ``homonim`` selects the lowest resolution of the *source* and *reference* CRSs as the *processing CRS*.  This follows the `original formulation <https://raw.githubusercontent.com/dugalh/homonim/main/docs/radiometric_homogenisation_preprint.pdf>`_ of the method.  In the majority of cases, the *reference* is at a lower resolution than the *source*, meaning the *processing CRS* will be the *reference* CRS.

Before processing ``homonim`` re-projects one of the *source* or *reference* so that both are in the same *processing CRS*.   In the typical case for surface reflectance correction, the *source* image is re-projected to the *reference CRS*.  Then, once calculated, the correction parameters are re-projected back to the *source* CRS, and applied to the *source* image to produce the corrected image.  The diagram below illustrates this process.

.. image:: https://raw.githubusercontent.com/dugalh/homonim/update_docs/docs/fusion_block_diagram.png
   :alt: example


Similarly, when comparing images, ``homonim`` would re-project the *source* to the *reference* in the typical case.  The comparison is then performed in this (the *processing CRS*), with no further re-projections.

``homonim`` will also work in the unusual case where the *reference* is at a higher resolution than the *source*, but here the default *processing CRS* is the *source* CRS.

By default ``homonim`` uses *average* resampling when downsampling (re-projecting from high to low resolution), and *cubic spline* resampling when upsampling (re-projecting from low to high resolution).  The reasons for these choices are explained in the `paper <https://raw.githubusercontent.com/dugalh/homonim/main/docs/radiometric_homogenisation_preprint.pdf>`_.

While the defaults settings are recommended, the ``homonim`` CLI and API do allow the user to specify the *processing CRS*, and resampling methods for *downsampling* and *upsampling*.  On the command line, the relevant parameters are :option:`--proc-crs <homonim fuse --proc-crs>`, :option:`--downsampling <homonim fuse --downsampling>` and :option:`--upsampling <homonim fuse --upsampling>`.  Via the API, the corresponding settings are the ``proc_crs`` argument of :meth:`RasterFuse` and the ``model_config`` argument of :meth:`homonim.RasterFuse.process`.


..
    The user can however force the *processing CRS* to higher resolution of the *source* or *reference* CRS's.  This may be useful in certain special cases (e.g. investigating im correction methods).

..
    TO DO: refer to block processing parameters.
    TO DO: a why use homonim section with its advantages over other methods? speed (DFT & block proc), spatially varying correction &
    TO DO: an advanced section that discusses things like processing crs, block processing & mask_partial
