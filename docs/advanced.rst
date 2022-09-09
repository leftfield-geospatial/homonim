Advanced topics
===============

Block processing
----------------

``homonim`` splits and processes *source* - *reference* image pairs in blocks.  This is done to avoid memory problems when processing large images, and to allow concurrent block processing, improving speed.  The block size (MB) and number of concurrent blocks can be set by the user.  See the relevant sections of the API and CLI docs for more details.

Processing CRS and re-projections
---------------------------------

Processing (i.e. model fitting or image comparison) takes place in one of the *source* or *reference* image CRS's (coordinate reference system) and pixel grid.  This CRS and pixel grid is termed the *processing CRS*.  By default ``homonim`` selects the lowest resolution of the *source* and *reference* CRSs as the *processing CRS*.  This follows the `original formulation <https://raw.githubusercontent.com/dugalh/homonim/main/docs/radiometric_homogenisation_preprint.pdf>`_ of the method, and is the recommended setting.  With these defaults, the *processing CRS* will be the *reference* CRS in the majority of cases i.e. where the *reference* is at a coarser resolution than the *source*.  The

In the majority of cases, users should leave the *processing CRS* and *up/downsampling* settings on their defaults.


With these defaults, the *processing CRS* will be the *reference* CRS in the majority of cases.  To perform processing, ``homonim`` re-projects one of the *source* or *reference* into the *processing CRS*,

.. image:: https://raw.githubusercontent.com/dugalh/homonim/main/docs/fusion_block_diagram.png
   :alt: example

..
    The user can however force the *processing CRS* to higher resolution of the *source* or *reference* CRS's.  This may be useful in certain special cases (e.g. investigating im correction methods).

..
    TO DO: refer to block processing parameters.
    TO DO: a why use homonim section with its advantages over other methods? speed (DFT & block proc), spatially varying correction &
    TO DO: an advanced section that discusses things like processing crs, block processing & mask_partial
