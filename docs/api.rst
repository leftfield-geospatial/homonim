API
===

Getting started
---------------

The :class:`~homonim.fuse.RasterFuse` class implements surface reflectance correction.  Here we use it to correct an aerial image to surface reflectance, by fusing it with a Sentinel-2 reference.  The images are taken from the ``homonim`` test data.

.. literalinclude:: examples/api_example.py
    :language: python
    :start-after: [correct-start]
    :end-before: [correct-end]


:class:`~homonim.compare.RasterCompare` can be used for comparing two images.  In the next snippet, we use it to compare the source and corrected aerial images with a second reference using :class:`~homonim.compare.RasterCompare`.  The tabulated results give an indication of surface reflectance accuracy before and after correction.

.. literalinclude:: examples/api_example.py
    :language: python
    :start-after: [compare-start]
    :end-before: [compare-end]

The output:

.. code:: text

    ABBREV   DESCRIPTION
    -------- -----------------------------------------
    r²       Pearson's correlation coefficient squared
    RMSE     Root Mean Square Error
    rRMSE    Relative RMSE (RMSE/mean(ref))
    N        Number of pixels

         Band    r²   RMSE   rRMSE     N
    --------- ----- ------ ------- -----
       Source 0.390 93.517   2.454 28383
    Corrected 0.924 16.603   0.489 28383

Take a look at the :ref:`tutorial <Tutorial>` for a more comprehensive example.

Reference
---------

RasterFuse
^^^^^^^^^^

.. currentmodule:: homonim

.. autoclass:: RasterFuse
    :special-members: __init__

.. rubric:: Methods

.. autosummary::
    :toctree: _generated

    ~RasterFuse.create_model_config
    ~RasterFuse.create_block_config
    ~RasterFuse.create_out_profile
    ~RasterFuse.process

.. rubric:: Attributes

.. autosummary::
    :toctree: _generated

    ~RasterFuse.src_bands
    ~RasterFuse.ref_bands
    ~RasterFuse.proc_crs
    ~RasterFuse.closed


RasterCompare
^^^^^^^^^^^^^

.. currentmodule:: homonim

.. autoclass:: RasterCompare
    :special-members: __init__

.. rubric:: Methods

.. autosummary::
    :toctree: _generated

    ~RasterCompare.process
    ~RasterCompare.create_config
    ~RasterCompare.schema_table
    ~RasterCompare.stats_table

.. rubric:: Attributes

.. autosummary::
    :toctree: _generated

    ~RasterCompare.src_bands
    ~RasterCompare.ref_bands
    ~RasterCompare.schema
    ~RasterCompare.proc_crs
    ~RasterFuse.closed


ParamStats
^^^^^^^^^^

.. currentmodule:: homonim

.. autoclass:: ParamStats
    :special-members: __init__

.. rubric:: Methods

.. autosummary::
    :toctree: _generated

    ~ParamStats.stats
    ~ParamStats.schema_table
    ~ParamStats.stats_table

.. rubric:: Attributes

.. autosummary::
    :toctree: _generated

    ~ParamStats.schema
    ~ParamStats.metadata
    ~ParamStats.closed


enums
^^^^^

Model
~~~~~

.. currentmodule:: homonim.enums

.. autoclass:: Model
    :members:

ProcCrs
~~~~~~~

.. currentmodule:: homonim.enums

.. autoclass:: ProcCrs
    :members:

.. _tutorial:

Tutorial
--------

.. toctree::
    :maxdepth: 1

    examples/api_tutorial.ipynb
