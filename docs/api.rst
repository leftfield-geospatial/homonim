API
===

Getting started
---------------

The example below demonstrates how to use the :class:`~homonim.fuse.RasterFuse` and :class:`~homonim.compare.RasterCompare` classes for surface reflectance correction and validation.

.. include:: ../README.rst
    :start-after: api_example_start
    :end-before: api_example_end

You can also take a look at the :ref:`tutorial <Tutorial>` for a more detailed walk through.

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
