API
===

RasterFuse
----------

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

    ~RasterFuse.proc_crs
    ~RasterFuse.closed


RasterCompare
-------------

.. currentmodule:: homonim

.. autoclass:: RasterCompare
    :special-members: __init__

.. rubric:: Methods

.. autosummary::
    :toctree: _generated

    ~RasterCompare.compare
    ~RasterCompare.create_config
    ~RasterCompare.stats_table

.. rubric:: Attributes

.. autosummary::
    :toctree: _generated

    ~RasterCompare.schema
    ~RasterCompare.schema_table
    ~RasterCompare.proc_crs
    ~RasterFuse.closed


ParamStats
----------

.. currentmodule:: homonim

.. autoclass:: ParamStats
    :special-members: __init__

.. rubric:: Methods

.. autosummary::
    :toctree: _generated

    ~ParamStats.stats
    ~ParamStats.stats_table

.. rubric:: Attributes

.. autosummary::
    :toctree: _generated

    ~ParamStats.metadata
    ~ParamStats.closed


enums
-----

Model
^^^^^^

.. currentmodule:: homonim.enums

.. autoclass:: Model
    :members:

ProcCrs
^^^^^^^

.. currentmodule:: homonim.enums

.. autoclass:: ProcCrs
    :members:

Testing
-------

.. currentmodule:: homonim.raster_pair

.. autoclass:: RasterPairReader
    :members:

