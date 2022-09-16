API
===

Getting started
---------------

Surface reflectance correction is implemented in the :class:`~homonim.fuse.RasterFuse` class.  Here we use it to correct an aerial image to surface reflectance, by fusing it with a Sentinel-2 reference.

.. code:: python

    from pathlib import Path
    from homonim import RasterFuse, RasterCompare, Model

    # urls of source and reference test images
    src_file = (
        'https://raw.githubusercontent.com/dugalh/homonim/main/'
        'tests/data/source/ngi_rgb_byte_1.tif'
    )
    ref_file = (
        'https://raw.githubusercontent.com/dugalh/homonim/main/'
        'tests/data/reference/sentinel2_b432_byte.tif'
    )

    # path to corrected file to create
    corr_file = './corrected.tif'

    # Correct src_file to surface reflectance by fusion with ref_file, using the
    # `gain-blk-offset` model and a kernel of 5 x 5 pixels.
    with RasterFuse(src_file, ref_file) as fuse:
        fuse.process(corr_file, Model.gain_blk_offset, (5, 5), overwrite=True)

Next, we use :class:`~homonim.compare.RasterCompare` to compare the raw and corrected aerial images with a second reference i.e. a Landsat-8 image.  The comparison results give an indication of the accuracy improvement due to fusion.

.. code:: python

    # url of independent landsat reference for evaluation
    cmp_ref_file = (
        'https://raw.githubusercontent.com/dugalh/homonim/main/'
        'tests/data/reference/landsat8_byte.tif'
    )

    # Compare source and corrected similarity with the independent reference,
    # cmp_ref_file, giving an indication of the improvement in surface reflectance
    # accuracy.
    print('\nComparison key:\n' + RasterCompare.schema_table())
    for cmp_src_file in [src_file, corr_file]:
        print(
            f'\nComparing {Path(cmp_src_file).name} with '
            f'{Path(cmp_ref_file).name}:'
        )
        with RasterCompare(cmp_src_file, cmp_ref_file) as compare:
            cmp_stats = compare.process()
            print(compare.stats_table(cmp_stats))

You can also take a look at the :ref:`tutorial <Tutorial>` for a more detailed walk through the API.

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
