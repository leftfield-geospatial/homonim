|Tests| |codecov| |PyPI version| |conda-forge version| |docs| |License: AGPL v3|

homonim
=======

.. image:: https://raw.githubusercontent.com/dugalh/homonim/main/docs/readme_eg.webp
   :alt: example


.. short_descr_start

Correct aerial and satellite imagery to surface reflectance.

.. short_descr_end
.. description_start

Description
-----------

``homonim`` provides a command line interface and API for correcting remotely sensed imagery to approximate surface reflectance.  It implements a form of *spectral harmonisation*, that adjusts for spatially varying atmospheric and anisotropic (BRDF) effects, by *fusion* with satellite surface reflectance data.  Manual reflectance measurements and target placements are not required.

``homonim`` is useful for pre-processing in quantitative mapping applications, and for reducing seamlines and other visual artefacts in image mosaics.  It can be applied to multi-spectral drone, aerial and satellite imagery.  The consistency of multi-temporal and multi-sensor data can improved through its use.

.. description_end

See the documentation site for more detail: https://homonim.readthedocs.io/.

.. install_start

Installation
------------

``homonim`` is available as a python 3 package, via ``pip`` and ``conda``.

conda
~~~~~

Under Windows, using ``conda`` is the easiest way to resolve binary dependencies. The
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ installation provides a minimal ``conda``.

.. code:: shell

   conda install -c conda-forge homonim

pip
~~~

.. code:: shell

   pip install homonim

.. install_end

Getting started
---------------

Command line interface
~~~~~~~~~~~~~~~~~~~~~~

.. cli_start

``homonim`` command line functionality is accessed through the commands:

-  ``fuse``: Correct image(s) to surface reflectance.
-  ``compare``: Compare image(s) with a reference.
-  ``stats``: Report parameter statistics.

Get help on ``homonim`` with:

.. code:: shell

   homonim --help

and help on a ``homonim`` command with:

.. code:: shell

   homonim <command> --help

.. cli_end

Examples
^^^^^^^^

Correct *source.tif* to surface reflectance by fusion with *reference.tif*, using the default settings:

.. code:: shell

    homonim fuse source.tif reference.tif

Correct images matching *source\*.tif* to surface reflectance by fusion with *reference.tif*.  Use a 5 x 5 pixel kernel and the ``gain-blk-offset`` model for correction, and place corrected images in the *./corrected* directory:

.. code:: shell

    homonim fuse -k 5 5 -m gain-blk-offset -od ./corrected source*.tif reference.tif

Statistically compare *source.tif* and *corrected.tif* with *reference.tif*.

.. code:: shell

   homonim compare source.tif corrected.tif reference.tif


API
~~~

Example
^^^^^^^

Surface reflectance correction of an aerial image using a Sentinel-2 reference.

.. comment
    The code below is copied from docs/examples/api_example and # [*] comments removed

.. api_example_start

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

.. api_example_end

Reference imagery
~~~~~~~~~~~~~~~~~

`geedim <https://github.com/dugalh/geedim>`_ can be used as a companion tool for searching and downloading cloud-free reference imagery.   Alternatively, satellite imagery is available from a number of sources, including the `Google <https://developers.google.com/earth-engine/datasets>`_, `Amazon <https://aws.amazon.com/earth/>`_ and `Microsoft <https://planetarycomputer.microsoft.com/catalog>`_ repositories.


Usage
-----

See the documentation `here <https://homonim.readthedocs.io/>`_.

Terminology
-----------

``homonim`` is shorthand for *homogenise image* and is a reference to `the paper <https://www.researchgate.net/publication/328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data>`_ on which it is based.

Credits
-------

``homonim`` relies on these excellent packages:

-  `rasterio <https://github.com/rasterio/rasterio>`__
-  `opencv <https://github.com/opencv/opencv>`__
-  `numpy <https://github.com/numpy/numpy>`__

License
-------

``homonim`` is licensed under the terms of the `AGPLv3 <https://www.gnu.org/licenses/agpl-3.0.en.html>`__. This project is developed in collaboration with `InnovUS <https://www.innovus.co.za/>`__ at Stellenbosch University, alternative licenses can be arranged by `contacting <mailto:madeleink@sun.ac.za>`__ them.

Citation
--------

Please cite use of the code as:

-  Dugal Harris & Adriaan Van Niekerk (2019) Radiometric homogenisation of aerial images by calibrating with satellite data, *International Journal of Remote Sensing*, **40:7**, 2623-2647, DOI: https://doi.org/10.1080/01431161.2018.1528404.

Bibtex::

    @article{doi:10.1080/01431161.2018.1528404,
        author = {Dugal Harris and Adriaan Van Niekerk},
        title = {Radiometric homogenisation of aerial images by calibrating with satellite data},
        journal = {International Journal of Remote Sensing},
        volume = {40},
        number = {7},
        pages = {2623-2647},
        year  = {2019},
        publisher = {Taylor & Francis},
        doi = {10.1080/01431161.2018.1528404},
        URL = {https://doi.org/10.1080/01431161.2018.1528404},
    }


.. |Tests| image:: https://github.com/dugalh/homonim/actions/workflows/run-unit-tests.yml/badge.svg
   :target: https://github.com/dugalh/homonim/actions/workflows/run-unit-tests.yml
.. |codecov| image:: https://codecov.io/gh/dugalh/homonim/branch/main/graph/badge.svg?token=A01698K96C
   :target: https://codecov.io/gh/dugalh/homonim
.. |License: AGPL v3| image:: https://img.shields.io/badge/License-AGPL_v3-blue.svg
   :target: https://www.gnu.org/licenses/agpl-3.0
.. |PyPI version| image:: https://img.shields.io/pypi/v/homonim?color=blue
   :target: https://pypi.org/project/homonim/
.. |conda-forge version| image:: https://img.shields.io/conda/vn/conda-forge/homonim.svg?color=blue
   :alt: conda-forge
   :target: https://anaconda.org/conda-forge/homonim
.. |docs| image:: https://readthedocs.org/projects/homonim/badge/?version=latest
   :target: https://homonim.readthedocs.io/en/latest/?badge=latest
