|Tests| |codecov| |PyPI version| |conda-forge version| |docs| |License: AGPL v3|

homonim
=======

.. short_descr_start

Correct aerial and satellite imagery to surface reflectance.

.. short_descr_end

.. description_start

Description
-----------

``homonim`` provides a command line interface and API for correcting remotely sensed imagery to approximate surface
reflectance.  It is a form of *spectral harmonisation*, that adjusts for spatially varying atmospheric and anisotropic
(BRDF) effects, by fusion with satellite surface reflectance data.  Manual reflectance measurements and target
placements are not required.

It is useful as a pre-processing step for quantitative mapping applications such as biomass estimation or
precision agriculture, and for reducing seamlines and other visual artefacts in image mosaics.  It can be applied to
multi-spectral drone, aerial and satellite imagery.

``homonim`` is based on the method described in the paper:
`Radiometric homogenisation of aerial images by calibrating with satellite data <https://www.researchgate.net/publication/328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data>`__.

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

Quick Start
-----------

Download the ``homonim`` github repository to get the test imagery. If you have ``git``, you can clone it with:

.. code:: shell

   git clone https://github.com/dugalh/homonim.git

Alternatively, download it from `here <https://github.com/dugalh/homonim/archive/refs/heads/main.zip>`__, extract the
zip archive and rename the *homonim-main* directory to *homonim*.

Using the ``gain-blk-offset`` model and a 5 x 5 pixel kernel, correct the aerial images with the Sentinel-2
reference.

.. code:: shell

   homonim fuse -m gain-blk-offset -k 5 5 -od . ./homonim/tests/data/source/*_RGB.tif ./homonim/tests/data/reference/COPERNICUS-S2-20151003T075826_20151003T082014_T35HKC_B432_Byte.tif

Statistically compare the raw and corrected aerial images with the included Landsat-8 reference.

.. code:: shell

   homonim compare ./homonim/tests/data/source/*_RGB.tif ./*FUSE*.tif ./homonim/tests/data/reference/LANDSAT-LC08-C02-T1_L2-LC08_171083_20150923_B432_Byte.tif


Example
-------

Mosaics of 0.5 m resolution aerial imagery before and after correction. A Landsat-7 surface reflectance image was
used as reference, and is shown in the background. Correction was performed using the ``gain-blk-offset`` model and
a 5 x 5 pixel kernel.

.. image:: https://raw.githubusercontent.com/dugalh/homonim/main/data/readme_eg.jpg
   :alt: example


Usage
-----

See the documentation `here <https://homonim.readthedocs.io/>`__.

Terminology
-----------

``homonim`` is shorthand for *homogenise image* and is a reference to `the paper <https://www.researchgate
.net/publication/328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data>`_ on which
it is based.

Credits
-------

``homonim`` makes extensive use of the following excellent projects:

-  `rasterio <https://github.com/rasterio/rasterio>`__
-  `opencv <https://github.com/opencv/opencv>`__
-  `numpy <https://github.com/numpy/numpy>`__

License
-------

``homonim`` is licensed under the terms of the `AGPLv3 <https://www.gnu.org/licenses/agpl-3.0.en.html>`__. This project
is developed in collaboration with `InnovUS <https://www.innovus.co.za/>`__ at Stellenbosch University, alternative
licenses can be arranged by `contacting <mailto:sjdewet@sun.ac.za>`__ them.

Citation
--------

Please cite use of the code as: - Harris, D., Van Niekerk, A., 2019. Radiometric homogenisation of aerial images by
calibrating with satellite data. *Int. J. Remote Sens.* **40**, 2623–2647.
https://doi.org/10.1080/01431161.2018.1528404.

Author
------

**Dugal Harris** - dugalh@gmail.com

.. |Tests| image:: https://github.com/dugalh/homonim/actions/workflows/run-unit-tests.yml/badge.svg
   :target: https://github.com/dugalh/homonim/actions/workflows/run-unit-tests.yml
.. |codecov| image:: https://codecov.io/gh/dugalh/homonim/branch/main/graph/badge.svg?token=A01698K96C
   :target: https://codecov.io/gh/dugalh/homonim
.. |License: AGPL v3| image:: https://img.shields.io/badge/License-AGPL_v3-blue.svg
   :target: https://www.gnu.org/licenses/agpl-3.0
.. |PyPI version| image:: https://badge.fury.io/py/homonim.svg
   :target: https://badge.fury.io/py/homonim
.. |conda-forge version| image:: https://img.shields.io/conda/vn/conda-forge/homonim.svg
   :alt: conda-forge
   :target: https://anaconda.org/conda-forge/homonim
.. |docs| image:: https://readthedocs.org/projects/homonim/badge/?version=latest
   :target: https://homonim.readthedocs.io/en/latest/?badge=latest
