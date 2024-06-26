Regression modelling
====================

``homonim`` can improve accuracy and consistency in quantitative remote sensing applications.  This case study demonstrates the use of ``homonim`` to improve the relevance of image features for modelling aboveground carbon (AGC).  The images and ground truth data are taken from an `AGC mapping study <https://github.com/leftfield-geospatial/map-thicket-agc>`_.

Correction
----------

A small mosaic of 4 `NGI <https://ngi.dalrrd.gov.za/index.php/what-we-do/aerial-photography-and-imagery>`_ aerial images covering the study site were corrected to surface reflectance using ``homonim`` and a Sentinel-2 reference.  The *gain_blk_offset* :ref:`model <background:model>` and a :ref:`kernel shape <background:kernel shape>` of 15 x 15 pixels produced the best performance.  AGC ground truth data for 85 plots are overlaid on the corrected mosaic below.

.. figure:: regression_modelling-agc_map.jpg
    :align: center
    :width: 80%

Evaluation
----------

For this problem, `NDVI <https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index>`_ is reasonably good predictor of AGC.  The next figure shows the correlation between NDVI and AGC in each ground truth plot, before and after correction to surface reflectance.

.. figure:: regression_modelling-eval.png

The comparison gives an indication of the improvement in the predictive power the imagery.  Even though source image variations due to BRDF and atmospheric effects are small, correction to surface reflectance strengthens the NDVI - AGC correlation.

.. note::
    The figures in this case study are generated by the `regression modelling tutorial <../tutorials/regression_modelling.ipynb>`_.

    Ground truth data is licensed under the terms of the `CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>`_.  More details on the data and original study can be found in the related `github repository <https://github.com/leftfield-geospatial/map-thicket-agc>`_ and `paper <https://www.researchgate.net/publication/353313021_Very_high_resolution_aboveground_carbon_mapping_in_subtropical_thicket>`_.
