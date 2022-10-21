Aerial mosaic correction
========================

This case study is an abbreviation of the `original presentation of the method <https://www.researchgate.net/publication/328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data>`_.  It demonstrates the correction and accuracy improvement of a large mosaic of `NGI <https://ngi.dalrrd.gov.za/index.php/what-we-do/aerial-photography-and-imagery>`_ aerial images.

The mosaic consists of ± 2000 mages, captured over the Little Karoo (South Africa) from 22 Jan to 8 Feb 2010.  The images have a 50 cm spatial resolution and 4 spectral bands (red, green, blue and near-infrared).

.. figure:: aerial_mosaic-study_area.png
    :scale: 50 %
    :align: center


Correction
----------

A `MODIS NBAR <https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD43A4>`_ image was chosen for the reference.  Availability of images from other satellites was limited for the source capture dates.  The spatial resolution of the image is low (500 m), but it satisfies the recommendations discussed in the :ref:`background <reference_image>`, and so makes a reasonable choice.

The *source* aerial mosaic is overlaid on the MODIS reference in the figure below.  Seamlines between images, "hot spots", and other colour variations resulting from atmospheric and BRDF effects are clearly visible.

.. _source-mosaic:

.. figure:: aerial_mosaic-source_mosaic.jpg
    :width: 80%
    :align: center

    **Source mosaic**

Correction was performed with the *gain* model and a kernel of 1 pixel.  The small kernel was chosen to mitigate the effect of large (500 m) MODIS pixels.  The next figure shows the corrected mosaic overlaid on the MODIS reference.

.. figure:: aerial_mosaic-corrected_mosaic.jpg
    :width: 80%
    :align: center

    **Corrected mosaic**

There is a clear improvement from the source mosaic.  Seamlines and other variations are no longer visible, and there is a good match between the corrected images and the MODIS backdrop.

Evaluation
----------

A simple way of evaluating the relative improvement in surface reflectance accuracy, is to compare the source and corrected mosaics with a reference image.  Rather than compare with the MODIS NBAR reference (which was used for fitting the correction models), the mosaics were compared with an "independent" SPOT-5 image.  This 10 m resolution SPOT-5 image covered a portion of the study area.

.. image:: aerial_mosaic-spot5_extent.jpg
    :width: 50 %
    :align: center

|

After correcting the SPOT-5 image to surface reflectance with ATCOR-3, it was compared to the source and corrected mosaics.  SPOT-5 does not have a blue band, so this was omitted from the comparison.

.. figure:: aerial_mosaic-source_spot5_scatter.png
    :align: center

    **Source - reference correlation**

.. figure:: aerial_mosaic-corrected_spot5_scatter.png
    :align: center

    **Corrected - reference correlation**

The scatter plots and *r*:sup:`2` values show a sizeable improvement after correction.  Further details and discussion on this example can be found in the `paper <https://www.researchgate.net/publication/328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data>`_
