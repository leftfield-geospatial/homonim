Classification
==============

This case study uses aerial images and ground truth data from a `vegetation classification study <https://www.researchgate.net/publication/329137175_Regional_mapping_of_spekboom_canopy_cover_using_very_high_resolution_aerial_imagery>`_ to show how ``homonim`` can improve classifier performance.

There are 4 aerial images with 50 cm spatial resolution and 4 spectral bands (red, green, blue and near-infrared).  Aerial imagery was supplied by `NGI <https://ngi.dalrrd.gov.za/index.php/what-we-do/aerial-photography-and-imagery>`_.  Ground truth data consists of 161 polygons with labels for 3 vegetation classes:

- **Spekboom**: a species of succulent shrub.
- **Tree**: woody trees.
- **Background**: other vegetation, bare ground etc not included in the other classes.

Some example polygons are shown below, overlaid on an aerial image.

.. figure:: classification-groundtruth_polygons.jpg
    :width: 50 %
    :align: center

    **Class polygons**

The aerial images were corrected with ``homonim`` using the *gain* model, a 1 x 1 pixel kernel, and a `MODIS NBAR <https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD43A4>`_ reference image.  Class densities (Gaussian kernel density estimates) are shown before and after correction, for the blue and NIR bands below.

.. figure:: classification-spectral_kde.jpg
    :align: center

    **Class densities**

The *spekboom* and *tree* classes appear more compact, and better separated after correction.  That said, it is difficult to say anything conclusively from this figure.

To quantify the effect of surface reflectance correction, a per-pixel naive Bayes classifier was evaluated on the source and corrected imagery.  (Evaluation used the raw red, green, blue and NIR band pixel values as features, and a 10-fold cross-validation for training and testing.)  Normalised confusion matrix, accuracy, and AUC (area under the ROC curve) values are tabulated below.

+----------------+-----------------------------------------------------+----------+------+
|                | Confusion matrix                                    | Accuracy | AUC  |
+================+=====================================================+==========+======+
| **Source**     |  +----------------+------------+----------+------+  | 58.41 %  | 0.73 |
|                |  |                | Background | Spekboom | Tree |  |          |      |
|                |  +================+============+==========+======+  |          |      |
|                |  | **Background** |       0.43 |     0.13 | 0.43 |  |          |      |
|                |  +----------------+------------+----------+------+  |          |      |
|                |  | **Spekboom**   |       0.09 |     0.62 | 0.29 |  |          |      |
|                |  +----------------+------------+----------+------+  |          |      |
|                |  | **Tree**       |       0.07 |     0.23 | 0.70 |  |          |      |
|                |  +----------------+------------+----------+------+  |          |      |
+----------------+-----------------------------------------------------+----------+------+
| **Corrected**  |  +----------------+------------+----------+------+  | 67.47 %  | 0.81 |
|                |  |                | Background | Spekboom | Tree |  |          |      |
|                |  +================+============+==========+======+  |          |      |
|                |  | **Background** |       0.56 |     0.17 | 0.28 |  |          |      |
|                |  +----------------+------------+----------+------+  |          |      |
|                |  | **Spekboom**   |       0.07 |     0.72 | 0.21 |  |          |      |
|                |  +----------------+------------+----------+------+  |          |      |
|                |  | **Tree**       |       0.04 |     0.21 | 0.74 |  |          |      |
|                |  +----------------+------------+----------+------+  |          |      |
+----------------+-----------------------------------------------------+----------+------+

The improvement in accuracy after correction implies that the corrected surface reflectance is less noisy, and better at describing the vegetation classes.  This case study demonstrates the benefits of pre-processing with ``homonim`` in multi-spectral classification problems.
