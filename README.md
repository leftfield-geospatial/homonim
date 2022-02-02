# homonim

Radiometric homogenisation of aerial and satellite imagery by fusion with satellite surface reflectance data.  

## Description

`homonim` corrects multi-spectral aerial and satellite imagery to approximate surface reflectance, by fusion with concurrent and collocated satellite surface reflectance data.  It is a form of *spectral harmonisation*, that  adjusts for spatially varying atmospheric and anisotropic (BRDF) effects, without the need for manual reflectance measurements, or target placements.  

It is useful as a pre-processing step for quantitative mapping applications, such as biomass estimation or precision agriculture, and  can be applied to drone, aerial or satellite imagery.  

`homonim` is based on the method described in [*Radiometric homogenisation of aerial images by calibrating with satellite data*](https://www.researchgate.net/publication/328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data).

## Installation
TBD
<!--`homonim` is available as a python 3 package, via `pip` and `conda`.  Under Windows, we recommend using `conda` to simplify the installation of binary dependencies.  The [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installation provides a minimal `conda`.
### conda
```shell
$ conda install -c conda-forge homonim
```
### pip
```shell
$ pip install homonim
```-->
## Quick Start
Homogenise an image with a reference, using the `gain-blk-offset` method, a sliding kernel of 5x5 pixels:
```shell
$ homonim fuse --method gain-blk-offset --kernel-shape 5 5 <path to source image> <path to reference image> 
```
Statistically compare an image, pre- and post-homogenisation, with a reference image:
```shell
$ homonim compare <path to source image> <path to homogenised image> <path to reference image>
```

## Example
Mosaics of 0.5 m resolution aerial imagery before and after homogenisation.  A Landsat-7 surface reflectance image was used as reference, and is shown in the background.  Homogenisation was performed using the `im-blk-offset` method and a 5 x 5 pixel kernel.  

<img src="data/readme_eg.jpeg" data-canonical-src="data/readme_eg.jpg" alt="Before and after homogenisation" width="800"/>

## Usage
### Terminology
`homonim` uses a ***reference*** image to homogenise a ***source*** image.  
Modelling is performed in the lowest resolution of the source and reference CRSs (co-ordinate reference systems) 
While `homonim` implements a form of *spectral harmonisation*, we have used the term *homogenisation* to describe the method, in keeping with the [original formulation](https://www.researchgate.net/publication/328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data).
### Background
`homonim` approximates the surface reflectance relationship between *source*  and *reference* images, with localised linear models.  Models are estimated for each pixel location inside a small rectangular kernel (window), using a fast DFT approach.  The homogenised output is produced by applying the model parameters to the source image.  

### Image preparation
`homonim` uses a *reference* image to homogenise a *source* image.  The *reference* would typically be a satellite surface reflectance image,  (typically, a raw/unadjusted drone, aerial or satellite image).  Examples of suitable *reference* image collections are those produced by Landsat, Sentinel-1/2 and MODIS.  For best results, the *source* should be a raw or top-of-atmosphere (TOA) type image, without modifications (e.g. colour-balancing or gamma-correction).  

The *reference* and *source* image pair should be:
- Co-located:
  - The bounds of the *reference* image must encompass those of the *source*.
  - The image pair should be accurately co-registered / orthorectified.
- Concurrent:
  - Images should have similar capture dates, with a minimum of land cover or phenological change between them.
- Spectrally similar:
  - `homonim` expects the ordering of the bands in *source* and *reference* to match i.e. *reference* band 1 corresponds to *source* band 1, *reference* band 2 corresponds to *source* band 2 etc
  - The *reference* should be chosen to have similar bands to the *source*.
   
  
#### Co-located
#### Spectrally similar


### fuse
Typically, model parameters are estimated in the lowest resolution of the source and reference image grids, requiring one of the images to be re-projected into the CRS (coordinate reference system) of the other.  This is termed the *processing CRS*.  Where the *processing CRS* is that of the reference image, estimated model parameters are re-projected back to the source image CRS .  
### compare
### stats


## License
`homonim` is licensed under the terms of the [AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html).  This project is developed in collaboration with [InnovUS](https://www.innovus.co.za/) at Stellenbosch University, alternative licenses can be arranged by [contacting](mailto:sjdewet@sun.ac.za) them.

## Citation
Please cite use of the code as: 
- Harris, D., Van Niekerk, A., 2019. Radiometric homogenisation of aerial images by calibrating with satellite data. *Int. J. Remote Sens.* **40**, 2623–2647. [https://doi.org/10.1080/01431161.2018.1528404](https://doi.org/10.1080/01431161.2018.1528404). 

## Author
**Dugal Harris** - [dugalh@gmail.com](mailto:dugalh@gmail.com)
