# homonim

Radiometric homogenisation of aerial and satellite imagery by fusion with satellite surface reflectance data.  

## Description

`homonim` corrects multi-spectral aerial and satellite imagery to approximate surface reflectance, by fusion with concurrent and collocated satellite surface reflectance data.  It is a form of *spectral harmonisation*, that  adjusts for spatially varying atmospheric and anisotropic (BRDF) effects, without the need for manual reflectance measurements, or target placements.  

It is useful as a pre-processing step for quantitative mapping applications, such as biomass estimation or precision agriculture, and can be applied to drone, aerial or satellite imagery.  

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
$ homonim fuse --method gain-blk-offset --kernel-shape 5 5 source.tif reference.tif 
```
Statistically compare an image, pre- and post-homogenisation, with a reference image:
```shell
$ homonim compare source.tif homogenised.tif reference.tif
```

## Example
Mosaics of 0.5 m resolution aerial imagery before and after homogenisation.  A Landsat-7 surface reflectance image was used as reference, and is shown in the background.  Homogenisation was performed using the `im-blk-offset` method and a 5 x 5 pixel kernel.  

<img src="data/readme_eg.jpg" data-canonical-src="data/readme_eg.jpg" alt="Before and after homogenisation" width="600"/>

## Usage
### Background
`homonim` uses a *reference* surface reflectance image to adjust a *source* image.  Typically the *reference* image is a satellite image at a coarser resolution that the *source* image. The surface reflectance relationship between *source*  and *reference* images is approximated with localised linear models.  Models are estimated for each pixel location inside a small rectangular kernel (window), using a fast DFT approach.  The homogenised output is produced by applying the model parameters to the source image.  

There are three methods for fitting the linear model:

 Method | Description
-------|------------
`gain` | A gain-only model, suitable for low haze *source* images, and *reference* images with zero offset.  
`gain-blk-offset` | The gain-only model with per-block offset normalisation.  The default and most robust method, suitable for most image combinations.
`gain-offset` | A gain and offset model.  The most accurate of the methods, but sensitive to differences between *source* and *reference* (e.g. shadows or land cover changes).  Suitable for well matched *source* and *reference* images. 


### Image preparation
To get started, a suitable *reference* image needs to be acquired.  For best results, the *reference* and *source* image(s) should be concurrent, co-located (accurately co-registered / orthorectified), and spectrally similar (with overlapping band spectral responses).

The *reference* image bounds should encompass those of the *source* image(s), and *source* / *reference* band ordering should match (i.e. reference band 1 corresponds to source band 1, reference band 2 corresponds to source band 2 etc).  [`rasterio`](https://rasterio.readthedocs.io/en/latest/cli.html) or [`gdal`](https://gdal.org/programs/index.html) command line tools can be used to clip images, re-order bands etc. as necessary.  These packages are included in the `homonim` installation.  

Examples of suitable surface reflectance image collections for the *reference* image are those produced by Landsat, Sentinel-2 and MODIS.  There are a number of platforms, and associated tools, for acquiring these images, including the [Google](https://developers.google.com/earth-engine/datasets) and [Amazon](https://aws.amazon.com/earth/) repositories.  

[`geedim`](https://github.com/dugalh/geedim) is intended as a companion tool to `homonim`, and allows search, basic cloud/shadow-free compositing, and download of [Google Earth Engine](https://developers.google.com/earth-engine/datasets) surface reflectance imagery.  More details [here](https://github.com/dugalh/geedim).

Where possible, `homonim` should be applied to raw  *source* imagery (i.e. without colour-balancing or gamma-correction etc.), to satisfy the assumptions of the method.  This is not strictly necessary however, and adjusted *source* images will still benefit from homogenisation.  

## Command-line interface
homonim functionality is accessed through the command line sub-commands: `fuse`, `compare` and `stats`.   
```
homonim --help
```
```
Usage: homonim [OPTIONS] COMMAND [ARGS]...

Options:
  -v, --verbose  Increase verbosity.
  -q, --quiet    Decrease verbosity.
  --help         Show this message and exit.

Commands:
  compare  Compare image(s) with a reference.
  fuse     Radiometrically homogenise image(s) by fusion with a reference.
  stats    Print parameter image statistics.
```
### fuse
The main `homonim` command for homogenising a *source* image with a *reference*.  There are numerous options, with all having sensible default values.  For most situations, the user only needs to specify  
```
homonim fuse --help
```
```
Usage: homonim fuse [OPTIONS] INPUTS... REFERENCE

  Radiometrically homogenise image(s) by fusion with a reference.

  INPUTS      Path(s) to source image(s) to be homogenised.

  REFERENCE   Path to a surface reflectance reference image.

  Reference image extents should encompass those of the source image(s), and
  source / reference band ordering should match (i.e. reference band 1
  corresponds to source band 1, reference band 2 corresponds to source band 2
  etc).

  For best results, the reference and source image(s) should be concurrent,
  co-located (accurately co-registered / orthorectified), and spectrally
  similar (with overlapping band spectral responses).

  Examples:
  ---------

  Homogenise 'input.tif' with 'reference.tif', using the 'gain-blk-offset'
  method, and a kernel of 5 x 5 pixels.

      $ homonim fuse -m gain-blk-offset -k 5 5 input.tif reference.tif

  Homogenise files matching 'input*.tif' with 'reference.tif', using the
  'gain-offset' method and a kernel of 15 x 15 pixels. Place homogenised files
  in the './homog' directory, produce parameter images, and mask partially
  covered pixels in the homogenised images.

      $ homonim fuse -m gain-offset -k 15 15 -od ./homog --param-image
        --mask-partial input*.tif reference.tif

Options:
  -m, --method [gain|gain-blk-offset|gain-offset]
                                  Homogenisation method.  [default:
                                  gain_blk_offset]
  -k, --kernel-shape <HEIGHT WIDTH>
                                  Kernel height and width in pixels (of the
                                  the lowest resolution of the source and
                                  reference images).  [default: 5, 5]
  -od, --output-dir DIRECTORY     Directory in which to create homogenised
                                  image(s). [default: use source image
                                  directory]
  -ovw, --overwrite               Overwrite existing output file(s).
                                  [default: False]
  -cmp, --compare                 Statistically compare source and homogenised
                                  images with the reference.
  -nbo, --no-build-ovw            Turn off overview building for the
                                  homogenised image(s).
  -pc, --proc-crs [auto|src|ref]  The image CRS in which to perform
                                  processing.  [default: auto]
  -c, --conf FILE                 Path to an optional configuration file.
  -pi, --param-image              Create a debug image, containing model
                                  parameters and R² values, for each source
                                  file.
  -mp, --mask-partial             Mask homogenised pixels produced from
                                  partial kernel or image coverage.
  -t, --threads INTEGER           Number of image blocks to process
                                  concurrently (0 = use all cpus).  [default:
                                  8]
  -mbm, --max-block-mem FLOAT     Maximum image block size for concurrent
                                  processing (MB)  [default: 100]
  -ds, --downsampling [nearest|bilinear|cubic|cubic_spline|lanczos|average|mode|max|min|med|q1|q3|sum|rms]
                                  Resampling method for downsampling.
                                  [default: average]
  -us, --upsampling [nearest|bilinear|cubic|cubic_spline|lanczos|average|mode|max|min|med|q1|q3|sum|rms]
                                  Resampling method for upsampling.  [default:
                                  cubic_spline]
  -rit, --r2-inpaint-thresh FLOAT 0-1
                                  R² threshold below which to inpaint model
                                  parameters from surrounding areas. For
                                  'gain-offset' method only.  [default: 0.25;
                                  0<=x<=1]
  --out-driver TEXT               Output format driver.  [default: GTiff]
  --out-dtype [uint8|uint16|int16|uint32|int32|float32|float64]
                                  Output image data type.  [default: float32]
  --out-nodata [NUMBER|null|nan]  Output image nodata value.  [default: nan]
  -co, --out-profile NAME=VALUE   Driver specific creation options.  See the
                                  rasterio documentation for more information.
  --help                          Show this message and exit.
```
### compare
### stats

## Terminology
While `homonim` implements a form of *spectral harmonisation*, we have used the term *homogenisation* to describe the method, in keeping with the [original formulation](https://www.researchgate.net/publication/328317307_Radiometric_homogenisation_of_aerial_images_by_calibrating_with_satellite_data).

## License
`homonim` is licensed under the terms of the [AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html).  This project is developed in collaboration with [InnovUS](https://www.innovus.co.za/) at Stellenbosch University, alternative licenses can be arranged by [contacting](mailto:sjdewet@sun.ac.za) them.

## Citation
Please cite use of the code as: 
- Harris, D., Van Niekerk, A., 2019. Radiometric homogenisation of aerial images by calibrating with satellite data. *Int. J. Remote Sens.* **40**, 2623–2647. [https://doi.org/10.1080/01431161.2018.1528404](https://doi.org/10.1080/01431161.2018.1528404). 

## Author
**Dugal Harris** - [dugalh@gmail.com](mailto:dugalh@gmail.com)
