# TessellationService
TessellationService is a python script, intended to extract annotated image regions by the [AnnotationService](https://github.com/SasNaw/AnnotationService), to create a training sample for neural networks.

It supports all image formats supported by [OpenSlide](http://openslide.org/) and additionally [Deep Zoom Images](https://msdn.microsoft.com/en-us/library/cc645077(v=vs.95).aspx).

## Installation
    $ git clone https://github.com/SasNaw/TessellationService.git

## Dependencies
- Python installation of [OpenSlide](http://openslide.org/download/)
- [NumPy](http://www.scipy.org/scipylib/download.html)
- [OpenCV](http://docs.opencv.org/2.4/index.html)

## How to run:
When saving made annotations, the AnnotationService creates a JSON file ([image name].[ext].json) next to the annotated image ([AnnotationService directory]/static/wsi/[[path to img/] image]). This file, together with the corresponding image, is used as input for the TessellationService:

	$ python TessellationService [path to JSON file created by the AnnotationService]
	
3 modes of extraction are supported:
* extraction of region's bounding box in original size
* extraction of region's bounding box scaled to a provided [width] and [height] pixels (-r [width] [height])
* approximation of region via multiple tiles of [width] x [height] pixels (-t [width] [height])

For an overview of all parameters see

	$ python TessellationService -h
