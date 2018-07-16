# batchgenerators by MIC@DKFZ
batchgenerators is a python package that we developed at the Division of Medical Image Computing at the German Cancer
Research Center (DKFZ) to suit all our deep learning data augmentation needs.
It is not (yet) perfect, but we feel it is good enough to be shared with the community. If you encounter bug, feel free
to contact us or open a github issue.


## Suported Augmentations
We supports a variety of augmentations, all of which are compatible with **2D and 3D input data**! (This is something
that was missing in most other frameworks).

* **Spatial Augmentations**
  * mirroring
  * channel translation (to simulate registration errors)
  * elastic deformations
  * rotations
  * scaling
  * resampling
* **Color Augmentations**
  * brightness (additive, multiplivative)
  * contrast
  * gamma (like gamma correction in photo editing)
* **Noise Augmentations**
  * Gaussian Noise
  * Rician Noise
  * ...will be expanded in future commits
* **Cropping**
  * random crop
  * center crop
  * padding

Note: Stack transforms by using batchgenerators.transforms.abstract_transforms.Compose. Finish it up by plugging the
composed transform into our **multithreader**: batchgenerators.dataloading.multi_threaded_augmenter.MultiThreadedAugmenter


## How to use it
The working principle is simple: Derive from DataLoaderBase class, reimplement generate_train_batch member function and
use it to stack your augmentations!
For an example see `batchgenerators/examples/example_ipynb.ipynb`


## Data Structure
The data structure that is used internally (and with which you have to comply when implementing generate_train_batch)
is kept simple as well: It is just a regular python dictionary! We did this to allow maximum flexibility in the kind of
data that is passed along through the pipeline. The dictionary must have a 'data' key:value pair. It optionally can
handle a 'seg' key:vlaue pair to hold a segmentation. If a 'seg' key:value pair is present all spatial transformations
will also be applied to the segmentation! A part from 'data' and 'seg' you are free to do whatever you want (your image
classification/regression target for example). All key:value pairs other than 'data' and 'seg' will be passed through the
pipeline unmodified.

'data' value must have shape (b, c, x, y) for 2D or shape (b, c, x, y, z) for 3D!
'seg' value must have shape (b, c, x, y) for 2D or shape (b, c, x, y, z) for 3D! Color channel may be used here to
allow for several segmentation maps. If you have only one segmentation, make sure to have shape (b, 1, x, y (, z))

##How to install locally

Install dependencies
```
pip install numpy scipy nilearn matplotlib scikit-image nibabel
```

Install batchgenerators
```
git clone https://github.com/MIC-DKFZ/batchgenerators
cd batchgenerators
pip install -e .
```

Using `-e` will make pip use a symlink to the source. So when you pull the newest changes of the repo your pip
installation will automatically use the newest code. If not using -e, using --upgrade is recommended because we may push
changes/bugfixes without changing the version number.

Import as follows
```
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
```

Note: This package also includes 'generators'. Support for those will be dropped in the future. That was our old design.