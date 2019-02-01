# batchgenerators by MIC@DKFZ
batchgenerators is a python package that we developed at the Division of Medical Image Computing at the German Cancer
Research Center (DKFZ) to suit all our deep learning data augmentation needs.
It is not (yet) perfect, but we feel it is good enough to be shared with the community. If you encounter bug, feel free
to contact us or open a github issue.

[![Build Status](https://travis-ci.org/MIC-DKFZ/batchgenerators.svg?branch=master)](https://travis-ci.org/MIC-DKFZ/batchgenerators)

## Windows is not (yet) supported!!
Batchgenerators makes heavy use of python multiprocessing and python multiprocessing on windows is a problem. 
We are trying to find a solution but as of now batchgenerators won't work on Windows!

### Important!
Starting from version 1.14.6 numpy has issues with multiprocessing. Mutrix multiplications (which we are using 
to rotate coordinate systems for data augmentation) now run mutlithreaded on all available threads. 
This can cause chaos if you are using a multiprocessing pipeline, beacause each background worker will spawn a lot of 
threads to do the matrix multiplication (8 workers on a 16 Core machine = up to 8*16=256 threads. duh.). There is nothing we (dkfz devs) can do to 
tackle that problem, but this will only be a real issue in very specific configurations of data augmentation. If you 
notice unnecessarily high CPU load, downgrade numpy to 1.14.5 (pip install numpy==1.14.5) to solve the issue (or try OMP_NUM_THREADS=1). 
Numpy devs are aware of this problem and trying to find a solution (see https://github.com/numpy/numpy/issues/11826#issuecomment-425087772)

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

## How to install locally

Install dependencies (some of them are only needed for certain functionalities)
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

## Release Notes

- 0.18:
    - all augmentations (there are some exceptions though) are implemented on a per-sample basis. This should make it 
    easier to use the augmentations outside of the Transforms of batchgenerators 
    - applicable Transforms now have a keyword p_per_sample with which the user can specify a probability with which this
     transform is applied to a sample. Before, this was handled by RndTransform and applied to the whole batch (so 
     either all samples were augmented or none). Now this decision is made on a per-sample basis and increases 
     variability by a lot.
    - following the previous point, RndTransform is now deprecated
    - AlternativeMultiThreadedAugmenter is now deprecated as well (no need to have this anymore)
    - pytorch users can now transform numpy arrays to pytorch tensors within batchgenerators (NumpyToTensor). For some 
    reason, inter-process communication is faster with tensors (~factor 4), so this is recommended!
    - if numpy arrays were converted to pytorch tensors, MultithreadedAugmenter now allows to pin the memory as well 
    (pin_memory=True). This will happen in a background thread (inspired by pytorch DataLoader). pinned memory can be 
    copied to the GPU much faster. My (Fabian) classification experiment with Resnet50 got a speed boost of 12% from just 
    that.
