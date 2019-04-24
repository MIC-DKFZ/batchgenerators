from setuptools import setup

setup(name='batchgenerators',
      version='0.19.3',
      description='Data augmentation toolkit',
      url='https://github.com/MIC-DKFZ/batchgenerators',
      author='Division of Medical Image Computing, German Cancer Research Center',
      author_email='f.isensee@dkfz-heidelberg.de',
      license='Apache License Version 2.0, January 2004',
      packages=['batchgenerators', 'batchgenerators.augmentations',
      'batchgenerators.examples', 'batchgenerators.transforms', 'batchgenerators.dataloading',
                'batchgenerators.utilities'],
      install_requires=[
            "numpy>=1.10.2",
            "scipy",
            "scikit-image",
            "scikit-learn",
            "future",
            "unittest2"
      ],
      keywords=['deep learning', 'image segmentation', 'image classification', 'medical image analysis',
                  'medical image segmentation', 'data augmentation'],
      download_url="https://github.com/MIC-DKFZ/batchgenerators/archive/v0.18.1.tar.gz"
      )
