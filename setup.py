from setuptools import setup

setup(name='batchgenerators',
      version='0.18',
      description='Data augmentation toolkit',
      url='https://github.com/MIC-DKFZ/batchgenerators',
      author='Division of Medical Image Computing, German Cancer Research Center',
      author_email='f.isensee@dkfz-heidelberg.de',
      license='Apache License Version 2.0, January 2004',
      packages=['batchgenerators', 'batchgenerators.augmentations',
      'batchgenerators.examples', 'batchgenerators.transforms', 'batchgenerators.dataloading'],
      install_requires=[
            "numpy>=1.10.2",
            "scipy",
            "scikit-image",
            "future",
            "unittest2"
      ],
      zip_safe=False)