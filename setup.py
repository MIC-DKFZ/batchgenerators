from setuptools import setup, find_packages

setup(name='batchgenerators',
      version='0.25',
      description='Data augmentation toolkit',
      url='https://github.com/MIC-DKFZ/batchgenerators',
      author='Division of Medical Image Computing, German Cancer Research Center AND Applied Computer Vision Lab, '
             'Helmholtz Imaging Platform',
      author_email='f.isensee@dkfz-heidelberg.de',
      license='Apache License Version 2.0, January 2004',
      packages=find_packages(exclude=["tests"]),
      install_requires=[
            "numpy>=1.10.2",
            "scipy",
            "scikit-image",
            "scikit-learn",
            "threadpoolctl",
            "pandas"
      ],
      keywords=['data augmentation', 'deep learning', 'image segmentation', 'image classification',
                'medical image analysis', 'medical image segmentation'],
      )
