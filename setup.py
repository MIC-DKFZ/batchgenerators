from setuptools import setup

setup(name='batchgenerators',
      version='0.18.2',
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
      classifiers=[
          'Development Status :: 3 - Alpha',
          # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
          'Intended Audience :: Developers',  # Define that your audience are developers
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: MIT License',  # Again, pick a license
          'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 2.7',
      ],
      keywords=['deep learning', 'image segmentation', 'image classification', 'medical image analysis',
                  'medical image segmentation', 'data augmentation'],
      download_url="https://github.com/MIC-DKFZ/batchgenerators/archive/v0.18.1.tar.gz"
      )
