from setuptools import setup

setup(name='batchgenerators',
      version='0.16',
      description='Awesome stuff for dank learning',
      url='http://github.com/storborg/funniest',
      author='Fabian Isensee',
      author_email='f.isensee@dkfz-heidelberg.de',
      license='MIT',
      packages=['batchgenerators', 'batchgenerators.augmentations', 'batchgenerators.generators', 'batchgenerators.examples'],
      zip_safe=False)