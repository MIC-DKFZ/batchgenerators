##How to install locally

Install dependencies
```
pip install numpy scipy nilearn matplotlib scikit-image nibabel
```

Install dldabg
```
git clone https://phabricator.mitk.org/source/dldabg.git
cd dldabg
pip install -e .
```

Using `-e` will make pip use a symlink to the source. So when you pull the newest changes of the repo your pip installation will automatically use the newest code.

Import as follows
```
from DeepLearningBatchGeneratorUtils.MultiThreadedGenerator import MultiThreadedGenerator
```

For how to use it see `DeepLearningBatchGeneratorUtils/examples.py`.