# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_noise
from batchgenerators.augmentations.noise_augmentations import augment_rician_noise


class RicianNoiseTransform(AbstractTransform):
    """Adds rician noise with the given variance.
    The Noise of MRI data tends to have a rician distribution: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2254141/

    Args:
        noise_variance (tuple of float): samples variance of Gaussian distribution used to calculate
        the rician distribution from this interval

    CAREFUL: This transform will modify the value range of your data!
    """
    def __init__(self, noise_variance=(0, 0.1)):
        self.noise_variance = noise_variance

    def __call__(self, **data_dict):
        data_dict["data"] = augment_rician_noise(data_dict['data'], noise_variance=self.noise_variance)
        return data_dict


class GaussianNoiseTransform(AbstractTransform):
    """Adds additive Gaussian Noise

    Args:
        noise_variance (tuple of float): samples variance of Gaussian distribution from this interval

    CAREFUL: This transform will modify the value range of your data!
    """
    def __init__(self, noise_variance=(0, 0.1)):
        self.noise_variance = noise_variance

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        data = augment_gaussian_noise(data, self.noise_variance)
        data_dict["data"] = data
        return data_dict


