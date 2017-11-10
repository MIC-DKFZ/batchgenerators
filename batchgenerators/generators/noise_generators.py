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


from builtins import range

import numpy as np
import random
from warnings import warn

from batchgenerators.augmentations.noise_augmentations import augment_gaussian_noise
from batchgenerators.augmentations.noise_augmentations import augment_rician_noise


def rician_noise_generator(generator, noise_variance=(0, 0.1)):
    '''
    Adds rician noise with the given variance.
    The Noise of MRI data tends to have a rician distribution: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2254141/
    '''
    for data_dict in generator:
        assert "data" in list(
            data_dict.keys()), "your data generator needs to return a python dictionary with at least a 'data' key value pair"

        data_dict["data"] = augment_rician_noise(data_dict['data'], noise_variance=noise_variance)
        yield data_dict


def gaussian_noise_generator(generator, noise_variance=(0, 0.1)):
    warn("using deprecated generator center_crop_seg_generator", Warning)
    '''
    Adds gaussian noise with the given variance.

    '''
    for data_dict in generator:
        assert "data" in list(
            data_dict.keys()), "your data generator needs to return a python dictionary with at least a 'data' key value pair"

        data_dict["data"] = augment_gaussian_noise(data_dict['data'], noise_variance=noise_variance)
        yield data_dict

