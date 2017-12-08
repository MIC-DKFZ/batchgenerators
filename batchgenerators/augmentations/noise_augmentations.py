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
from scipy.ndimage import gaussian_filter

def augment_rician_noise(data, noise_variance=(0, 0.1)):
    for sample_idx in range(data.shape[0]):
        sample = data[sample_idx]
        variance = random.uniform(noise_variance[0], noise_variance[1])
        sample = np.sqrt(
            (sample + np.random.normal(0.0, variance, size=sample.shape)) ** 2 +
             np.random.normal(0.0, variance, size=sample.shape) ** 2)
        data[sample_idx] = sample
    return data

def augment_gaussian_noise(data, noise_variance=(0, 0.1)):
    for sample_idx in range(data.shape[0]):
        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]
        else:
            variance = random.uniform(noise_variance[0], noise_variance[1])
        data[sample_idx] = data[sample_idx] + np.random.normal(0.0, variance, size=data[sample_idx].shape)
    return data


def augment_gaussian_blur(data, sigma_range, per_channel=True):
    for sample_idx in range(data.shape[0]):
        if not per_channel:
            sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        for c in range(data.shape[1]):
            if per_channel:
                sigma = np.random.uniform(sigma_range[0], sigma_range[1])
            data[sample_idx, c] = gaussian_filter(data[sample_idx, c], sigma, order=0)
    return data