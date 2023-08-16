# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
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

import random
from typing import Tuple

import numpy as np
from batchgenerators.augmentations.utils import get_range_val, mask_random_squares
from builtins import range
from scipy.ndimage import gaussian_filter


def augment_rician_noise(data_sample, noise_variance=(0, 0.1)):
    variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = np.sqrt(
        (data_sample + np.random.normal(0.0, variance, size=data_sample.shape)) ** 2 +
        np.random.normal(0.0, variance, size=data_sample.shape) ** 2) * np.sign(data_sample)
    return data_sample


def setup_augment_gaussian_noise(noise_variance: Tuple[float, float], per_channel: bool, size: int):
    if not per_channel:
        variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else \
            random.uniform(noise_variance[0], noise_variance[1])
        variance = np.repeat(variance, size)
    else:
        variance = np.repeat(noise_variance[0], size) if noise_variance[0] == noise_variance[1] else \
            np.random.uniform(noise_variance[0], noise_variance[1], size=size)
    return variance


def augment_gaussian_noise(data_sample: np.ndarray, noise_variance: Tuple[float, float] = (0, 0.1),
                           p_per_channel: float = 1, per_channel: bool = False, batched: bool = False) -> np.ndarray:
    mask = np.random.uniform(size=data_sample.shape[1 if batched else 0]) < p_per_channel
    size = np.count_nonzero(mask)
    if size:
        if batched:
            num_samples = data_sample.shape[0]
            mask = np.atleast_2d(mask).repeat(num_samples, axis=0)
            size *= num_samples

        variance = setup_augment_gaussian_noise(noise_variance, per_channel, size)
        data_sample[mask] += np.random.normal(0.0, variance, data_sample[mask].T.shape).T

    return data_sample


def augment_gaussian_blur(data_sample: np.ndarray, sigma_range: Tuple[float, float], per_channel: bool = True,
                          p_per_channel: float = 1, different_sigma_per_axis: bool = False,
                          p_isotropic: float = 0) -> np.ndarray:
    if not per_channel:
        # Godzilla Had a Stroke Trying to Read This and F***ing Died
        # https://i.kym-cdn.com/entries/icons/original/000/034/623/Untitled-3.png
        sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                               ((np.random.uniform() < p_isotropic) and
                                                different_sigma_per_axis)) \
            else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
    else:
        sigma = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                                       ((np.random.uniform() < p_isotropic) and
                                                        different_sigma_per_axis)) \
                    else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
            data_sample[c] = gaussian_filter(data_sample[c], sigma, order=0)
    return data_sample


def augment_blank_square_noise(data_sample, square_size, n_squares, noise_val=(0, 0), channel_wise_n_val=False,
                               square_pos=None):
    # rnd_n_val = get_range_val(noise_val)
    rnd_square_size = get_range_val(square_size)
    rnd_n_squares = get_range_val(n_squares)

    data_sample = mask_random_squares(data_sample, square_size=rnd_square_size, n_squares=rnd_n_squares,
                                      n_val=noise_val, channel_wise_n_val=channel_wise_n_val,
                                      square_pos=square_pos)
    return data_sample
