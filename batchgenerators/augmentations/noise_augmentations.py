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

import random
from builtins import range

import numpy as np
from batchgenerators.augmentations.utils import get_range_val
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
            sigma = get_range_val(sigma_range)
        for c in range(data.shape[1]):
            if per_channel:
                sigma = get_range_val(sigma_range)
            data[sample_idx, c] = gaussian_filter(data[sample_idx, c], sigma, order=0)
    return data


def augment_blank_square_noise(data, square_size, n_squares, noise_val=(0, 0)):
    def mask_random_square(img, square_size, n_val):
        """Masks (sets = 0) a random square in an image"""

        img_h = img.shape[-2]
        img_w = img.shape[-1]

        img = img.copy()

        w_start = np.random.randint(0, img_w - square_size)
        h_start = np.random.randint(0, img_h - square_size)

        if img.ndim == 2:
            img[h_start:(h_start + square_size), w_start:(w_start + square_size)] = n_val
        elif img.ndim == 3:
            img[:, h_start:(h_start + square_size), w_start:(w_start + square_size)] = n_val
        elif img.ndim == 4:
            img[:, :, h_start:(h_start + square_size), w_start:(w_start + square_size)] = n_val

        return img

    def mask_random_squares(img, square_size, n_squares, n_val):
        """Masks a given number of squares in an image"""
        for i in range(n_squares):
            img = mask_random_square(img, square_size, n_val)
        return img

    for sample_idx in range(data.shape[0]):

        rnd_n_val = get_range_val(noise_val)
        rnd_square_size = get_range_val(square_size)
        rnd_n_squares = get_range_val(n_squares)

        data[sample_idx] = mask_random_squares(data[sample_idx], square_size=rnd_square_size, n_squares=rnd_n_squares,
                                               n_val=rnd_n_val)

    return data
