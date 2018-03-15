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

import numpy as np
from batchgenerators.augmentations.utils import get_range_val
from builtins import range
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


def augment_blank_square_noise(data, square_size, n_squares, noise_val=(0, 0), channel_wise_n_val=False,
                               square_pos=None):
    def mask_random_square(img, square_size, n_val, channel_wise_n_val=False, square_pos=None):
        """Masks (sets = 0) a random square in an image"""

        img_h = img.shape[-2]
        img_w = img.shape[-1]

        img = img.copy()

        if square_pos is None:
            w_start = np.random.randint(0, img_w - square_size)
            h_start = np.random.randint(0, img_h - square_size)
        else:
            pos_wh = square_pos[np.random.randint(0, len(square_pos))]
            w_start = pos_wh[0]
            h_start = pos_wh[1]

        if img.ndim == 2:
            rnd_n_val = get_range_val(n_val)
            img[h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
        elif img.ndim == 3:
            if channel_wise_n_val:
                for i in range(img.shape[0]):
                    rnd_n_val = get_range_val(n_val)
                    img[i, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
            else:
                rnd_n_val = get_range_val(n_val)
                img[:, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
        elif img.ndim == 4:
            if channel_wise_n_val:
                for i in range(img.shape[0]):
                    rnd_n_val = get_range_val(n_val)
                    img[:, i, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
            else:
                rnd_n_val = get_range_val(n_val)
                img[:, :, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val

        return img

    def mask_random_squares(img, square_size, n_squares, n_val, channel_wise_n_val=False, square_pos=None):
        """Masks a given number of squares in an image"""
        for i in range(n_squares):
            img = mask_random_square(img, square_size, n_val, channel_wise_n_val=channel_wise_n_val,
                                     square_pos=square_pos)
        return img

    for sample_idx in range(data.shape[0]):
        # rnd_n_val = get_range_val(noise_val)
        rnd_square_size = get_range_val(square_size)
        rnd_n_squares = get_range_val(n_squares)

        data[sample_idx] = mask_random_squares(data[sample_idx], square_size=rnd_square_size, n_squares=rnd_n_squares,
                                               n_val=noise_val, channel_wise_n_val=channel_wise_n_val,
                                               square_pos=square_pos)

    return data
