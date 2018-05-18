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

from batchgenerators.augmentations.utils import general_cc_var_num_channels, illumination_jitter


def augment_contrast(data, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
    for sample in range(data.shape[0]):
        if not per_channel:
            mn = data[sample].mean()
            if preserve_range:
                minm = data[sample].min()
                maxm = data[sample].max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data[sample] = (data[sample] - mn) * factor + mn
            if preserve_range:
                data[sample][data[sample] < minm] = minm
                data[sample][data[sample] > maxm] = maxm
        else:
            for c in range(data[sample].shape[0]):
                mn = data[sample][c].mean()
                if preserve_range:
                    minm = data[sample][c].min()
                    maxm = data[sample][c].max()
                if np.random.random() < 0.5 and contrast_range[0] < 1:
                    factor = np.random.uniform(contrast_range[0], 1)
                else:
                    factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
                data[sample][c] = (data[sample][c] - mn) * factor + mn
                if preserve_range:
                    data[sample][c][data[sample][c] < minm] = minm
                    data[sample][c][data[sample][c] > maxm] = maxm
    return data


def augment_brightness_additive(data, mu, sigma, per_channel=True):
    for sample_idx in range(data.shape[0]):
        sample = data[sample_idx]
        if not per_channel:
            rnd_nb = np.random.normal(mu, sigma)
            sample += rnd_nb
        else:
            for c in range(sample.shape[0]):
                rnd_nb = np.random.normal(mu, sigma)
                sample[c] += rnd_nb
        data[sample_idx] = sample
    return data


def augment_brightness_multiplicative(data, multiplier_range=(0.5, 2), per_channel=True):
    for sample_idx in range(data.shape[0]):
        sample = data[sample_idx]
        multiplier = random.uniform(multiplier_range[0], multiplier_range[1])
        if not per_channel:
            sample *= multiplier
        else:
            for c in range(sample.shape[0]):
                sample[c] *= multiplier
        data[sample_idx] = sample
    return data


def augment_gamma(data, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False, retain_stats=False):
    for sample in range(data.shape[0]):
        if invert_image:
            data = - data
        if not per_channel:
            if retain_stats:
                mn = data[sample].mean()
                sd = data[sample].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data[sample].min()
            rnge = data[sample].max() - minm
            data[sample] = np.power(((data[sample] - minm) / float(rnge + epsilon)), gamma) * rnge + minm
            if retain_stats:
                data[sample] = data[sample] - data[sample].mean() + mn
                data[sample] = data[sample] / data[sample].std() * sd
        else:
            for c in range(data.shape[1]):
                if retain_stats:
                    mn = data[sample][c].mean()
                    sd = data[sample][c].std()
                if np.random.random() < 0.5 and gamma_range[0] < 1:
                    gamma = np.random.uniform(gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                minm = data[sample][c].min()
                rnge = data[sample][c].max() - minm
                data[sample][c] = np.power(((data[sample][c] - minm) / float(rnge + epsilon)), gamma) * rnge + minm
                if retain_stats:
                    data[sample][c] = data[sample][c] - data[sample][c].mean() + mn
                    data[sample][c] = data[sample][c] / data[sample][c].std() * sd
    return data


def augment_illumination(data, white_rgb):
    idx = np.random.choice(len(white_rgb), data.shape[0])
    for sample in range(data.shape[0]):
        _, img = general_cc_var_num_channels(data[sample], 0, 5, 0, None, 1., 7, False)
        rgb = np.array(white_rgb[idx[sample]]) * np.sqrt(3)
        for c in range(data[sample].shape[0]):
            data[sample, c] = img[c] * rgb[c]
    return data


def augment_PCA_shift(data, U, s, sigma=0.2):
    for sample in range(data.shape[0]):
        data[sample] = illumination_jitter(data[sample], U, s, sigma)
        data[sample] -= data[sample].min()
        data[sample] /= data[sample].max()
    return data
