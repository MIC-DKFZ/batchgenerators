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

from builtins import range
from typing import Tuple, Union, Callable

import numpy as np
from batchgenerators.augmentations.utils import general_cc_var_num_channels, illumination_jitter


def augment_contrast(data_sample: np.ndarray,
                     contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                     preserve_range: bool = True,
                     per_channel: bool = True,
                     p_per_channel: float = 1) -> np.ndarray:
    size = data_sample.shape[0]
    if per_channel:
        if callable(contrast_range):
            factor = [contrast_range() for _ in range(size)]
        else:
            factor = []
            for _ in range(size):
                if np.random.random() < 0.5 and contrast_range[0] < 1:
                    factor.append(np.random.uniform(contrast_range[0], 1))
                else:
                    factor.append(np.random.uniform(max(contrast_range[0], 1), contrast_range[1]))
        factor = np.array(factor)
    else:
        if callable(contrast_range):
            factor = contrast_range()
        else:
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

    mask = np.random.uniform(size=size) < p_per_channel
    workon = data_sample[mask]
    if len(workon) > 0:
        axes = tuple(range(1, len(data_sample.shape)))
        mean = workon.mean(axis=axes)
        if preserve_range:
            minm = workon.min(axis=axes)
            maxm = workon.max(axis=axes)

        data_sample[mask] = (workon.T * factor + mean * (1 - factor)).T  # writing directly in data_sample

        if preserve_range:
            np.clip(data_sample[mask], minm, maxm, out=data_sample[mask])

    return data_sample


def augment_brightness_additive(data_sample, mu:float, sigma:float , per_channel:bool=True, p_per_channel:float=1.):
    """
    data_sample must have shape (c, x, y(, z)))
    :param data_sample: 
    :param mu: 
    :param sigma: 
    :param per_channel: 
    :param p_per_channel: 
    :return: 
    """
    size = data_sample.shape[0]
    if per_channel:
        rnd_nb = np.random.normal(mu, sigma, size=size)
    else:
        rnd_nb = np.repeat(np.random.normal(mu, sigma), size)
    rnd_nb[np.random.uniform(size=size) > p_per_channel] = 0.0
    axes = tuple(range(len(data_sample.shape) - 1))
    data_sample += np.expand_dims(rnd_nb, axis=axes).T  # Broadcasting rules require this
    return data_sample


def augment_brightness_multiplicative(data_sample, multiplier_range=(0.5, 2), per_channel=True):
    if not per_channel:
        multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
    else:
        axes = [1 for _ in range(len(data_sample.shape))]
        axes[0] = data_sample.shape[0]
        multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1], size=axes)

    data_sample *= multiplier
    return data_sample


def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats: Union[bool, Callable[[], bool]] = False):
    if invert_image:
        data_sample = - data_sample

    if not per_channel:
        retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats_here:
            data_sample -= data_sample.mean()
            data_sample *= sd / (data_sample.std() + 1e-8)
            data_sample += mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats_here:
                data_sample[c] -= data_sample[c].mean()
                data_sample[c] *= sd / (data_sample[c].std() + 1e-8)
                data_sample[c] += mn
    if invert_image:
        data_sample = - data_sample
    return data_sample


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
