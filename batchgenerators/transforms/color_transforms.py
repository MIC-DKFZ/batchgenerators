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

from typing import Union, Tuple, Callable

import numpy as np

from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_brightness_additive, \
    augment_brightness_multiplicative, augment_gamma, augment_illumination, augment_PCA_shift
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class ContrastAugmentationTransform(AbstractTransform):
    def __init__(self,
                 contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                 preserve_range: bool = True,
                 per_channel: bool = True,
                 data_key: str = "data",
                 p_per_sample: float = 1,
                 p_per_channel: float = 1):
        """
        Augments the contrast of data
        :param contrast_range:
            (float, float): range from which to sample a random contrast that is applied to the data. If
                            one value is smaller and one is larger than 1, half of the contrast modifiers will be >1
                            and the other half <1 (in the inverval that was specified)
            callable      : must be contrast_range() -> float
        :param preserve_range: if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation.
        :param per_channel: whether to use the same contrast modifier for all color channels or a separate one for each
        channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_contrast(data_dict[self.data_key][b],
                                                               contrast_range=self.contrast_range,
                                                               preserve_range=self.preserve_range,
                                                               per_channel=self.per_channel,
                                                               p_per_channel=self.p_per_channel)
        return data_dict


class NormalizeTransform(AbstractTransform):
    def __init__(self, means, stds, data_key='data'):
        self.data_key = data_key
        self.stds = stds
        self.means = means

    def __call__(self, **data_dict):
        for c in range(data_dict[self.data_key].shape[1]):
            data_dict[self.data_key][:, c] -= self.means[c]
            data_dict[self.data_key][:, c] /= self.stds[c]
        return data_dict


class BrightnessTransform(AbstractTransform):
    def __init__(self, mu, sigma, per_channel=True, data_key="data", p_per_sample=1, p_per_channel=1):
        """
        Augments the brightness of data. Additive brightness is sampled from Gaussian distribution with mu and sigma
        :param mu: mean of the Gaussian distribution to sample the added brightness from
        :param sigma: standard deviation of the Gaussian distribution to sample the added brightness from
        :param per_channel: whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]

        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data[b] = augment_brightness_additive(data[b], self.mu, self.sigma, self.per_channel,
                                                      p_per_channel=self.p_per_channel)

        data_dict[self.data_key] = data
        return data_dict


class BrightnessMultiplicativeTransform(AbstractTransform):
    def __init__(self, multiplier_range=(0.5, 2), per_channel=True, data_key="data", p_per_sample=1):
        """
        Augments the brightness of data. Multiplicative brightness is sampled from multiplier_range
        :param multiplier_range: range to uniformly sample the brightness modifier from
        :param per_channel:  whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_brightness_multiplicative(data_dict[self.data_key][b],
                                                                                self.multiplier_range,
                                                                                self.per_channel)
        return data_dict


class GammaTransform(AbstractTransform):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data",
                 retain_stats: Union[bool, Callable[[], bool]] = False, p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation. retain_stats
        can also be callable (signature retain_stats() -> bool)
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gamma(data_dict[self.data_key][b], self.gamma_range,
                                                            self.invert_image,
                                                            per_channel=self.per_channel,
                                                            retain_stats=self.retain_stats)
        return data_dict


class IlluminationTransform(AbstractTransform):
    """Do not use this for now"""

    def __init__(self, white_rgb, data_key="data"):
        self.data_key = data_key
        self.white_rgb = white_rgb

    def __call__(self, **data_dict):
        data_dict[self.data_key] = augment_illumination(data_dict[self.data_key], self.white_rgb)
        return data_dict


class FancyColorTransform(AbstractTransform):
    """Do not use this for now"""

    def __init__(self, U, s, sigma=0.2, data_key="data"):
        self.data_key = data_key
        self.s = s
        self.U = U
        self.sigma = sigma

    def __call__(self, **data_dict):
        data_dict[self.data_key] = augment_PCA_shift(data_dict[self.data_key], self.U, self.s, self.sigma)
        return data_dict


class ClipValueRange(AbstractTransform):
    def __init__(self, min=None, max=None, data_key="data"):
        """
        Clips the value range of data to [min, max]
        :param min:
        :param max:
        :param data_key:
        """
        self.data_key = data_key
        self.min = min
        self.max = max

    def __call__(self, **data_dict):
        data_dict[self.data_key] = np.clip(data_dict[self.data_key], self.min, self.max)
        return data_dict
