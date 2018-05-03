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


from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_brightness_additive, \
    augment_brightness_multiplicative, augment_gamma, augment_illumination, augment_PCA_shift
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class ContrastAugmentationTransform(AbstractTransform):
    """Augments the contrast of data

    Args:
        contrast range (tuple of float): range from which to sample a random contrast that is applied to the data. If
        one value is smaller and one is larger than 1, half of the contrast modifiers will be >1 and the other half <0
        (in the inverval that was specified)

        preserve_range (bool): if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation

        per_channel (bool): whether to use the same contrast modifier for all color channels or a separate one for each
        channel

    """
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, data_key="data"):
        self.data_key = data_key
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        data_dict['data'] = augment_contrast(data_dict[self.data_key], contrast_range=self.contrast_range,
                                             preserve_range=self.preserve_range, per_channel=self.per_channel)

        return data_dict


class BrightnessTransform(AbstractTransform):
    """Augments the brightness of data. Additive brightness is sampled from Gaussian distribution with mu and sigma

    Args:
        mu (float): mean of the Gaussian distribution to sample the added brightness from

        sigma (float): standard deviation of the Gaussian distribution to sample the added brightness from

        per_channel (bool): whether to use the same brightness modifier for all color channels or a separate one for
        each channel

    CAREFUL: This transform will modify the value range of your data!

    """
    def __init__(self, mu, sigma, per_channel=True, data_key="data"):
        self.data_key = data_key
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        data_dict[self.data_key] = augment_brightness_additive(data_dict[self.data_key], self.mu, self.sigma,
                                                               self.per_channel)
        return data_dict


class BrightnessMultiplicativeTransform(AbstractTransform):
    """Augments the brightness of data. Multiplicative brightness is sampled from multiplier_range

    Args:
        multiplier_range (tuple of float): range to uniformly sample the brightness modifier from

        per_channel (bool): whether to use the same brightness modifier for all color channels or a separate one for
        each channel

    CAREFUL: This transform will modify the value range of your data!

    """
    def __init__(self, multiplier_range=(0.5, 2), per_channel=True, data_key="data"):
        self.data_key = data_key
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        data_dict[self.data_key] = augment_brightness_multiplicative(data_dict[self.data_key], self.multiplier_range,
                                                              self.per_channel)
        return data_dict


class GammaTransform(AbstractTransform):
    """Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

    Args:
        gamma_range (tuple of float): range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified)

        invert_image: whether to invert the image before applying gamma augmentation

        retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation

    """

    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data", retain_stats=False):
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        data_dict[self.data_key] = augment_gamma(data_dict[self.data_key], self.gamma_range, self.invert_image,
                                                 per_channel=self.per_channel, retain_stats=self.retain_stats)
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


