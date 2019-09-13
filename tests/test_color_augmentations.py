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


import unittest
import numpy as np
from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_brightness_additive,\
    augment_brightness_multiplicative, augment_gamma


class TestAugmentContrast(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        self.data_3D = np.random.random((2, 64, 56, 48))
        self.data_2D = np.random.random((2, 64, 56))
        self.factor = (0.75, 1.25)

        self.d_3D = augment_contrast(self.data_3D, contrast_range=self.factor, preserve_range=False, per_channel=False)
        self.d_2D = augment_contrast(self.data_2D, contrast_range=self.factor, preserve_range=False, per_channel=False)

    def test_augment_contrast_3D(self):

        mean = np.mean(self.data_3D)

        idx0 = np.where(self.data_3D < mean)  # where the data is lower than mean value
        idx1 = np.where(self.data_3D > mean)  # where the data is greater than mean value

        contrast_lower_limit_0 = self.factor[1] * (self.data_3D[idx0] - mean) + mean
        contrast_lower_limit_1 = self.factor[0] * (self.data_3D[idx1] - mean) + mean
        contrast_upper_limit_0 = self.factor[0] * (self.data_3D[idx0] - mean) + mean
        contrast_upper_limit_1 = self.factor[1] * (self.data_3D[idx1] - mean) + mean

        # augmented values lower than mean should be lower than lower limit and greater than upper limit
        self.assertTrue(np.all(np.logical_and(self.d_3D[idx0] >= contrast_lower_limit_0,
                                              self.d_3D[idx0] <= contrast_upper_limit_0)),
                        "Augmented contrast below mean value not within range")
        # augmented values greater than mean should be lower than upper limit and greater than lower limit
        self.assertTrue(np.all(np.logical_and(self.d_3D[idx1] >= contrast_lower_limit_1,
                                              self.d_3D[idx1] <= contrast_upper_limit_1)),
                        "Augmented contrast above mean not within range")

    def test_augment_contrast_2D(self):

        mean = np.mean(self.data_2D)

        idx0 = np.where(self.data_2D < mean)  # where the data is lower than mean value
        idx1 = np.where(self.data_2D > mean)  # where the data is greater than mean value

        contrast_lower_limit_0 = self.factor[1] * (self.data_2D[idx0] - mean) + mean
        contrast_lower_limit_1 = self.factor[0] * (self.data_2D[idx1] - mean) + mean
        contrast_upper_limit_0 = self.factor[0] * (self.data_2D[idx0] - mean) + mean
        contrast_upper_limit_1 = self.factor[1] * (self.data_2D[idx1] - mean) + mean

        # augmented values lower than mean should be lower than lower limit and greater than upper limit
        self.assertTrue(np.all(np.logical_and(self.d_2D[idx0] >= contrast_lower_limit_0,
                                              self.d_2D[idx0] <= contrast_upper_limit_0)),
                        "Augmented contrast below mean value not within range")
        # augmented values greater than mean should be lower than upper limit and greater than lower limit
        self.assertTrue(np.all(np.logical_and(self.d_2D[idx1] >= contrast_lower_limit_1,
                                              self.d_2D[idx1] <= contrast_upper_limit_1)),
                        "Augmented contrast above mean not within range")


class TestAugmentBrightness(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        self.data_input_3D = np.random.random((2, 64, 56, 48))
        self.data_input_2D = np.random.random((2, 64, 56))
        self.factor = (0.75, 1.25)
        self.multiplier_range = [2,4]

        self.d_3D_per_channel = augment_brightness_additive(np.copy(self.data_input_3D), mu=100, sigma=10,
                                                            per_channel=True)
        self.d_3D = augment_brightness_additive(np.copy(self.data_input_3D), mu=100, sigma=10, per_channel=False)

        self.d_2D_per_channel = augment_brightness_additive(np.copy(self.data_input_2D), mu=100, sigma=10,
                                                            per_channel=True)
        self.d_2D = augment_brightness_additive(np.copy(self.data_input_2D), mu=100, sigma=10, per_channel=False)

        self.d_3D_per_channel_mult = augment_brightness_multiplicative(np.copy(self.data_input_3D),
                                                                       multiplier_range=self.multiplier_range,
                                                                       per_channel=True)
        self.d_3D_mult = augment_brightness_multiplicative(np.copy(self.data_input_3D),
                                                           multiplier_range=self.multiplier_range, per_channel=False)

        self.d_2D_per_channel_mult = augment_brightness_multiplicative(np.copy(self.data_input_2D),
                                                                       multiplier_range=self.multiplier_range,
                                                                       per_channel=True)
        self.d_2D_mult = augment_brightness_multiplicative(np.copy(self.data_input_2D),
                                                           multiplier_range=self.multiplier_range, per_channel=False)

    def test_augment_brightness_additive_3D(self):
        add_factor = self.d_3D-self.data_input_3D
        self.assertTrue(len(np.unique(add_factor.round(decimals=8)))==1,
                        "Added brightness factor is not equal for all channels")

        add_factor = self.d_3D_per_channel - self.data_input_3D
        self.assertTrue(len(np.unique(add_factor.round(decimals=8))) == self.data_input_3D.shape[0],
                        "Added brightness factor is not different for each channels")

    def test_augment_brightness_additive_2D(self):
        add_factor = self.d_2D-self.data_input_2D
        self.assertTrue(len(np.unique(add_factor.round(decimals=8)))==1,
                        "Added brightness factor is not equal for all channels")

        add_factor = self.d_2D_per_channel - self.data_input_2D
        self.assertTrue(len(np.unique(add_factor.round(decimals=8))) == self.data_input_2D.shape[0],
                        "Added brightness factor is not different for each channels")

    def test_augment_brightness_multiplicative_3D(self):
        mult_factor = self.d_3D_mult/self.data_input_3D
        self.assertTrue(len(np.unique(mult_factor.round(decimals=6)))==1,
                        "Multiplied brightness factor is not equal for all channels")

        mult_factor = self.d_3D_per_channel_mult/self.data_input_3D
        self.assertTrue(len(np.unique(mult_factor.round(decimals=6))) == self.data_input_3D.shape[0],
                        "Multiplied brightness factor is not different for each channels")

    def test_augment_brightness_multiplicative_2D(self):
        mult_factor = self.d_2D_mult/self.data_input_2D
        self.assertTrue(len(np.unique(mult_factor.round(decimals=6)))==1,
                        "Multiplied brightness factor is not equal for all channels")

        mult_factor = self.d_2D_per_channel_mult/self.data_input_2D
        self.assertTrue(len(np.unique(mult_factor.round(decimals=6))) == self.data_input_2D.shape[0],
                        "Multiplied brightness factor is not different for each channels")


class TestAugmentGamma(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        self.data_input_3D = np.random.random((2, 64, 56, 48))
        self.data_input_2D = np.random.random((2, 64, 56))

        self.d_3D = augment_gamma(np.copy(self.data_input_2D), gamma_range=(0.2, 1.2), per_channel=False)

    def test_augment_gamma_3D(self):
        self.assertTrue(self.d_3D.min().round(decimals=3) == self.data_input_3D.min().round(decimals=3) and
                        self.d_3D.max().round(decimals=3) == self.data_input_3D.max().round(decimals=3),
                        "Input range does not equal output range")


if __name__ == '__main__':
    unittest.main()
