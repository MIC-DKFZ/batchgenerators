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

import unittest
from itertools import product

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


class TestRangesUpToOne(unittest.TestCase):
    """
    Contrast and gamma ranges that end at or below one must be sampled from the range, as the spatial scale range is
    (issue #137). A range ending at one used to return exactly one for half of the draws.
    """

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())

    @staticmethod
    def _sample_contrast(value_range, per_channel):
        # each channel has mean 1, so its second value becomes 1 + factor
        data = np.array([[0., 2.], [0., 2.]])
        return augment_contrast(data, value_range, preserve_range=False, per_channel=per_channel)[:, 1] - 1

    @staticmethod
    def _sample_gamma(value_range, per_channel):
        # with minimum 0, range 1 and no epsilon each channel's 0.5 becomes 0.5 ** gamma
        data = np.array([[0., 0.5, 1.], [0., 0.5, 1.]])
        return -np.log2(augment_gamma(data, value_range, epsilon=0, per_channel=per_channel)[:, 1])

    def test_ranges_up_to_one_are_respected(self):
        for sample, value_range, per_channel, seed in product((self._sample_contrast, self._sample_gamma),
                                                               ((0.7, 0.9), (0.8, 0.8), (0.7, 1)), (False, True),
                                                               range(20)):
            with self.subTest(sample=sample.__name__, value_range=value_range, per_channel=per_channel, seed=seed):
                np.random.seed(seed)
                sampled = sample(value_range, per_channel)
                self.assertTrue(np.all(sampled >= value_range[0] - 1e-9), sampled)
                self.assertTrue(np.all(sampled <= value_range[1] + 1e-9), sampled)
                # uniform never returns its upper bound, so a factor of exactly one comes from the old upper branch
                self.assertTrue(np.all(sampled < 1), sampled)

    def test_number_of_draws_is_unchanged(self):
        # contrast: per channel a p_per_channel check, plus a branch choice and a factor (per channel or once)
        # gamma: a branch choice and a gamma, per channel or once
        draws = {('_sample_contrast', False): 4, ('_sample_contrast', True): 6,
                 ('_sample_gamma', False): 2, ('_sample_gamma', True): 4}
        for sample, value_range, per_channel in product((self._sample_contrast, self._sample_gamma),
                                                        ((0.7, 0.9), (0.8, 0.8), (0.75, 1.25), (0.7, 1.5), (1.1, 1.5),
                                                         (0.7, 1), (1, 1)), (False, True)):
            with self.subTest(sample=sample.__name__, value_range=value_range, per_channel=per_channel):
                expected = np.random.RandomState(0)
                expected.random_sample(draws[(sample.__name__, per_channel)])
                np.random.seed(0)
                sample(value_range, per_channel)
                actual_state, expected_state = np.random.get_state(), expected.get_state()
                np.testing.assert_array_equal(actual_state[1], expected_state[1])
                self.assertEqual(actual_state[2:], expected_state[2:])


if __name__ == '__main__':
    unittest.main()
