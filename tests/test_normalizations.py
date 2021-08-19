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

import numpy as np
from batchgenerators.augmentations.normalizations import range_normalization, zero_mean_unit_variance_normalization, \
    cut_off_outliers


class TestNormalization(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)

    def test_range_normalization_per_channel(self):
        print('Test test_range_normalization_per_channel. [START]')
        data = 10 * np.random.random((32, 4, 64, 56, 48))
        data[:, 0, :, :] = 30 * data[:, 0, :, :]

        data3 = 5 * np.ones((8, 2, 64, 56, 48))
        data4 = np.array([])

        rng1 = (0, 1)
        rng2 = (-2, 2)
        rng3 = (0, 1)

        data_normalized = range_normalization(data, rnge=rng1, per_channel=True)
        data_normalized2 = range_normalization(data, rnge=rng2, per_channel=True)
        data_normalized3 = range_normalization(data3, rnge=rng3, per_channel=True)
        data_normalized4 = range_normalization(data4, rnge=rng1, per_channel=True)

        print('Test normalization with range [0,1]. [START]')
        for b in range(data.shape[0]):
            for c in range(data.shape[1]):
                self.assertAlmostEqual(data_normalized[b, c, :, :].max(), rng1[1],
                                       msg="not scaled to correct max range limit")
                self.assertAlmostEqual(data_normalized[b, c, :, :].min(), rng1[0],
                                       msg="not scaled to correct min range limit")
        print('Test normalization with range [0,1]. [DONE]')

        print('Test normalization with range [-2,2]. [START]')
        for b in range(data.shape[0]):
            for c in range(data.shape[1]):
                self.assertAlmostEqual(data_normalized2[b, c, :, :].max(), rng2[1],
                                       msg="not scaled to correct max range limit")
                self.assertAlmostEqual(data_normalized2[b, c, :, :].min(), rng2[0],
                                       msg="not scaled to correct min range limit")
        print('Test normalization with range [-2,2]. [DONE]')

        print('Test normalization of constant data with range [0,1]. [START]')
        for b in range(data3.shape[0]):
            for c in range(data3.shape[1]):
                self.assertAlmostEqual(data_normalized3[b, c, :, :].max(), rng3[0],
                                       msg="not scaled to correct max range limit")
                self.assertAlmostEqual(data_normalized3[b, c, :, :].min(), rng3[0],
                                       msg="not scaled to correct min range limit")
        print('Test normalization of constant data with range [0,1]. [DONE]')

        print('Test normalization of empty data array with range [0,1]. [START]')
        self.assertEqual(data_normalized4.size, 0, msg="not an empty array")
        print('Test normalization of empty data array with range [0,1]. [DONE]')

        # print('Test RuntimeWarning of constant data with zero eps. [START]')
        # self.assertWarns(RuntimeWarning, range_normalization, data3, rnge = rng3, per_channel = True, eps = 0)
        # print('Test RuntimeWarning of constant data with zero eps. [DONE]')

        print('Test test_range_normalization_per_channel. [DONE]')

    def test_range_normalization_whole_image(self):
        print('Test test_range_normalization_whole_image. [START]')
        data = 10 * np.random.random((32, 4, 64, 56, 48))
        data[:, 0, :, :] = 3 * data[:, 0, :, :]

        data3 = 5 * np.ones((8, 2, 64, 56, 48))
        data4 = np.array([])

        rng1 = (0, 1)
        rng2 = (-2, 2)
        rng3 = (0, 1)

        data_normalized1 = range_normalization(data, rnge=rng1, per_channel=False, eps=0)
        data_normalized2 = range_normalization(data, rnge=rng2, per_channel=False)
        data_normalized3 = range_normalization(data3, rnge=rng3, per_channel=False)
        data_normalized4 = range_normalization(data4, rnge=rng1, per_channel=False)

        print('Test normalization with range [0,1]. [START]')
        for b in range(data.shape[0]):
            self.assertAlmostEqual(data_normalized1[b].min(), rng1[0], delta=1e-4,
                                   msg="not scaled to correct min range limit")
            self.assertAlmostEqual(data_normalized1[b].max(), rng1[1], delta=1e-4,
                                   msg="not scaled to correct max range limit")
            self.assertEqual(np.unravel_index(np.argmax(data_normalized1[b], axis=None), data_normalized1[b].shape)[0],
                             0, msg="max not in the right channel")
        print('Test normalization with range [0,1]. [DONE]')

        print('Test normalization with range [-2, 2]. [START]')
        for b in range(data.shape[0]):
            self.assertAlmostEqual(data_normalized2[b].min(), rng2[0], delta=1e-4,
                                   msg="not scaled to correct min range limit")
            self.assertAlmostEqual(data_normalized2[b].max(), rng2[1], delta=1e-4,
                                   msg="not scaled to correct max range limit")
            self.assertEqual(np.unravel_index(np.argmax(data_normalized2[b], axis=None), data_normalized1[b].shape)[0],
                             0, msg="max not in the right channel")
        print('Test normalization with range [-2, 2]. [DONE]')

        print('Test normalization of constant data with range [0,1]. [START]')
        for b in range(data3.shape[0]):
            self.assertAlmostEqual(data_normalized3[b].min(), rng3[0], delta=1e-4,
                                   msg="not scaled to correct min range limit")
            self.assertAlmostEqual(data_normalized3[b].max(), rng3[0], delta=1e-4,
                                   msg="not scaled to correct max range limit")
            # self.assertEqual(np.unravel_index(np.argmax(data_normalized3[b], axis=None), data_normalized1[b].shape)[0], 0, msg="max not in the right channel")
        print('Test normalization of constant data  with range [0,1]. [DONE]')

        print('Test normalization of empty data array with range [0,1]. [START]')
        self.assertEqual(data_normalized4.size, 0, msg="not an empty array")
        print('Test normalization of empty data array with range [0,1]. [DONE]')

        # print('Test RuntimeWarning of constant data with zero eps. [START]')
        # self.assertWarns(RuntimeWarning, range_normalization, data3, rnge = rng3, per_channel = False, eps = 0)
        # print('Test RuntimeWarning of constant data with zero eps. [DONE]')

        print('Test test_range_normalization_whole_image. [DONE]')

    def test_zero_mean_unit_variance_per_channel(self):
        print('Test test_zero_mean_unit_variance_per_channel. [START]')
        data = np.random.random((32, 4, 64, 56, 48))
        data2 = 5 * np.ones((32, 4, 64, 56, 48))
        data3 = np.array([])

        data_normalized1 = zero_mean_unit_variance_normalization(data, per_channel=True, epsilon=0)
        data_normalized2 = zero_mean_unit_variance_normalization(data2, per_channel=True)
        data_normalized3 = zero_mean_unit_variance_normalization(data3, per_channel=True)

        print('Test standard use-case. [START]')
        for b in range(data.shape[0]):
            for c in range(data.shape[1]):
                self.assertAlmostEqual(data_normalized1[b, c, :, :].mean(), 0, msg="mean not zeros")
                self.assertAlmostEqual(data_normalized1[b, c, :, :].std(), 1, msg="std not 1")
        print('Test standard use-case. [DONE]')

        print('Test constant input data. [START]')
        for b in range(data2.shape[0]):
            for c in range(data2.shape[1]):
                self.assertAlmostEqual(data_normalized2[b, c, :, :].mean(), 0, msg="mean not zeros")
                self.assertAlmostEqual(data_normalized2[b, c, :, :].std(), 0, msg="std not 1")
        print('Test constant input data. [DONE]')

        # print('Test RuntimeWarning of constant data with zero eps. [START]')
        # self.assertWarns(RuntimeWarning, zero_mean_unit_variance_normalization, data2, per_channel=True, epsilon=0)
        # print('Test RuntimeWarning of constant data with zero eps. [DONE]')

        print('Test normalization of empty data array. [START]')
        self.assertEqual(data_normalized3.size, 0, msg="not an empty array")
        print('Test normalization of empty data array. [DONE]')

        print('Test test_zero_mean_unit_variance_per_channel. [DONE]')

    def test_zero_mean_unit_variance_whole_image(self):
        print('Test test_zero_mean_unit_variance_whole_image. [START]')
        data = np.random.random((32, 4, 64, 56, 48))
        data2 = 5 * np.ones((32, 4, 64, 56, 48))
        data3 = np.array([])

        data_normalized1 = zero_mean_unit_variance_normalization(data, per_channel=False, epsilon=0)
        data_normalized2 = zero_mean_unit_variance_normalization(data2, per_channel=False)
        data_normalized3 = zero_mean_unit_variance_normalization(data3, per_channel=False)

        print('Test standard use-case. [START]')
        for b in range(data.shape[0]):
            self.assertAlmostEqual(data_normalized1[b].mean(), 0, msg="mean not zeros")
            self.assertAlmostEqual(data_normalized1[b].std(), 1, msg="std not 1")
        print('Test standard use-case. [DONE]')

        print('Test constant input data. [START]')
        for b in range(data2.shape[0]):
            self.assertAlmostEqual(data_normalized2[b].mean(), 0, msg="mean not zeros")
            self.assertAlmostEqual(data_normalized2[b].std(), 0, msg="std not 1")
        print('Test constant input data. [DONE]')

        # print('Test RuntimeWarning of constant data with zero eps. [START]')
        # self.assertWarns(RuntimeWarning, zero_mean_unit_variance_normalization, data2, per_channel=False, epsilon=0)
        # print('Test RuntimeWarning of constant data with zero eps. [DONE]')

        print('Test normalization of empty data array. [START]')
        self.assertEqual(data_normalized3.size, 0, msg="not an empty array")
        print('Test normalization of empty data array. [DONE]')

        print('Test test_zero_mean_unit_variance_whole_image. [DONE]')

    def test_cut_off_outliers_per_channel(self):
        print('Test test_cut_off_outliers_per_channel. [START]')
        data = np.ones((32, 4, 64, 56, 48))

        data[:, :, 1, 1, 1] = 999999
        data[:, :, 1, 1, 2] = -999999

        data_normalized = cut_off_outliers(data, per_channel=True)
        for b in range(data.shape[0]):
            for c in range(data.shape[1]):
                self.assertAlmostEqual(data_normalized[b, c].min(), 1.0, msg="Lowe outlier not removed.")
                self.assertAlmostEqual(data_normalized[b, c].max(), 1.0, msg="Upper outlier not removed.")

        print('Test test_cut_off_outliers_per_channel. [START]')

    def test_cut_off_outliers_whole_image(self):
        print('Test test_cut_off_outliers_whole_image. [START]')
        data = np.ones((32, 4, 64, 56, 48))

        data[:, :, 1, 1, 1] = 999999
        data[:, :, 1, 1, 2] = -999999

        data_normalized = cut_off_outliers(data, per_channel=False)
        for b in range(data.shape[0]):
            self.assertAlmostEqual(data_normalized[b].min(), 1.0, msg="Lowe outlier not removed.")
            self.assertAlmostEqual(data_normalized[b].max(), 1.0, msg="Upper outlier not removed.")

        print('Test test_cut_off_outliers_whole_image. [START]')


if __name__ == '__main__':
    unittest.main()
