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
from batchgenerators.augmentations.color_augmentations import augment_contrast


class TestAugmentContrast(unittest.TestCase):

    def setUp(self):
        # np.random.seed(1234)
        self.data = np.random.random((2, 64, 56, 48))
        self.factor = (0.75, 1.25)

        self.d = augment_contrast(self.data, contrast_range=self.factor, preserve_range=False, per_channel=False)

    def test_augment_contrast_3D(self):

        mean = np.mean(self.data)

        idx0 = np.where(self.data < mean)  # where the data is lower than mean value
        idx1 = np.where(self.data > mean)  # where the data is greater than mean value

        contrast_lower_limit_0 = self.factor[1]*(self.data[idx0]-mean) + mean
        contrast_lower_limit_1 = self.factor[0]*(self.data[idx1]-mean) + mean
        contrast_upper_limit_0 = self.factor[0]*(self.data[idx0]-mean) + mean
        contrast_upper_limit_1 = self.factor[1]*(self.data[idx1]-mean) + mean

        # augmented values lower than mean should be lower than lower limit and greater than upper limit
        self.assertTrue(np.all(np.logical_and(self.d[idx0] >= contrast_lower_limit_0,
                                              self.d[idx0] <= contrast_upper_limit_0)),
                        "Augmented contrast below mean value not within range")
        # augmented values greater than mean should be lower than upper limit and greater than lower limit
        self.assertTrue(np.all(np.logical_and(self.d[idx1] >= contrast_lower_limit_1,
                                              self.d[idx1] <= contrast_upper_limit_1)),
                        "Augmented contrast above mean not within range")
