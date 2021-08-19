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
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop


class TestRandomCrop(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)

    def test_random_crop_3D(self):
        data = np.random.random((32, 4, 64, 56, 48))
        seg = np.ones(data.shape)

        d, s = random_crop(data, seg, 32, 0)

        self.assertTrue(all(i == j for i, j in zip((32, 4, 32, 32, 32), d.shape)), "data has unexpected return shape")
        self.assertTrue(all(i == j for i, j in zip((32, 4, 32, 32, 32), s.shape)), "seg has unexpected return shape")

        self.assertEqual(np.sum(s == 0), 0, "Zeros encountered in seg meaning that we did padding which should not have"
                                            " happened here!")

    def test_random_crop_2D(self):
        data = np.random.random((32, 4, 64, 56))
        seg = np.ones(data.shape)

        d, s = random_crop(data, seg, 32, 0)

        self.assertTrue(all(i == j for i, j in zip((32, 4, 32, 32), d.shape)), "data has unexpected return shape")
        self.assertTrue(all(i == j for i, j in zip((32, 4, 32, 32), s.shape)), "seg has unexpected return shape")

        self.assertEqual(np.sum(s == 0), 0, "Zeros encountered in seg meaning that we did padding which should not have"
                                            " happened here!")

    def test_random_crop_3D_from_List(self):
        data = [np.random.random((4, 64+i, 56+i, 48+i)) for i in range(32)]
        seg = [np.random.random((4, 64+i, 56+i, 48+i)) for i in range(32)]

        d, s = random_crop(data, seg, 32, 0)

        self.assertTrue(all(i == j for i, j in zip((32, 4, 32, 32), d.shape)), "data has unexpected return shape")
        self.assertTrue(all(i == j for i, j in zip((32, 4, 32, 32), s.shape)), "seg has unexpected return shape")

        self.assertEqual(np.sum(s == 0), 0, "Zeros encountered in seg meaning that we did padding which should not have"
                                            " happened here!")

    def test_random_crop_2D_from_List(self):
        data = [np.random.random((4, 64+i, 56+i)) for i in range(32)]
        seg = [np.random.random((4, 64+i, 56+i)) for i in range(32)]

        d, s = random_crop(data, seg, 32, 0)

        self.assertTrue(all(i == j for i, j in zip((32, 4, 32, 32), d.shape)), "data has unexpected return shape")
        self.assertTrue(all(i == j for i, j in zip((32, 4, 32, 32), s.shape)), "seg has unexpected return shape")

        self.assertEqual(np.sum(s == 0), 0, "Zeros encountered in seg meaning that we did padding which should not have"
                                            " happened here!")


if __name__ == '__main__':
    unittest.main()
