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
from unittest.mock import patch
import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop
from batchgenerators.augmentations.utils import random_crop_2D_image, random_crop_2D_image_batched, \
    random_crop_3D_image, random_crop_3D_image_batched


class TestRandomCrop(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)

    def _assert_last_origin(self, dim, size, margin, expected_slice, padding):
        data = np.arange(1, size ** dim + 1).reshape((1, 1) + (size,) * dim)
        seg = data + 100
        with patch('numpy.random.randint', side_effect=lambda low, high: high - 1):
            cropped, cropped_seg = random_crop(data, seg, crop_size=3, margins=margin)

        slices = (slice(None), slice(None)) + (expected_slice,) * dim
        pad_width = [(0, 0), (0, 0)] + [padding] * dim
        np.testing.assert_array_equal(cropped, np.pad(data[slices], pad_width))
        np.testing.assert_array_equal(cropped_seg, np.pad(seg[slices], pad_width))

    def test_last_origin_2D(self):
        self._assert_last_origin(2, 4, 0, slice(1, 4), (0, 0))

    def test_last_origin_3D(self):
        self._assert_last_origin(3, 4, 0, slice(1, 4), (0, 0))

    def test_last_origin_with_positive_margin_2D(self):
        self._assert_last_origin(2, 6, 1, slice(2, 5), (0, 0))

    def test_last_origin_with_positive_margin_3D(self):
        self._assert_last_origin(3, 6, 1, slice(2, 5), (0, 0))

    def test_last_origin_with_negative_margin_2D(self):
        self._assert_last_origin(2, 2, -1, slice(0, 2), (0, 1))

    def test_last_origin_with_negative_margin_3D(self):
        self._assert_last_origin(3, 2, -1, slice(0, 2), (0, 1))

    def _assert_single_origin(self, dim, size, crop_size, margin, expected_slice, padding):
        data = np.arange(1, size ** dim + 1).reshape((1, 1) + (size,) * dim)
        with patch('numpy.random.randint') as randint:
            cropped, seg = random_crop(data, crop_size=crop_size, margins=margin)
        randint.assert_not_called()
        expected = data[(slice(None), slice(None)) + (expected_slice,) * dim]
        expected = np.pad(expected, [(0, 0), (0, 0)] + [padding] * dim)
        np.testing.assert_array_equal(cropped, expected)
        self.assertIsNone(seg)

    def test_single_valid_origin_2D(self):
        self._assert_single_origin(2, 5, 3, 1, slice(1, 4), (0, 0))

    def test_single_valid_origin_3D(self):
        self._assert_single_origin(3, 5, 3, 1, slice(1, 4), (0, 0))

    def test_oversized_crop_2D(self):
        self._assert_single_origin(2, 2, 4, 0, slice(0, 2), (1, 1))

    def test_oversized_crop_3D(self):
        self._assert_single_origin(3, 2, 4, 0, slice(0, 2), (1, 1))

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


class TestRandomCropUtils(unittest.TestCase):
    """The random crops in augmentations/utils.py must be able to draw the last valid origin too."""

    def _assert_last_origin(self, crop, shape, n_spatial):
        img = np.arange(np.prod(shape)).reshape(shape)
        with patch('numpy.random.randint', side_effect=lambda low, high: high - 1):
            cropped = crop(img, 3)
        expected = img[(slice(None),) * (len(shape) - n_spatial) + (slice(-3, None),) * n_spatial]
        np.testing.assert_array_equal(cropped, expected)

    def test_last_origin_2D(self):
        self._assert_last_origin(random_crop_2D_image, (5, 6), 2)

    def test_last_origin_2D_batched(self):
        self._assert_last_origin(random_crop_2D_image_batched, (2, 1, 5, 6), 2)

    def test_last_origin_3D(self):
        self._assert_last_origin(random_crop_3D_image, (5, 6, 4), 3)

    def test_last_origin_3D_batched(self):
        self._assert_last_origin(random_crop_3D_image_batched, (2, 1, 5, 6, 4), 3)


if __name__ == '__main__':
    unittest.main()
