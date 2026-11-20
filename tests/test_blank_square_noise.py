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

import random
import unittest
from unittest.mock import patch
import numpy as np
from batchgenerators.augmentations.utils import mask_random_square
from batchgenerators.transforms.noise_transforms import BlankSquareNoiseTransform


class TestBlankSquareNoise(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        random.seed(1234)

    def _masks(self, shape, channel_wise_n_val, square_size=2):
        np.random.seed(1234)
        random.seed(1234)
        data = np.ones((1,) + shape, dtype=np.float32)
        transform = BlankSquareNoiseTransform(squre_size=square_size, n_squres=1, noise_val=(-2., -1.),
                                              channel_wise_n_val=channel_wise_n_val)
        out = transform(data=data)['data'][0]
        return out, out != 1

    def _assert_channel_wise_matches_shared_square(self, shape, square_size=2):
        out, mask = self._masks(shape, True, square_size)
        _, shared_mask = self._masks(shape, False, square_size)
        # the square sits at the same place as without channel_wise_n_val, in every channel
        np.testing.assert_array_equal(mask, shared_mask)
        n_spatial = int(np.prod(shape[1:-2]))
        for c in range(shape[0]):
            self.assertEqual(mask[c].sum(), n_spatial * square_size ** 2)
            # one value per channel
            self.assertEqual(len(np.unique(out[c][mask[c]])), 1)
        # and the channels got their own values
        self.assertEqual(len(np.unique(out[mask])), shape[0])

    def test_channel_wise_2D(self):
        self._assert_channel_wise_matches_shared_square((3, 5, 5))

    def test_channel_wise_3D(self):
        self._assert_channel_wise_matches_shared_square((2, 6, 5, 5))

    def test_channel_wise_3D_more_channels_than_slices(self):
        self._assert_channel_wise_matches_shared_square((3, 2, 5, 5))

    def test_last_origin(self):
        img = np.ones((1, 5, 5))
        with patch('numpy.random.randint', side_effect=lambda low, high: high - 1):
            out = mask_random_square(img, 2, (0, 0))
        expected = np.ones((1, 5, 5))
        expected[:, 3:, 3:] = 0
        np.testing.assert_array_equal(out, expected)

    def test_square_as_large_as_image(self):
        out = mask_random_square(np.ones((2, 4, 4)), 4, (0, 0))
        np.testing.assert_array_equal(out, np.zeros((2, 4, 4)))


if __name__ == '__main__':
    unittest.main()
