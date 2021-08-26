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

from batchgenerators.augmentations.spatial_transformations import augment_rot90, augment_resize, augment_transpose_axes


class AugmentTransposeAxes(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        self.data_3D = np.random.random((2, 4, 5, 6))
        self.seg_3D = np.random.random(self.data_3D.shape)

    def test_transpose_axes(self):
        n_iter = 1000
        tmp = 0
        for i in range(n_iter):
            data_out, seg_out = augment_transpose_axes(self.data_3D, self.seg_3D, axes=(1, 0))

            if np.array_equal(data_out, np.swapaxes(self.data_3D, 1, 2)):
                tmp += 1
        self.assertAlmostEqual(tmp, n_iter/2., delta=10)


class AugmentResize(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        self.data_3D = np.random.random((2, 12, 14, 31))
        self.seg_3D = np.random.random(self.data_3D.shape)

    def test_resize(self):
        data_resized, seg_resized = augment_resize(self.data_3D, self.seg_3D, target_size=15)

        mean_resized = float(np.mean(data_resized))
        mean_original = float(np.mean(self.data_3D))

        self.assertAlmostEqual(mean_original, mean_resized, places=2)

        self.assertTrue(all((data_resized.shape[i] == 15 and seg_resized.shape[i] == 15) for i in
                            range(1, len(data_resized.shape))))

    def test_resize2(self):
        data_resized, seg_resized = augment_resize(self.data_3D, self.seg_3D, target_size=(7, 5, 6))

        mean_resized = float(np.mean(data_resized))
        mean_original = float(np.mean(self.data_3D))

        self.assertAlmostEqual(mean_original, mean_resized, places=2)

        self.assertTrue(all([i == j for i, j in zip(data_resized.shape[1:], (7, 5, 6))]))
        self.assertTrue(all([i == j for i, j in zip(seg_resized.shape[1:], (7, 5, 6))]))


class AugmentRot90(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        self.data_3D = np.random.random((2, 4, 5, 6))
        self.seg_3D = np.random.random(self.data_3D.shape)
        self.num_rot = [1]

    def test_rotation_checkerboard(self):
        data_2d_checkerboard = np.zeros((1, 2, 2))
        data_2d_checkerboard[0, 0, 0] = 1
        data_2d_checkerboard[0, 1, 1] = 1

        data_rotated_list = []
        n_iter = 1000
        for i in range(n_iter):
            d_r, _ = augment_rot90(np.copy(data_2d_checkerboard), None, num_rot=[4,1], axes=[0, 1])
            data_rotated_list.append(d_r)

        data_rotated_np = np.array(data_rotated_list)
        sum_data_list = np.sum(data_rotated_np, axis=0)
        a = np.unique(sum_data_list)
        self.assertAlmostEqual(a[0], n_iter/2, delta=20)
        self.assertTrue(len(a) == 2)

    def test_rotation(self):
        data_rotated, seg_rotated = augment_rot90(np.copy(self.data_3D), np.copy(self.seg_3D), num_rot=self.num_rot,
                                                  axes=[0, 1])

        for i in range(self.data_3D.shape[1]):
            self.assertTrue(np.array_equal(self.data_3D[:, i, :, :], np.flip(data_rotated[:, :, i, :], axis=1)))
            self.assertTrue(np.array_equal(self.seg_3D[:, i, :, :], np.flip(seg_rotated[:, :, i, :], axis=1)))

    def test_randomness_rotation_axis(self):
        tmp = 0
        for j in range(100):
            data_rotated, seg_rotated = augment_rot90(np.copy(self.data_3D), np.copy(self.seg_3D), num_rot=self.num_rot,
                                                      axes=[0, 1, 2])
            if np.array_equal(self.data_3D[:, 0, :, :], np.flip(data_rotated[:, :, 0, :], axis=1)):
                tmp += 1
        self.assertAlmostEqual(tmp, 33, places=2)

    def test_rotation_list(self):
        num_rot = [1, 3]
        data_rotated, seg_rotated = augment_rot90(np.copy(self.data_3D), np.copy(self.seg_3D), num_rot=num_rot,
                                                  axes=[0, 1])
        tmp = 0
        for i in range(self.data_3D.shape[1]):
            # check for normal and inverse rotations
            normal_rotated = np.array_equal(self.data_3D[:, i, :, :], data_rotated[:, :, -i-1, :])
            inverse_rotated = np.array_equal(self.data_3D[:, i, :, :], np.flip(data_rotated[:, :, i, :], axis=1))
            if normal_rotated:
                tmp += 1
            self.assertTrue(normal_rotated or inverse_rotated)
            self.assertTrue(np.array_equal(self.seg_3D[:, i, :, :], seg_rotated[:, :, -i - 1, :]) or
                            np.array_equal(self.seg_3D[:, i, :, :], np.flip(seg_rotated[:, :, i, :], axis=1)))

    def test_randomness_rotation_number(self):
        tmp = 0
        num_rot = [1, 3]
        n_iter = 1000
        for j in range(n_iter):
            data_rotated, seg_rotated = augment_rot90(np.copy(self.data_3D), np.copy(self.seg_3D), num_rot=num_rot,
                                                      axes=[0, 1])
            normal_rotated = np.array_equal(self.data_3D[:, 0, :, :], data_rotated[:, :, - 1, :])
            if normal_rotated:
                tmp += 1
        self.assertAlmostEqual(tmp, n_iter / 2., delta=20)


if __name__ == '__main__':
    unittest.main()
