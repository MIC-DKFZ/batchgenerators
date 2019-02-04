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
from batchgenerators.augmentations.spatial_transformations import augment_zoom

class TestAugmentZoom(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(1234)
        self.data3D = np.zeros((2, 64, 56, 48))
        self.data3D[:, 21:41, 12:32, 13:33] = 1
        self.seg3D = self.data3D

        self.zoom_factors = 2
        self.d3D, self.s3D = augment_zoom(self.data3D, self.seg3D, zoom_factors=self.zoom_factors, order=0, order_seg=0)

        self.data2D = np.zeros((2, 64, 56))
        self.data2D[:, 21:41, 12:32] = 1
        self.seg2D = self.data2D
        self.d2D, self.s2D = augment_zoom(self.data2D, self.seg2D, zoom_factors=self.zoom_factors, order=0, order_seg=0)

    def test_augment_zoom_3D_dimensions(self):
        np.testing.assert_array_equal(self.zoom_factors * np.array(self.data3D.shape[1:]), np.array(self.d3D.shape[1:]), "image has unexpected return shape")
        self.assertTrue(self.data3D.shape[0] == self.d3D.shape[0], "color has unexpected return shape")
        self.assertTrue(self.seg3D.shape[0] == self.s3D.shape[0], "seg color channel has unexpected return shape")
        np.testing.assert_array_equal(self.zoom_factors * np.array(self.seg3D.shape[1:]), np.array(self.s3D.shape[1:]), "seg has unexpected return shape")

    def test_augment_zoom_3D_values(self):
        self.assertTrue(self.zoom_factors ** 3 * sum(self.data3D.flatten()) == sum(self.d3D.flatten()), "image has unexpected values inside")
        self.assertTrue(self.zoom_factors ** 3 * sum(self.seg3D.flatten()) == sum(self.s3D.flatten()), "segmentation has unexpected values inside")
        self.assertTrue(np.all(self.d3D[:, 42:82, 24:64, 26:66].flatten()), "image data is not zoomed correctly")
        idx = np.where(1 - self.d3D)
        tmp = self.d3D[idx]
        self.assertFalse(np.all(tmp), "image has unexpected values outside")
        idx = np.where(1 - self.s3D)
        tmp = self.s3D[idx]
        self.assertFalse(np.all(tmp), "segmentation has unexpected values outside")

    def test_augment_zoom_2D_dimensions(self):
        np.testing.assert_array_equal(self.zoom_factors * np.array(self.data2D.shape[1:]), np.array(self.d2D.shape[1:]), "image has unexpected return shape")
        self.assertTrue(self.data2D.shape[0] == self.d2D.shape[0], "color has unexpected return shape")
        self.assertTrue(self.seg2D.shape[0] == self.s2D.shape[0], "seg color channel has unexpected return shape")
        np.testing.assert_array_equal(self.zoom_factors * np.array(self.seg2D.shape[1:]), np.array(self.s2D.shape[1:]), "seg has unexpected return shape")

    def test_augment_zoom_2D_values(self):
        self.assertTrue(self.zoom_factors ** 2 * sum(self.data2D.flatten()) == sum(self.d2D.flatten()), "image has unexpected values inside")
        self.assertTrue(self.zoom_factors ** 2 * sum(self.seg2D.flatten()) == sum(self.s2D.flatten()), "segmentation has unexpected values inside")
        self.assertTrue(np.all(self.d2D[:, 42:82, 24:64].flatten()), "image data is not zoomed correctly")
        idx = np.where(1 - self.d2D)
        tmp = self.d2D[idx]
        self.assertFalse(np.all(tmp), "image has unexpected values outside")
        idx = np.where(1 - self.s2D)
        tmp = self.s2D[idx]
        self.assertFalse(np.all(tmp), "segmentation has unexpected values outside")


if __name__ == '__main__':
    unittest.main()
