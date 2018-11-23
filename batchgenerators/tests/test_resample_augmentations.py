import unittest
import numpy as np
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy

class TestAugmentResample(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        self.data_3D = np.random.random((2, 64, 56, 48))
        self.data_2D = np.random.random((2, 64, 56))

        self.data_3D_unique = np.reshape(range(1, (1 * 64 * 56 * 48)+1), newshape=(1, 64, 56, 48))
        self.data_2D_unique = np.reshape(range(1, (1 * 64 * 56)+1), newshape=(1, 64, 56))

        self.d_3D = augment_linear_downsampling_scipy(np.copy(self.data_3D), zoom_range=[0.5, 1.5], per_channel=False)
        self.d_2D = augment_linear_downsampling_scipy(np.copy(self.data_2D), zoom_range=[0.5, 1.5], per_channel=False)

        self.zoom_factor = 0.5
        print("=============")
        self.d_3D_upsample = augment_linear_downsampling_scipy(np.copy(self.data_3D_unique), zoom_range=[self.zoom_factor, self.zoom_factor], per_channel=False, order_downsample=0)
        print("=============")
        self.d_2D_upsample = augment_linear_downsampling_scipy(np.copy(self.data_2D_unique), zoom_range=[self.zoom_factor, self.zoom_factor], per_channel=False, order_downsample=0)

    def test_augment_resample(self):
        self.assertTrue(self.data_3D.shape == self.d_3D.shape, "shape of transformed data not the same as original one (3D)")
        self.assertTrue(self.data_2D.shape == self.d_2D.shape, "shape of transformed data not the same as original one (2D)")

    def test_augment_resample_upsample(self):
        print(int(len(np.unique(self.data_3D_unique))*pow(self.zoom_factor, 3)))
        print(len(np.unique(self.d_3D_upsample)))
        self.assertTrue(int(len(np.unique(self.data_3D_unique))*pow(self.zoom_factor, 3)) + 1 == len(np.unique(self.d_3D_upsample)))