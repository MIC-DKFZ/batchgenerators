import unittest
import numpy as np
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy


class TestAugmentResample(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        self.data_3D = np.random.random((2, 64, 56, 48))
        self.data_2D = np.random.random((2, 64, 56))

        self.data_3D_unique = np.reshape(range(2 * 64 * 56 * 48), newshape=(2, 64, 56, 48))
        self.data_2D_unique = np.reshape(range(2 * 64 * 56), newshape=(2, 64, 56))

        self.d_3D = augment_linear_downsampling_scipy(np.copy(self.data_3D), zoom_range=[0.5, 1.5], per_channel=False)
        self.d_2D = augment_linear_downsampling_scipy(np.copy(self.data_2D), zoom_range=[0.5, 1.5], per_channel=False)

        self.d_3D_channel = augment_linear_downsampling_scipy(np.copy(self.data_3D), zoom_range=[0.5, 1.5],
                                                              per_channel=False, channels=[0])
        self.d_2D_channel = augment_linear_downsampling_scipy(np.copy(self.data_2D), zoom_range=[0.5, 1.5],
                                                              per_channel=False, channels=[0])

        self.zoom_factor = 0.5
        self.d_3D_upsample = augment_linear_downsampling_scipy(np.copy(self.data_3D_unique),
                                                               zoom_range=[self.zoom_factor, self.zoom_factor],
                                                               per_channel=False, order_downsample=0)
        self.d_2D_upsample = augment_linear_downsampling_scipy(np.copy(self.data_2D_unique),
                                                               zoom_range=[self.zoom_factor, self.zoom_factor],
                                                               per_channel=False, order_downsample=0)

    def test_augment_resample(self):
        self.assertTrue(self.data_3D.shape == self.d_3D.shape,
                        "shape of transformed data not the same as original one (3D)")
        self.assertTrue(self.data_2D.shape == self.d_2D.shape,
                        "shape of transformed data not the same as original one (2D)")

    def test_augment_resample_upsample(self):
        self.assertTrue(int(len(np.unique(self.data_3D_unique))*pow(self.zoom_factor, 3)) == len(np.unique(self.d_3D_upsample)),
                        "number of unique values after resampling is not correct")

    def test_augment_resample_channel(self):
        np.testing.assert_array_equal(self.d_3D_channel[1], self.data_3D[1],
                                      "channel that should not be augmented is changed (3D)")
        np.testing.assert_array_equal(self.d_2D_channel[1], self.data_2D[1],
                                      "channel that should not be augmented is changed (2D)")

        self.assertFalse(np.all(self.d_3D_channel[0] == self.data_3D[0]),
                         "channel that should be augmented is not changed (3D)")
        self.assertFalse(np.all(self.d_2D_channel[0] == self.data_2D[0]),
                         "channel that should be augmented is not changed (2D)")


if __name__ == '__main__':
    unittest.main()
