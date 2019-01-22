import unittest
import numpy as np
from batchgenerators.augmentations.normalizations import range_normalization, zero_mean_unit_variance_normalization,\
    mean_std_normalization

class TestNormalization(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)

    def test_range_normalization_per_channel(self):
        data = 10*np.random.random((32, 4, 64, 56, 48))
        data[:,0,:,:] = 30*data[:,0,:,:]

        rng = (0,1)
        d = range_normalization(data, rnge=rng, per_channel=True)

        for b in range(data.shape[0]):
            for c in range(data.shape[1]):
                self.assertAlmostEqual(d[b, c, :, :].max(), rng[1], msg="not scaled to correct max range limit")
                self.assertAlmostEqual(d[b, c, :, :].min(), rng[0], msg="not scaled to correct min range limit")

    def test_range_normalization_whole_image(self):
        data = 10 * np.random.random((32, 4, 64, 56, 48))

        data[:, 0, :, :] = 3 * data[:, 0, :, :]
        data2 = data.copy()

        rng = (0, 1)
        d = range_normalization(data, rnge=rng, per_channel=False, eps=0)
        for b in range(data.shape[0]):
            for c in range(data.shape[1]):
                if c==0:
                    self.assertAlmostEqual(d[b, c, :, :].max(), rng[1], msg="not scaled to correct max range limit")
                else:
                    weight = data2[b,c,:,:].max()/30
                    self.assertAlmostEqual(d[b, c, :, :].max(), weight*rng[1], delta=1e-5, msg="not scaled to correct max range limit")
                self.assertAlmostEqual(d[b, c, :, :].min(), rng[0], delta=1e-4, msg="not scaled to correct min range limit")

    def test_zero_mean_unit_variance_per_channel(self):
        data = np.random.random((32, 4, 64, 56, 48))

        d = zero_mean_unit_variance_normalization(data, per_channel=True, epsilon=0)

        for b in range(data.shape[0]):
            for c in range(data.shape[1]):
                self.assertAlmostEqual(d[b, c, :, :].mean(), 0, msg="mean not zeros")
                self.assertAlmostEqual(d[b, c, :, :].std(), 1, msg="std not 1")

    def test_zero_mean_unit_variance_whole_image(self):
        data = np.random.random((32, 4, 64, 56, 48))
        data[:, 0, :, :] = 3 * data[:, 0, :, :]
        d = zero_mean_unit_variance_normalization(data, per_channel=False, epsilon=0)

        for b in range(data.shape[0]):
            self.assertAlmostEqual(d[b, :, :, :].mean(), 0, msg="mean not zeros")
            self.assertAlmostEqual(d[b, :, :, :].std(), 1, msg="std not 1")


if __name__ == '__main__':
    unittest.main()