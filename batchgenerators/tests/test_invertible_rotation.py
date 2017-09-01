__author__ = 'Simon Kohl'

import unittest
import numpy as np
from skimage import data
from MockBatchGenerator import MockRepeatBatchGenerator
from DeepLearningBatchGeneratorUtils.InvertibleGenerators import InvertibleRotationGenerator


class TestInvRot(unittest.TestCase):

    def setUp(self):
        self.seed = 1234
        np.random.seed(self.seed)
        self.BATCH_SIZE = 10
        self.num_batches = 1000

        ### 2D initializations
        cam = data.camera()
        ### invariant shape
        self.i_shape = tuple(int(i/2) for i in np.floor(cam.shape/np.sqrt(2)))

        self.cam = cam[np.newaxis, np.newaxis, :, :]
        self.x_2D = self.cam
        self.y_2D = self.cam

        ### 3D initializations
        self.cam_3D = np.random.rand(20,20,20)[np.newaxis, np.newaxis, :, :, :]
        self.x_3D = self.cam_3D
        self.y_3D = self.cam_3D

    def test_2D_rotation(self):
        ### test whether a batch of 2D images can be rotated and rotated back

        batch_gen = MockRepeatBatchGenerator((self.x_2D, self.y_2D), BATCH_SIZE=self.BATCH_SIZE, seed=self.seed, num_batches=self.num_batches)
        inv_rot_batch_gen = InvertibleRotationGenerator(batch_gen, seed=42)
        batch_gen = inv_rot_batch_gen.generate()

        batch, rotated_batch = next(batch_gen)
        inverse_rotated_batch = inv_rot_batch_gen.invert(rotated_batch)

        ### check whether rotation is invertible in the invariant central region of the 2D image
        center = tuple([int(i/2) for i in inverse_rotated_batch['data'].shape[2:]])
        print center, self.i_shape
        print center[0]-self.i_shape[0],center[0]+self.i_shape[0]
        print center[1]-self.i_shape[1],center[1]+self.i_shape[1]

        allowed_error_fractions = [0.2, 0.1]
        for i,type in enumerate(['data', 'seg']):
            center_batch = batch[type][:,:,center[0]-self.i_shape[0]:center[0]+self.i_shape[0],
                                        center[1]-self.i_shape[1]:center[1]+self.i_shape[1]]

            center_inverse_batch = inverse_rotated_batch[type][:,:,center[0]-self.i_shape[0]:center[0]+self.i_shape[0],
                                                                               center[1]-self.i_shape[1]:center[1]+self.i_shape[1]]
            shape = center_batch.shape
            all_pixels = shape[0]*shape[1]*shape[2]*shape[3]
            interpolation_error_fraction = np.sum((center_batch != center_inverse_batch).astype(np.uint8))/all_pixels
            self.assertTrue(interpolation_error_fraction < allowed_error_fractions[i])


if __name__ == '__main__':
    unittest.main()