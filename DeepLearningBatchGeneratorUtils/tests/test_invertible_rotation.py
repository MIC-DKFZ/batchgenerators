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
		inv_rot_batch_gen = InvertibleRotationGenerator(batch_gen)
		batch_gen = inv_rot_batch_gen.generate()

		batch, rotated_batch = next(batch_gen)
		inverse_rotated_batch = inv_rot_batch_gen.invert(rotated_batch)

		self.assertTrue(batch == inverse_rotated_batch)

	def test_3D_rotation(self):
		### test whether a batch of 3D images can be rotated and rotated back

		batch_gen = MockRepeatBatchGenerator((self.x_3D, self.y_3D), BATCH_SIZE=self.BATCH_SIZE, seed=self.seed, num_batches=self.num_batches)
		inv_rot_batch_gen = InvertibleRotationGenerator(batch_gen)
		batch_gen = inv_rot_batch_gen.generate()

		batch, rotated_batch = next(batch_gen)
		inverse_rotated_batch = inv_rot_batch_gen.invert(rotated_batch)

		self.assertTrue(batch == inverse_rotated_batch)


if __name__ == '__main__':
	unittest.main()