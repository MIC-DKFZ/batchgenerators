import unittest
import numpy as np
from MockBatchGenerator import MockBatchGenerator
from DeepLearningBatchGeneratorUtils.MultiThreadedGenerator import MultiThreadedGenerator

class TestMultiThreading(unittest.TestCase):

    def setUp(self):
        self.x = np.random.rand(100, 5)
        self.y = np.random.rand(100, 1)

    def test_test1(self):
        pass

    def test_nr_of_batches(self):

        NR_THREADS = 5
        NR_BATCHES_PER_THREAD = 10

        batch_gen = MockBatchGenerator((self.x, self.y), BATCH_SIZE=8, num_batches=NR_BATCHES_PER_THREAD, seed=None)
        batch_gen = MultiThreadedGenerator(batch_gen, NR_THREADS, NR_THREADS, seeds=[1234] * NR_THREADS)
        ctr = 0
        for batch in batch_gen:
            ctr += 1
        print("generated " + str(ctr) + " batches")

        self.assertEqual(ctr, NR_THREADS*NR_BATCHES_PER_THREAD, "number of batches do not match")