import unittest
import numpy as np
from MockBatchGenerator import MockBatchGenerator
from DeepLearningBatchGeneratorUtils.MultiThreadedGenerator import MultiThreadedGenerator
import logging

class TestMultiThreading(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(format='%(levelname)s: %(message)s')  # set formatting of output
        logging.getLogger().setLevel(logging.WARNING)  # to see debug messages use logging.DEBUG

        self.seed = 1234
        np.random.seed(self.seed)
        self.x = np.random.rand(100, 2)
        self.y = np.random.rand(100, 1)

    def test_nr_of_batches(self):
        NR_THREADS = 5
        NR_BATCHES_PER_THREAD = 10

        batch_gen = MockBatchGenerator((self.x, self.y), BATCH_SIZE=8, num_batches=NR_BATCHES_PER_THREAD, seed=self.seed)
        batch_gen = MultiThreadedGenerator(batch_gen, NR_THREADS, NR_THREADS, seeds=[self.seed] * NR_THREADS)
        ctr = 0
        for batch in batch_gen:
            ctr += 1
        # print("generated " + str(ctr) + " batches")

        self.assertEqual(ctr, NR_THREADS*NR_BATCHES_PER_THREAD, "number of batches do not match")


    def test_seeding_of_multi_threaded(self):
        '''
        Test if seeding works.

        Test if
        1. multiple threads always return batches in same order
        2. random sampling of testing data always returns same data
        '''
        NR_THREADS = 4
        NR_BATCHES_PER_THREAD = 3

        batch_gen = MockBatchGenerator((self.x, self.y), BATCH_SIZE=1, num_batches=NR_BATCHES_PER_THREAD, seed=False) #seed has to be False here!
        # generate list with increasing seed e.g. [1234, 1235, 1236], so we have different seed for every thread (but reproducible)
        seeds = [seed+idx for idx, seed in enumerate([self.seed]*NR_THREADS)]
        batch_gen = MultiThreadedGenerator(batch_gen, NR_THREADS, NR_THREADS, seeds=seeds)
        results = []
        for batch in batch_gen:
            results.append(batch["data"])
        results = np.array(results)

        true_results = [[[ 0.43689317,  0.612149  ]],
                        [[ 0.39720258,  0.78873014]],
                        [[ 0.90179605,  0.70652816]],
                        [[ 0.79686718,  0.55776083]],
                        [[ 0.31683612,  0.56809865]],
                        [[ 0.32570741,  0.19361869]],
                        [[ 0.76160391,  0.91440311]],
                        [[ 0.17306723,  0.13402121]],
                        [[ 0.97449514,  0.66778691]],
                        [[ 0.56594464,  0.00676406]],
                        [[ 0.35781727,  0.50099513]],
                        [[ 0.22921857,  0.89996519]]]

        elementwise_equality = np.isclose(results, true_results)   #isclose comparison needed for float
        self.assertTrue(np.all(elementwise_equality), "Order and/or content of MultiThreadedGenerator not reproducible")

