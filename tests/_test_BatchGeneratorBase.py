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


from __future__ import absolute_import
from builtins import str
from builtins import range
import unittest
import numpy as np
import sys


class TestMultiThreading(unittest.TestCase):

    def setUp(self):
        self.seed = 1234
        np.random.seed(self.seed)
        self.x = np.random.rand(100, 2)
        self.y = np.random.rand(100, 1)

    def test_nr_of_batches(self):
        batch_gen = MockBatchGenerator((self.x, self.y), BATCH_SIZE=2, num_batches=100,
                                       seed=self.seed)
        ctr = 0
        for _ in batch_gen:
            ctr += 1

        self.assertEqual(ctr, 100, 'BatchGenerator should have generated 100 batches, but instead generated %d'%ctr)

    def test_unimited_batched(self):
        batch_gen = MockBatchGenerator((self.x, self.y), BATCH_SIZE=2, num_batches=None,
                                       seed=self.seed)
        try:
            for _ in range(5000):
                _ = next(batch_gen)
        except Exception:
            self.fail("Producing unlimited batches failed with msg: %s" % str(sys.exc_info()))

    def test_seeding_False(self):
        # seed=False will not trigger a reseeding of the global numpy rng by the generator
        batch_gen = MockBatchGenerator((self.x, self.y), BATCH_SIZE=1, num_batches=12,
                                       seed=False)
        results = []
        for batch in batch_gen:
            results.append(batch["data"])
        results = np.array(results)

        true_results = [[[ 0.47163253,  0.10712682]],
       [[ 0.55246894,  0.27304326]],
       [[ 0.54512217,  0.45125405]],
       [[ 0.63372577,  0.43830988]],
       [[ 0.52822428,  0.95142876]],
       [[ 0.05980922,  0.18428708]],
       [[ 0.43772774,  0.78535858]],
       [[ 0.7791638 ,  0.59915478]],
       [[ 0.05980922,  0.18428708]],
       [[ 0.77997581,  0.27259261]],
       [[ 0.08477384,  0.33300247]],
       [[ 0.07334254,  0.0550064 ]]]

        elementwise_equality = np.isclose(results, true_results)   #isclose comparison needed for float
        self.assertTrue(np.all(elementwise_equality), "Order and/or content of BatchGeneratorBase not reproducible")

    def test_seeding_None(self):
        # seed=False will not trigger a reseeding of the global numpy rng by the generator
        batch_gen = MockBatchGenerator((self.x, self.y), BATCH_SIZE=1, num_batches=12,
                                       seed=None)
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
        self.assertFalse(np.all(elementwise_equality), "Order and/or content of BatchGeneratorBase are reproducible but should not be")

    def test_seeding_value(self):
        # seed=False will not trigger a reseeding of the global numpy rng by the generator
        batch_gen = MockBatchGenerator((self.x, self.y), BATCH_SIZE=1, num_batches=12,
                                       seed=2168726)
        results = []
        for batch in batch_gen:
            results.append(batch["data"])
        results = np.array(results)

        true_results = [[[ 0.77997581,  0.27259261]],
       [[ 0.38231745,  0.05387369]],
       [[ 0.63372577,  0.43830988]],
       [[ 0.05980922,  0.18428708]],
       [[ 0.62277659,  0.49368265]],
       [[ 0.37025075,  0.56119619]],
       [[ 0.59341133,  0.3660745 ]],
       [[ 0.31683612,  0.56809865]],
       [[ 0.97724143,  0.55689468]],
       [[ 0.43689317,  0.612149  ]],
       [[ 0.64088043,  0.12620532]],
       [[ 0.04332406,  0.56143308]]]

        elementwise_equality = np.isclose(results, true_results)   #isclose comparison needed for float
        self.assertTrue(np.all(elementwise_equality), "Order and/or content of BatchGeneratorBase not reproducible")