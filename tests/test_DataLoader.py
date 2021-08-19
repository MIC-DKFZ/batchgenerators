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
from copy import deepcopy
import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter


class DummyDataLoader(DataLoader):
    def __init__(self, data, batch_size, num_threads_in_multithreaded, seed_for_shuffle=1, return_incomplete=False,
                 shuffle=True, infinite=False):
        super(DummyDataLoader, self).__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        self.indices = data

    def generate_train_batch(self):
        idx = self.get_indices()
        return idx


class TestDataLoader(unittest.TestCase):
    def test_return_all_indices_single_threaded_shuffle_False(self):
        data = list(range(123))
        batch_sizes = [1, 3, 75, 12, 23]

        for b in batch_sizes:
            dl = DummyDataLoader(deepcopy(data), b, 1, 1, return_incomplete=True, shuffle=False, infinite=False)

            for _ in range(3):
                idx = []
                for i in dl:
                    idx += i

                self.assertTrue(len(idx) == len(data))
                self.assertTrue(all([i == j for i,j in zip(idx, data)]))

    def test_return_all_indices_single_threaded_shuffle_True(self):
        data = list(range(123))
        batch_sizes = [1, 3, 75, 12, 23]
        np.random.seed(1234)

        for b in batch_sizes:
            dl = DummyDataLoader(deepcopy(data), b, 1, 1, return_incomplete=True, shuffle=True, infinite=False)

            for _ in range(3):
                idx = []
                for i in dl:
                    idx += i

                self.assertTrue(len(idx) == len(data))

                self.assertTrue(not all([i == j for i, j in zip(idx, data)]))

                idx.sort()
                self.assertTrue(all([i == j for i,j in zip(idx, data)]))

    def test_infinite_single_threaded(self):
        data = list(range(123))

        dl = DummyDataLoader(deepcopy(data), 12, 1, 1, return_incomplete=True, shuffle=True, infinite=False)
        # this should raise a StopIteration
        with self.assertRaises(StopIteration):
            for i in range(1000):
                idx = next(dl)

        dl = DummyDataLoader(deepcopy(data), 12, 1, 1, return_incomplete=True, shuffle=True, infinite=True)
        # this should now not raise a StopIteration anymore
        for i in range(1000):
            idx = next(dl)

    def test_return_incomplete_single_threaded(self):
        data = list(range(123))
        batch_size = 12

        dl = DummyDataLoader(deepcopy(data), batch_size, 1, 1, return_incomplete=False, shuffle=False, infinite=False)
        # this should now not raise a StopIteration anymore
        total = 0
        ctr = 0
        for i in dl:
            ctr += 1
            assert len(i) == batch_size
            total += batch_size

        self.assertTrue(total == 120)
        self.assertTrue(ctr == 10)

        dl = DummyDataLoader(deepcopy(data), batch_size, 1, 1, return_incomplete=True, shuffle=False, infinite=False)
        # this should now not raise a StopIteration anymore
        total = 0
        ctr = 0
        for i in dl:
            ctr += 1
            total += len(i)

        self.assertTrue(total == 123)
        self.assertTrue(ctr == 11)

    def test_return_all_indices_multi_threaded_shuffle_False(self):
        data = list(range(123))
        batch_sizes = [1, 3, 75, 12, 23]
        num_workers = 3

        for b in batch_sizes:
            dl = DummyDataLoader(deepcopy(data), b, num_workers, 1, return_incomplete=True, shuffle=False, infinite=False)
            mt = MultiThreadedAugmenter(dl, None, num_workers, 1, None, False)

            for _ in range(3):
                idx = []
                for i in mt:
                    idx += i

                self.assertTrue(len(idx) == len(data))
                self.assertTrue(all([i == j for i,j in zip(idx, data)]))

    def test_return_all_indices_multi_threaded_shuffle_True(self):
        data = list(range(123))
        batch_sizes = [1, 3, 75, 12, 23]
        num_workers = 3

        for b in batch_sizes:
            dl = DummyDataLoader(deepcopy(data), b, num_workers, 1, return_incomplete=True, shuffle=True, infinite=False)
            mt = MultiThreadedAugmenter(dl, None, num_workers, 1, None, False)

            for _ in range(3):
                idx = []
                for i in mt:
                    idx += i

                self.assertTrue(len(idx) == len(data))

                self.assertTrue(not all([i == j for i, j in zip(idx, data)]))

                idx.sort()
                self.assertTrue(all([i == j for i,j in zip(idx, data)]))

    def test_infinite_multi_threaded(self):
        data = list(range(123))
        num_workers = 3

        dl = DummyDataLoader(deepcopy(data), 12, num_workers, 1, return_incomplete=True, shuffle=True, infinite=False)
        mt = MultiThreadedAugmenter(dl, None, num_workers, 1, None, False)

        # this should raise a StopIteration
        with self.assertRaises(StopIteration):
            for i in range(1000):
                idx = next(mt)

        dl = DummyDataLoader(deepcopy(data), 12, num_workers, 1, return_incomplete=True, shuffle=True, infinite=True)
        mt = MultiThreadedAugmenter(dl, None, num_workers, 1, None, False)
        # this should now not raise a StopIteration anymore
        for i in range(1000):
            idx = next(mt)

    def test_return_incomplete_multi_threaded(self):
        data = list(range(123))
        batch_size = 12
        num_workers = 3

        dl = DummyDataLoader(deepcopy(data), batch_size, num_workers, 1, return_incomplete=False, shuffle=False, infinite=False)
        mt = MultiThreadedAugmenter(dl, None, num_workers, 1, None, False)
        all_return = []
        total = 0
        ctr = 0
        for i in mt:
            ctr += 1
            assert len(i) == batch_size
            total += len(i)
            all_return += i

        self.assertTrue(total == 120)
        self.assertTrue(ctr == 10)
        self.assertTrue(len(np.unique(all_return)) == total)

        dl = DummyDataLoader(deepcopy(data), batch_size, num_workers, 1, return_incomplete=True, shuffle=False, infinite=False)
        mt = MultiThreadedAugmenter(dl, None, num_workers, 1, None, False)
        all_return = []
        total = 0
        ctr = 0
        for i in mt:
            ctr += 1
            total += len(i)
            all_return += i

        self.assertTrue(total == 123)
        self.assertTrue(ctr == 11)
        self.assertTrue(len(np.unique(all_return)) == len(data))

    def test_thoroughly(self):
        data_list = [list(range(123)),
            list(range(1243)),
            list(range(1)),
            list(range(7)),
                     ]
        worker_list = (1, 4, 7)
        batch_size_list = (1, 3, 32)
        seed_list = [318, None]
        epochs = 3

        for data in data_list:
            #print('data', len(data))
            for num_workers in worker_list:
                #print('num_workers', num_workers)
                for batch_size in batch_size_list:
                    #print('batch_size', batch_size)
                    for return_incomplete in [True, False]:
                        #print('return_incomplete', return_incomplete)
                        for shuffle in [True, False]:
                            #print('shuffle', shuffle)
                            for seed_for_shuffle in seed_list:
                                #print('seed_for_shuffle', seed_for_shuffle)
                                if return_incomplete:
                                    if len(data) % batch_size == 0:
                                        expected_num_batches = len(data) // batch_size
                                    else:
                                        expected_num_batches = len(data) // batch_size + 1
                                else:
                                    expected_num_batches = len(data) // batch_size

                                expected_num_items = len(data) if return_incomplete else expected_num_batches * batch_size

                                print("init")
                                dl = DummyDataLoader(deepcopy(data), batch_size, num_workers, seed_for_shuffle,
                                                     return_incomplete=return_incomplete, shuffle=shuffle,
                                                     infinite=False)

                                mt = MultiThreadedAugmenter(dl, None, num_workers, 5, None, False, wait_time=0)
                                mt._start()

                                for epoch in range(epochs):
                                    print("sampling")
                                    all_return = []
                                    total = 0
                                    ctr = 0
                                    for i in mt:
                                        ctr += 1
                                        total += len(i)
                                        all_return += i

                                    print('asserting')
                                    self.assertTrue(total == expected_num_items)
                                    self.assertTrue(ctr == expected_num_batches)
                                    self.assertTrue(len(np.unique(all_return)) == expected_num_items)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    unittest.main()
