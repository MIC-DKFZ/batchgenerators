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
from copy import deepcopy
import numpy as np
from batchgenerators.dataloading import DataLoader
from batchgenerators.dataloading import MultiThreadedAugmenter


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

                assert len(idx) == len(data)
                assert all([i == j for i,j in zip(idx, data)])

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

                assert len(idx) == len(data)

                assert not all([i == j for i, j in zip(idx, data)])

                idx.sort()
                assert all([i == j for i,j in zip(idx, data)])

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

        assert total == 120
        assert ctr == 10

        dl = DummyDataLoader(deepcopy(data), batch_size, 1, 1, return_incomplete=True, shuffle=False, infinite=False)
        # this should now not raise a StopIteration anymore
        total = 0
        ctr = 0
        for i in dl:
            ctr += 1
            total += len(i)

        assert total == 123
        assert ctr == 11

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

                assert len(idx) == len(data)
                assert all([i == j for i,j in zip(idx, data)])

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

                assert len(idx) == len(data)

                assert not all([i == j for i, j in zip(idx, data)])

                idx.sort()
                assert all([i == j for i,j in zip(idx, data)])

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
        # this should now not raise a StopIteration anymore
        total = 0
        ctr = 0
        for i in mt:
            ctr += 1
            assert len(i) == batch_size
            total += batch_size

        assert total == 120
        assert ctr == 10

        dl = DummyDataLoader(deepcopy(data), batch_size, num_workers, 1, return_incomplete=True, shuffle=False, infinite=False)
        mt = MultiThreadedAugmenter(dl, None, num_workers, 1, None, False)
        # this should now not raise a StopIteration anymore
        total = 0
        ctr = 0
        for i in mt:
            ctr += 1
            total += len(i)

        assert total == 123
        assert ctr == 11


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    unittest.main()
