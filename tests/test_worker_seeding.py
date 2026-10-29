import multiprocessing
import unittest

import numpy as np

from batchgenerators.dataloading import multi_threaded_augmenter, nondet_multi_threaded_augmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

try:
    import torch
except ImportError:
    torch = None


class TorchRandDL(SlimDataLoaderBase):
    def generate_train_batch(self):
        return {'torch': torch.rand(4).numpy(), 'numpy': np.random.rand(4)}


@unittest.skipIf(torch is None, 'needs torch')
@unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'needs the fork start method')
class TestWorkerSeeding(unittest.TestCase):
    """
    Forked workers inherit the parent's torch RNG state. Augmentations that draw from torch must still differ between
    workers when no seeds are given, and must be reproducible when they are.
    """

    def first_items(self, producer, seeds):
        ctx = multiprocessing.get_context('fork')
        items = []
        for thread_id, seed in enumerate(seeds):
            queue = ctx.Queue(maxsize=1)
            abort_event = ctx.Event()
            worker = ctx.Process(target=producer, args=(queue, TorchRandDL(None, 1), None, thread_id, seed, abort_event),
                                 daemon=True)
            worker.start()
            try:
                items.append(queue.get(timeout=30))
            finally:
                abort_event.set()
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
        return items

    def check_producer(self, producer):
        torch.manual_seed(0)
        unseeded = self.first_items(producer, [None, None, None])
        for i in range(1, 3):
            self.assertFalse(np.array_equal(unseeded[0]['torch'], unseeded[i]['torch']))
            self.assertFalse(np.array_equal(unseeded[0]['numpy'], unseeded[i]['numpy']))

        seeded = [self.first_items(producer, [1, 2, 3]) for _ in range(2)]
        self.assertFalse(np.array_equal(seeded[0][0]['torch'], seeded[0][1]['torch']))
        for a, b in zip(*seeded):
            np.testing.assert_array_equal(a['torch'], b['torch'])
            np.testing.assert_array_equal(a['numpy'], b['numpy'])

    def test_multi_threaded_augmenter(self):
        self.check_producer(multi_threaded_augmenter.producer)

    def test_nondet_multi_threaded_augmenter(self):
        self.check_producer(nondet_multi_threaded_augmenter.producer)


if __name__ == '__main__':
    unittest.main()
