import threading
import unittest
from time import sleep

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

try:
    import torch
except ImportError:
    torch = None


class TensorDL(SlimDataLoaderBase):
    def __init__(self, n_tensors, delay):
        super().__init__(None, 1)
        self.n_tensors = n_tensors
        self.delay = delay

    def generate_train_batch(self):
        sleep(self.delay)
        return {'t%d' % i: torch.zeros(10) for i in range(self.n_tensors)}


@unittest.skipIf(torch is None, 'needs torch')
class TestShutdownWithTensorBatches(unittest.TestCase):
    """
    Torch tensors travel between processes as file descriptors that the receiving process fetches from the sender
    while unpickling. A worker can still put a batch during shutdown and exit right after, so the results thread may
    fail to load it. That batch is discarded anyway, and the error must not escape the thread. With one worker and a
    queue size of 1, these batch timings hit that window in nearly every shutdown.
    """

    def setUp(self):
        self.thread_errors = []
        self.addCleanup(setattr, threading, 'excepthook', threading.excepthook)
        threading.excepthook = lambda args: self.thread_errors.append(args.exc_type.__name__)

    def check_shutdowns(self, make_augmenter):
        for _ in range(10):
            augmenter = make_augmenter()
            next(augmenter)
            augmenter._finish()
        self.assertEqual(self.thread_errors, [])

    def test_multi_threaded_augmenter(self):
        self.check_shutdowns(lambda: MultiThreadedAugmenter(TensorDL(1, 0.01), None, 1, 1))

    def test_nondet_multi_threaded_augmenter(self):
        self.check_shutdowns(lambda: NonDetMultiThreadedAugmenter(TensorDL(4, 0), None, 1, 1))


if __name__ == '__main__':
    unittest.main()
