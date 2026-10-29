import multiprocessing
import os
import signal
import unittest

import numpy as np

from batchgenerators.dataloading import multi_threaded_augmenter, nondet_multi_threaded_augmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase


class ZerosDL(SlimDataLoaderBase):
    def generate_train_batch(self):
        return {'data': np.zeros((self.batch_size, 1, 8, 8), dtype=np.float32)}


@unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'needs the fork start method')
class TestWorkersDieOnSIGTERM(unittest.TestCase):
    """
    Forked workers inherit the parent's signal handlers. A parent that catches SIGTERM (to shut down cleanly) must not
    keep its workers alive when a scheduler such as SLURM sends SIGTERM to every process of the job.
    """

    def setUp(self):
        self.addCleanup(signal.signal, signal.SIGTERM, signal.getsignal(signal.SIGTERM))
        signal.signal(signal.SIGTERM, lambda *args: None)

    def check_producer(self, producer):
        ctx = multiprocessing.get_context('fork')
        queue = ctx.Queue(maxsize=1)
        abort_event = ctx.Event()
        worker = ctx.Process(target=producer, args=(queue, ZerosDL(None, 2), None, 0, 1234, abort_event), daemon=True)
        worker.start()
        try:
            # the first item means the producer is past its signal setup
            queue.get(timeout=30)
            os.kill(worker.pid, signal.SIGTERM)
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        finally:
            abort_event.set()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join()

    def test_multi_threaded_augmenter(self):
        self.check_producer(multi_threaded_augmenter.producer)

    def test_nondet_multi_threaded_augmenter(self):
        self.check_producer(nondet_multi_threaded_augmenter.producer)


if __name__ == '__main__':
    unittest.main()
