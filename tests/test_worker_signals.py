import multiprocessing
import os
import signal
import unittest

from batchgenerators.dataloading import multi_threaded_augmenter, nondet_multi_threaded_augmenter
from tests.DataGenerators import ZerosDataLoader
from tests.augmenter_test_utils import forked_producer


@unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'needs the fork start method')
class TestWorkersDieOnSIGTERM(unittest.TestCase):
    """
    Forked workers inherit the parent's signal handlers. A parent that catches SIGTERM (to shut down cleanly) must not
    keep its workers alive when a scheduler such as SLURM sends SIGTERM to every process of the job.
    """

    def setUp(self):
        self.addCleanup(signal.signal, signal.SIGTERM, signal.getsignal(signal.SIGTERM))
        signal.signal(signal.SIGTERM, lambda *args: None)

    def _check_producer(self, producer):
        with forked_producer(producer, ZerosDataLoader((2, 1, 8, 8)), 0, 1234) as (worker, queue):
            # the first item means the producer is past its signal setup
            queue.get(timeout=30)
            os.kill(worker.pid, signal.SIGTERM)
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())

    def test_multi_threaded_augmenter(self):
        self._check_producer(multi_threaded_augmenter.producer)

    def test_nondet_multi_threaded_augmenter(self):
        self._check_producer(nondet_multi_threaded_augmenter.producer)


if __name__ == '__main__':
    unittest.main()
