import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import unittest
from time import sleep, time

import numpy as np

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

AUGMENTERS = {'mta': MultiThreadedAugmenter, 'nondet': NonDetMultiThreadedAugmenter}


def _fail_to_unpickle():
    raise ValueError('this batch cannot be unpickled')


class Unpicklable:
    def __reduce__(self):
        return _fail_to_unpickle, ()


class UnpicklableDL(SlimDataLoaderBase):
    def generate_train_batch(self):
        return {'data': Unpicklable()}


class LargeBatchDL(SlimDataLoaderBase):
    def generate_train_batch(self):
        # larger than a pipe's buffer, so a worker's queue feeder blocks once nobody reads
        return {'data': np.zeros((1, 1, 512, 512), dtype=np.float32)}


class TestResultsThreadErrorsSurface(unittest.TestCase):
    """
    An error in the results thread (here: a batch that cannot be unpickled) must reach the caller of next() instead of
    leaving it waiting for batches forever, and must not be reported as dead workers.
    """

    def setUp(self):
        self.thread_errors = []
        self.addCleanup(setattr, threading, 'excepthook', threading.excepthook)
        threading.excepthook = lambda args: self.thread_errors.append(args.exc_type.__name__)

    def check_error_surfaces(self, augmenter):
        result = []
        messages = []

        def fetch():
            try:
                next(augmenter)
                result.append('returned')
            except RuntimeError as e:
                result.append('RuntimeError')
                messages.append(str(e))

        # start the workers here, forking from the fetcher thread would trigger python's fork() warning
        augmenter._start()
        fetcher = threading.Thread(target=fetch, daemon=True)
        fetcher.start()
        fetcher.join(timeout=10)
        try:
            self.assertEqual(result, ['RuntimeError'])
            self.assertNotIn('no longer alive', messages[0])
            self.assertIn('ValueError', self.thread_errors)
        finally:
            augmenter._finish()

    def test_multi_threaded_augmenter(self):
        self.check_error_surfaces(MultiThreadedAugmenter(UnpicklableDL(None, 1), None, 1, 1))

    def test_nondet_multi_threaded_augmenter(self):
        self.check_error_surfaces(NonDetMultiThreadedAugmenter(UnpicklableDL(None, 1), None, 1, 1))


def _alive(pid):
    try:
        with open('/proc/%d/stat' % pid) as f:
            return f.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


@unittest.skipUnless(sys.platform.startswith('linux'), 'reads process states from /proc')
class TestWorkersExitWhenParentDies(unittest.TestCase):
    """
    A parent that is killed (SIGKILL, the OOM killer) never tells its workers to stop. They must notice on their own
    instead of running on as orphans.
    """

    def check_workers_exit(self, augmenter, fork_sibling=False):
        args = [sys.executable, os.path.abspath(__file__), 'helper', augmenter]
        helper = subprocess.Popen(args + (['fork_sibling'] if fork_sibling else []), stdout=subprocess.PIPE, text=True)
        workers = []
        siblings = []
        try:
            pids = [int(i) for i in helper.stdout.readline().split()]
            workers, siblings = pids[:3], pids[3:]
            self.assertEqual(len(workers), 3)
            helper.send_signal(signal.SIGKILL)
            helper.wait()
            deadline = time() + 10
            while time() < deadline and any(_alive(w) for w in workers):
                sleep(0.1)
            self.assertEqual([w for w in workers if _alive(w)], [])
        finally:
            helper.kill()
            helper.wait()
            helper.stdout.close()
            for pid in workers + siblings:
                if _alive(pid):
                    os.kill(pid, signal.SIGKILL)

    def test_multi_threaded_augmenter(self):
        self.check_workers_exit('mta')

    def test_nondet_multi_threaded_augmenter(self):
        self.check_workers_exit('nondet')

    # A process forked after the workers holds copies of the pipes through which they watch the parent, and keeps
    # them open after the parent is killed.
    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'needs the fork start method')
    def test_multi_threaded_augmenter_with_forked_sibling(self):
        self.check_workers_exit('mta', fork_sibling=True)

    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'needs the fork start method')
    def test_nondet_multi_threaded_augmenter_with_forked_sibling(self):
        self.check_workers_exit('nondet', fork_sibling=True)


def run_helper(augmenter, fork_sibling):
    """
    Starts an augmenter with three workers, prints their pids and waits to be killed. With fork_sibling, everything
    is forked, and a process forked after the workers outlives the helper; its pid is printed last.
    """
    if fork_sibling:
        multiprocessing.set_start_method('fork', force=True)
    mt = AUGMENTERS[augmenter](LargeBatchDL(None, 1), None, 3, 1)
    next(mt)
    pids = [p.pid for p in mt._processes]
    if fork_sibling:
        sibling = multiprocessing.Process(target=sleep, args=(60,))
        sibling.start()
        pids.append(sibling.pid)
    print(' '.join(str(p) for p in pids), flush=True)
    sleep(60)


if __name__ == '__main__':
    if sys.argv[1:2] == ['helper']:
        run_helper(sys.argv[2], sys.argv[3:4] == ['fork_sibling'])
    else:
        unittest.main()
