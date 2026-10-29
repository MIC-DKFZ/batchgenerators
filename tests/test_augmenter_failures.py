import threading
import unittest

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter


def _fail_to_unpickle():
    raise ValueError('this batch cannot be unpickled')


class Unpicklable:
    def __reduce__(self):
        return _fail_to_unpickle, ()


class UnpicklableDL(SlimDataLoaderBase):
    def generate_train_batch(self):
        return {'data': Unpicklable()}


class TestResultsThreadErrorsSurface(unittest.TestCase):
    """
    An error in the results thread (here: a batch that cannot be unpickled) must reach the caller of next() instead of
    leaving it waiting for batches forever.
    """

    def setUp(self):
        self.thread_errors = []
        self.addCleanup(setattr, threading, 'excepthook', threading.excepthook)
        threading.excepthook = lambda args: self.thread_errors.append(args.exc_type.__name__)

    def check_error_surfaces(self, augmenter):
        result = []

        def fetch():
            try:
                next(augmenter)
                result.append('returned')
            except RuntimeError:
                result.append('RuntimeError')

        # start the workers here, forking from the fetcher thread would trigger python's fork() warning
        augmenter._start()
        fetcher = threading.Thread(target=fetch, daemon=True)
        fetcher.start()
        fetcher.join(timeout=10)
        try:
            self.assertEqual(result, ['RuntimeError'])
            self.assertIn('ValueError', self.thread_errors)
        finally:
            augmenter._finish()

    def test_multi_threaded_augmenter(self):
        self.check_error_surfaces(MultiThreadedAugmenter(UnpicklableDL(None, 1), None, 1, 1))

    def test_nondet_multi_threaded_augmenter(self):
        self.check_error_surfaces(NonDetMultiThreadedAugmenter(UnpicklableDL(None, 1), None, 1, 1))


if __name__ == '__main__':
    unittest.main()
