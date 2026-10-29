import multiprocessing
import threading
from contextlib import contextmanager


def capture_thread_errors(test_case):
    """Returns a list that collects the names of the exceptions that escape a thread until test_case ends."""
    errors = []
    test_case.addCleanup(setattr, threading, 'excepthook', threading.excepthook)
    threading.excepthook = lambda args: errors.append(args.exc_type.__name__)
    return errors


@contextmanager
def forked_producer(producer, data_loader, thread_id, seed):
    """Runs an augmenter's producer in a forked worker with a queue of size 1. Yields the worker and the queue."""
    ctx = multiprocessing.get_context('fork')
    queue = ctx.Queue(maxsize=1)
    abort_event = ctx.Event()
    worker = ctx.Process(target=producer, args=(queue, data_loader, None, thread_id, seed, abort_event), daemon=True)
    worker.start()
    try:
        yield worker, queue
    finally:
        abort_event.set()
        worker.join(timeout=5)
        if worker.is_alive():
            worker.kill()
            worker.join()
