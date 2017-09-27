from __future__ import print_function

from future import standard_library

standard_library.install_aliases()
from builtins import range
from builtins import object
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
import numpy as np
import sys
import logging


class MultiThreadedGenerator(object):
    def __init__(self, generator, num_processes, num_cached_per_queue, seeds=None):
        if seeds is not None:
            assert len(seeds) == num_processes
        else:
            seeds = [None] * num_processes
        self.seeds = seeds
        self.generator = generator
        self.num_processes = num_processes
        self.num_cached_per_queue = num_cached_per_queue
        self._queues = []
        self._threads = []
        self._end_ctr = 0
        self._queue_loop = 0

    def __iter__(self):
        return self

    def _next_queue(self):
        r = self._queue_loop
        self._queue_loop += 1
        if self._queue_loop == self.num_processes:
            self._queue_loop = 0
        return r

    def __next__(self):
        if len(self._queues) == 0:
            self._start()
        try:
            item = self._queues[self._next_queue()].get()
            while item == "end":
                self._end_ctr += 1
                if self._end_ctr == self.num_processes:
                    logging.debug("MultiThreadedGenerator: finished data generation")
                    self._finish()
                    raise StopIteration

                item = self._queues[self._next_queue()].get()
            return item
        except KeyboardInterrupt:
            logging.error("MultiThreadedGenerator: caught exception: {}".format(sys.exc_info()))
            self._finish()
            raise KeyboardInterrupt

    def _start(self):
        if len(self._threads) == 0:
            print("starting workers")
            self._queue_loop = 0
            self._end_ctr = 0

            def producer(queue, generator):
                for item in generator:
                    queue.put(item)
                queue.put("end")

            for i in range(self.num_processes):
                np.random.seed(self.seeds[i])
                self._queues.append(MPQueue(self.num_cached_per_queue))
                self._threads.append(Process(target=producer, args=(self._queues[i], self.generator)))
                self._threads[-1].daemon = True
                self._threads[-1].start()
        else:
            logging.debug("MultiThreadedGenerator Warning: start() has been called but workers are already running")

    def _finish(self):
        if len(self._threads) != 0:
            logging.debug("MultiThreadedGenerator: workers terminated")
            for i, thread in enumerate(self._threads):
                thread.terminate()
                self._queues[i].close()
            self._queues = []
            self._threads = []
            self._queue = None
            self._end_ctr = 0
            self._queue_loop = 0

    def restart(self):
        self._finish()
        self._start()

    def __del__(self):
        logging.debug("MultiThreadedGenerator: destructor was called")
        self._finish()
