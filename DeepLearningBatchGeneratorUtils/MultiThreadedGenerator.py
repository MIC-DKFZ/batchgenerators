__author__ = 'Fabian Isensee'
# LICENCE FABIAN ONLY: HANDS OFF
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
import numpy as np
import sys

'''class MultiThreadedGenerator(object):
    def __init__(self, generator, num_processes, num_cached):
        self.generator = generator
        self.num_processes = num_processes
        self.num_cached = num_cached
        self._queue = None
        self._threads = []
        self.__end_ctr = 0

    def __iter__(self):
        return self

    def next(self):
            if self._queue is None:
                self._start()
            item = self._queue.get()
            while item == "end":
                self.__end_ctr += 1
                if self.__end_ctr == self.num_processes:
                    print "MultiThreadedGenerator: finished data generation"
                    self._finish()
                    raise StopIteration
                try:
                    item = self._queue.get()
                except:
                    print "MultiThreadedGenerator: caught exception:", sys.exc_info()[0]
                    self._finish()
                    raise StopIteration
            return item

    def _start(self):
        if len(self._threads) == 0:
            print "starting workers"
            self._queue = MPQueue(self.num_cached)

            def producer(queue, generator):
                try:
                    for item in generator:
                        queue.put(item)
                except:
                    print "aborted worker (caught exception)"
                finally:
                    queue.put("end")

            for _ in xrange(self.num_processes):
                np.random.seed()
                self._threads.append(Process(target=producer, args=(self._queue, self.generator)))
                self._threads[-1].daemon = True
                self._threads[-1].start()
        else:
            print "MultiThreadedGenerator Warning: start() has been called but workers are already running"

    def _finish(self):
        if len(self._threads) != 0:
            print "MultiThreadedGenerator: workers terminated"
            self._queue.close()
            for thread in self._threads:
                #if thread.is_alive():
                thread.terminate()
            self._threads = []
            self._queue = None
            self.__end_ctr = 0

    def __del__(self):
        print "MultiThreadedGenerator: destructor was called"
        self._finish()'''

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

    def next(self):
            if len(self._queues) == 0:
                self._start()
            try:
                item = self._queues[self._next_queue()].get()
                while item == "end":
                    self._end_ctr += 1
                    if self._end_ctr == self.num_processes:
                        print "MultiThreadedGenerator: finished data generation"
                        self._finish()
                        raise StopIteration

                    item = self._queues[self._next_queue()].get()
                return item
            except KeyboardInterrupt:
                print "MultiThreadedGenerator: caught exception:", sys.exc_info()
                self._finish()
                raise KeyboardInterrupt

    def _start(self):
        if len(self._threads) == 0:
            print "starting workers"
            self._queue_loop = 0
            self._end_ctr = 0

            def producer(queue, generator):
                for item in generator:
                    queue.put(item)
                queue.put("end")

            for i in xrange(self.num_processes):
                np.random.seed(self.seeds[i])
                self._queues.append(MPQueue(self.num_cached_per_queue))
                self._threads.append(Process(target=producer, args=(self._queues[i], self.generator)))
                self._threads[-1].daemon = True
                self._threads[-1].start()
        else:
            print "MultiThreadedGenerator Warning: start() has been called but workers are already running"

    def _finish(self):
        if len(self._threads) != 0:
            print "MultiThreadedGenerator: workers terminated"
            for i, thread in enumerate(self._threads):
                #if thread.is_alive():
                thread.terminate()
                self._queues[i].close()
            self._queues = []
            self._threads = []
            self._queue = None
            self._end_ctr = 0
            self._queue_loop = 0

    def __del__(self):
        print "MultiThreadedGenerator: destructor was called"
        self._finish()
