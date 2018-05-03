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
from multiprocessing import Pool


class MultiThreadedAugmenter(object):
    """ Makes your pipeline multi threaded. Yeah!

    If seeded we guarantee that batches are retunred in the same order and with the same augmentation every time this
    is run. This is realized internally by using une queue per worker and querying the queues one ofter the other.

    Args:
        data_loader (generator or DataLoaderBase instance): Your data loader. Must have a .next() function and return
        a dict that complies with our data structure

        transform (Transform instance): Any of our transformations. If you want to use multiple transformations then
        use our Compose transform! Can be None (in that case no transform will be applied)

        num_processes (int): number of processes

        num_cached_per_queue (int): number of batches cached per process (each process has its own
        multiprocessing.Queue). We found 2 to be ideal.

        seeds (list of int): one seed for each worker. Must have len(num_processes).
        If None then seeds = range(num_processes)
    """
    def __init__(self, data_loader, transform, num_processes, num_cached_per_queue=2, seeds=None):
        self.transform = transform
        if seeds is not None:
            assert len(seeds) == num_processes
        else:
            seeds = [None] * num_processes
        self.seeds = seeds
        self.generator = data_loader
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
            logging.debug("starting workers")
            self._queue_loop = 0
            self._end_ctr = 0

            def producer(queue, data_loader, transform):
                for item in data_loader:
                    if transform is not None:
                        item = transform(**item)
                    queue.put(item)
                queue.put("end")

            for i in range(self.num_processes):
                np.random.seed(self.seeds[i])
                self._queues.append(MPQueue(self.num_cached_per_queue))
                self._threads.append(Process(target=producer, args=(self._queues[i], self.generator, self.transform)))
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


class ProperMultiThreadedAugmenter(object):
    def __init__(self, dataloader, num_processes, num_cached, batch_size, transform, seeds=None):
        raise NotImplementedError("Work in progress, do not use! It does not work")
        self.transform = transform
        self.batch_size = batch_size
        self.num_cached = num_cached
        self.dataloader = dataloader
        self.num_processes = num_processes
        if seeds is None:
            seeds = [np.random.randint(999999) for i in range(num_processes)]
        self.seeds = seeds # TODO
        assert self.dataloader.BATCH_SIZE == 1, "batch size of dataloader must be 1!"
        self.sample_generating_process = None
        self.transformed_queue_filler = None

    def start(self):
        def sample_producer(queue, data_loader):
            try:
                for item in data_loader:
                    queue.put(item)
                queue.put("end")
            except KeyboardInterrupt:
                queue.put("end")

        self.sample_queue = MPQueue(self.num_cached)
        self.sample_generating_process = Process(target=sample_producer, args=(self.sample_queue, self.dataloader))
        self.sample_generating_process.start()

        self.transformed_queue = MPQueue(self.num_cached)

        def fill_transform_queue(sample_queue, transform, target_queue, batch_size, num_processes):
            pool = Pool(num_processes)
            try:
                stop = False
                while not stop:
                    items = []
                    for i in range(batch_size):
                        item = sample_queue.get()
                        if item == "end":
                            stop = True
                            target_queue.put("end")
                            break
                        items.append(item)
                    if stop:
                        target_queue.put("end")
                        continue
                    transformed = [pool.apply(transform, kwds=items[i]) for i in range(batch_size)]
                    target_queue.put(transformed)
            except Exception:
                pool.terminate()
                raise Exception

        self.transformed_queue_filler = Process(target=fill_transform_queue, args=(self.sample_queue, self.transform,
                                                                                   self.transformed_queue,
                                                                                   self.batch_size, self.num_processes))
        self.transformed_queue_filler.start()

    def __next__(self):
        if not (isinstance(self.transformed_queue_filler, Process) and self.transformed_queue_filler.is_alive() and
                isinstance(self.sample_generating_process, Process) and self.sample_generating_process.is_alive()):
            self.start()
        item = self.transformed_queue.get()
        if item == "end":
            raise StopIteration
        return self.dataloader.join(item)


if __name__ == "__main__":
    class Dataloader(object):
        def __init__(self):
            self.BATCH_SIZE = 1
            self.ctr = 0

        def __next__(self):
            self.ctr += 1
            return {"ctr":self.ctr}

        def __iter__(self):
            return self

        @staticmethod
        def join(items):
            return {"ctrs":[i['ctr'] for i in items]}

    class Transform():
        def __call__(self, **dct):
            dct['ctr'] /= 10
            return dct

    dl = Dataloader()
    tr = Transform()
    mt = ProperMultiThreadedAugmenter(dl, 2, 2, 3, tr, None)

