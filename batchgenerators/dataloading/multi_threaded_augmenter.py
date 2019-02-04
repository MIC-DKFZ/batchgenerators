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
import threading
standard_library.install_aliases()
from builtins import range
from multiprocessing import Process
from multiprocessing import Queue
from queue import Queue as thrQueue
import numpy as np
import sys
import logging
from multiprocessing import Event
from queue import Empty, Full
import traceback
from time import sleep


def producer(queue, data_loader, transform, thread_id, seed, abort_event):
    try:
        np.random.seed(seed)
        data_loader.set_thread_id(thread_id)
        item = None

        while True:
            # check if abort event was set
            if not abort_event.is_set():

                if item is None:

                    try:
                        item = next(data_loader)
                        if transform is not None:
                            item = transform(**item)
                    except StopIteration:
                        item = "end"

                try:
                    queue.put(item, timeout=0.2)
                    item = None
                except Full:
                    # queue was full because items in it were not consumed. Try again.
                    pass
            else:
                # abort_event was set. Drain queue, then give 'end'
                break

    except KeyboardInterrupt:
        # drain queue, then give 'end', set abort flag and reraise KeyboardInterrupt
        abort_event.set()

        raise KeyboardInterrupt

    except Exception:
        print("Exception in worker", thread_id)
        traceback.print_exc()
        # drain queue, give 'end', send abort_event so that other workers know to exit

        abort_event.set()


def pin_memory_loop(in_queues, out_queue, abort_event):
    import torch
    queue_ctr = 0
    item = None
    while True:
        try:
            if not abort_event.is_set():
                if item is None:
                    item = in_queues[queue_ctr % len(in_queues)].get(timeout=0.2)
                    if isinstance(item, dict):
                        for k in item.keys():
                            if isinstance(item[k], torch.Tensor):
                                item[k] = item[k].pin_memory()
                    queue_ctr += 1
                out_queue.put(item, timeout=0.2)
                item = None
            else:
                print('pin_memory_loop exiting...')
                break
        except Empty:
            pass
        except Full:
            pass


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

        pin_memory (bool): set to True if all torch tensors in data_dict are to be pinned. Pytorch only.
    """
    def __init__(self, data_loader, transform, num_processes, num_cached_per_queue=2, seeds=None, pin_memory=False):
        self.pin_memory = pin_memory
        self.transform = transform
        if seeds is not None:
            assert len(seeds) == num_processes
        else:
            seeds = list(range(num_processes))
        self.seeds = seeds
        self.generator = data_loader
        self.num_processes = num_processes
        self.num_cached_per_queue = num_cached_per_queue
        self._queues = []
        self._processes = []
        self._end_ctr = 0
        self._queue_loop = 0
        self.pin_memory_thread = None
        self.pin_memory_queue = None
        self.abort_event = Event()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def _next_queue(self):
        r = self._queue_loop
        self._queue_loop += 1
        if self._queue_loop == self.num_processes:
            self._queue_loop = 0
        return r

    def __get_next_item(self):
        success = False
        item = None

        while not success:
            try:
                if self.abort_event.is_set():
                    self._finish()
                    raise RuntimeError("MultiThreadedAugmenter.abort_event was set, something went wrong. Maybe one of "
                                       "your workers crashed")
                else:
                    if not self.pin_memory:
                        item = self._queues[self._next_queue()].get(timeout=0.2)
                        success = True
                    else:
                        item = self.pin_memory_queue.get(timeout=0.2)
                        success = True
            except Empty:
                pass

        return item

    def __next__(self):
        if len(self._queues) == 0:
            self._start()
        try:
            item = self.__get_next_item()

            while item == "end":
                self._end_ctr += 1
                if self._end_ctr == self.num_processes:
                    self._end_ctr = 0
                    self._queue_loop = 0
                    logging.debug("MultiThreadedGenerator: finished data generation")
                    raise StopIteration

                item = self.__get_next_item()

            return item

        except KeyboardInterrupt:
            logging.error("MultiThreadedGenerator: caught exception: {}".format(sys.exc_info()))
            self._finish()
            raise KeyboardInterrupt

    def _start(self):
        if len(self._processes) == 0:
            self.abort_event.clear()

            logging.debug("starting workers")
            self._queue_loop = 0
            self._end_ctr = 0

            for i in range(self.num_processes):
                self._queues.append(Queue(self.num_cached_per_queue))
                self._processes.append(Process(target=producer, args=(self._queues[i], self.generator, self.transform, i, self.seeds[i], self.abort_event)))
                self._processes[-1].daemon = True
                self._processes[-1].start()

            if self.pin_memory:
                self.pin_memory_queue = thrQueue(2)
                self.pin_memory_thread = threading.Thread(target=pin_memory_loop, args=(self._queues, self.pin_memory_queue, self.abort_event))
                self.pin_memory_thread.daemon = True
                self.pin_memory_thread.start()
        else:
            logging.debug("MultiThreadedGenerator Warning: start() has been called but workers are already running")

    def _finish(self):
        self.abort_event.set()
        sleep(2) # allow pin memory thread to finish
        if len(self._processes) != 0:
            logging.debug("MultiThreadedGenerator: workers terminated")
            for i, p in enumerate(self._processes):
                p.terminate()

                self._queues[i].close()
                self._queues[i].join_thread()

            self._queues = []
            self._processes = []
            self._queue = None
            self._end_ctr = 0
            self._queue_loop = 0

    def restart(self):
        self._finish()
        self._start()

    def __del__(self):
        logging.debug("MultiThreadedGenerator: destructor was called")
        self._finish()
