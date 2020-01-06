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
from time import sleep, time
from threadpoolctl import threadpool_limits


def producer(queue, data_loader, transform, thread_id, seed, abort_event):
    np.random.seed(seed)
    data_loader.set_thread_id(thread_id)
    item = None

    while True:
        # check if abort event was set
        if not abort_event.is_set():
            #print("worder %d event not set" % thread_id)
            if item is None:

                try:
                    item = next(data_loader)
                    if transform is not None:
                        item = transform(**item)
                except StopIteration:
                    item = "end"

            try:
                queue.put(item, timeout=2)
                item = None
            except Full:
                # queue was full because items in it were not consumed. Try again.
                pass
        else:
            #print("worder %d event is now set, exiting" % thread_id)
            break


def pin_memory_loop(in_queues, out_queue, abort_event, gpu):
    import torch
    torch.cuda.set_device(gpu)
    # print("gpu", torch.cuda.current_device())
    queue_ctr = 0
    item = None
    while True:
        try:
            if not abort_event.is_set():
                if item is None:
                    item = in_queues[queue_ctr % len(in_queues)].get(timeout=2)
                    if isinstance(item, dict):
                        for k in item.keys():
                            if isinstance(item[k], torch.Tensor):
                                item[k] = item[k].pin_memory()
                    queue_ctr += 1
                out_queue.put(item, timeout=2)
                item = None
            else:
                # print('pin_memory_loop exiting...')
                return
        except Empty:
            pass
        except Full:
            pass
        except KeyboardInterrupt:
            abort_event.set()
            print('pin_memory_loop exiting (KeyboardInterrupt)')
            raise KeyboardInterrupt
        except ConnectionResetError:
            print('ConnectionResetError in pin_memory_loop. This can happen when workers are terminated. Don\'t worry')
            return
        except EOFError:
            print('EOFError in pin_memory_loop. This can happen when workers are terminated. Don\'t worry')
            return
        except Exception:
            print("Exception in pin_memory_loop")
            traceback.print_exc()
            abort_event.set()
            return


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
        timeout (int): How long do we wait for the background workers to do stuff? If timeout seconds have passed and
        self.__get_next_item still has not gotten an item from the workers we will perform a check whether all
        background workers are still alive. If all are alive we wait, if not we set the abort flag.
    """
    def __init__(self, data_loader, transform, num_processes, num_cached_per_queue=2, seeds=None, pin_memory=False,
                 timeout=300):
        self.timeout = timeout
        self.pin_memory = pin_memory
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
        self._processes = []
        self._end_ctr = 0
        self._queue_loop = 0
        self.pin_memory_thread = None
        self.pin_memory_queue = None
        self.abort_event = Event()
        self.pin_memory_abort_event = Event()

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

        tmp = time()

        use_this_queue = self._next_queue()

        while not success:
            try:
                if self.abort_event.is_set():
                    self._finish()
                    raise RuntimeError("MultiThreadedAugmenter.abort_event was set, something went wrong. Maybe one of "
                                       "your workers crashed")
                else:
                    if not self.pin_memory:
                        item = self._queues[use_this_queue].get(timeout=2)
                    else:
                        item = self.pin_memory_queue.get(timeout=2)

                    success = True

                tmp = time()
            except Empty:
                if time() - tmp > self.timeout:
                    # check if all workers are still alive
                    all_alive = all([i.is_alive() for i in self._processes])
                    if not all_alive:
                        print("###########################################\nsome background workers are missing!\n####################################")
                        self.abort_event.set()
                        self.pin_memory_abort_event.set()
                pass

        return item

    def __next__(self):
        if len(self._queues) == 0:
            self._start()
        try:
            item = self.__get_next_item()

            while isinstance(item, str) and (item == "end"):
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
            self.abort_event.set()
            self.pin_memory_abort_event.set()
            self._finish()
            raise KeyboardInterrupt

    def _start(self):
        if len(self._processes) == 0:
            self.abort_event.clear()
            self.pin_memory_abort_event.clear()

            logging.debug("starting workers")
            self._queue_loop = 0
            self._end_ctr = 0

            if hasattr(self.generator, 'was_initialized'):
                self.generator.was_initialized = False

            with threadpool_limits(limits=1, user_api="blas"):
                for i in range(self.num_processes):
                    self._queues.append(Queue(self.num_cached_per_queue))
                    self._processes.append(Process(target=producer, args=(self._queues[i], self.generator, self.transform, i, self.seeds[i], self.abort_event)))
                    self._processes[-1].daemon = True
                    self._processes[-1].start()

            if self.pin_memory:
                import torch
                self.pin_memory_queue = thrQueue(2)
                self.pin_memory_thread = threading.Thread(target=pin_memory_loop, args=(self._queues, self.pin_memory_queue, self.pin_memory_abort_event, torch.cuda.current_device()))
                self.pin_memory_thread.daemon = True
                self.pin_memory_thread.start()
        else:
            logging.debug("MultiThreadedGenerator Warning: start() has been called but workers are already running")

    def _finish(self, timeout=10):
        self.pin_memory_abort_event.set()
        self.abort_event.set()

        start = time()
        if self.pin_memory_thread is not None:
            while self.pin_memory_thread.is_alive() and start + timeout > time():
                sleep(0.2)

        if len(self._processes) != 0:
            logging.debug("MultiThreadedGenerator: shutting down workers...")
            [i.terminate() for i in self._processes]

            for i, p in enumerate(self._processes):
                self._queues[i].close()
                self._queues[i].join_thread()

            self._queues = []
            self._processes = []
            self._queue = None
            self._end_ctr = 0
            self._queue_loop = 0

            del self.pin_memory_queue

    def restart(self):
        self._finish()
        self._start()

    def __del__(self):
        logging.debug("MultiThreadedGenerator: destructor was called")
        self._finish()
