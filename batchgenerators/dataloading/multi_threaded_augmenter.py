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
from warnings import warn
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


def producer(queue, data_loader, transform, thread_id, seed):
    try:
        np.random.seed(seed)
        data_loader.set_thread_id(thread_id)
        while True:
            for item in data_loader:
                if transform is not None:
                    item = transform(**item)
                queue.put(item)
            queue.put("end")
    except KeyboardInterrupt:
        queue.put("end")
        raise KeyboardInterrupt


def pin_memory_loop(in_queues, out_queue):
    import torch
    queue_ctr = 0
    while True:
        item = in_queues[queue_ctr % len(in_queues)].get()
        if isinstance(item, dict):
            for k in item.keys():
                if isinstance(item[k], torch.Tensor):
                    item[k] = item[k].pin_memory()
        queue_ctr += 1
        out_queue.put(item)


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
        self._threads = []
        self._end_ctr = 0
        self._queue_loop = 0
        self.pin_memory_thread = None
        self.pin_memory_queue = None

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
            if not self.pin_memory:
                item = self._queues[self._next_queue()].get()
            else:
                item = self.pin_memory_queue.get()

            while item == "end":
                self._end_ctr += 1
                if self._end_ctr == self.num_processes:
                    self._end_ctr = 0
                    self._queue_loop = 0
                    logging.debug("MultiThreadedGenerator: finished data generation")
                    #self._finish()
                    raise StopIteration

                if not self.pin_memory:
                    item = self._queues[self._next_queue()].get()
                else:
                    item = self.pin_memory_queue.get()

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

            for i in range(self.num_processes):
                self._queues.append(Queue(self.num_cached_per_queue))
                self._threads.append(Process(target=producer, args=(self._queues[i], self.generator, self.transform, i, self.seeds[i])))
                self._threads[-1].daemon = True
                self._threads[-1].start()

            if self.pin_memory:
                self.pin_memory_queue = thrQueue(2)
                self.pin_memory_thread = threading.Thread(target=pin_memory_loop, args=(self._queues, self.pin_memory_queue))
                self.pin_memory_thread.daemon = True
                self.pin_memory_thread.start()
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


class TransformAdapter(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data_dict):
        return self.transform(**data_dict)


class ProcessTerminateOnJoin(Process):
    def join(self, timeout=None):
        self.terminate()
        super(ProcessTerminateOnJoin, self).join(0.01)


def default_joiner(items):
    keys = items[0].keys()
    res = {}
    for k in keys:
        c = []
        for i in items:
            c.append(i[k])
        if isinstance(items[0][k], np.ndarray):
            res[k] = np.vstack(c)
        elif isinstance(items[0][k], list):
            res[k] = c
        elif isinstance(items[0][k], tuple):
            res[k] = tuple(c)
        else:
            raise ValueError("don't know how to join instances of %s to a batch"%str(type(items[0][k])))
    return res


class AlternativeMultiThreadedAugmenter(object):
    def __init__(self, dataloader, num_processes, num_raw_cached, num_transformed_cached, batch_size, transform, batch_joiner=default_joiner, seeds=None, verbose=False):
        """
        This one has two advantages and two disadvantages over the MultiThreadedAugmenter. Whereas the 'old'
        MultiThreadedAugmenter copies the entire data augmentation pipeline (including the dataloader) for the
        processes, this one holds only one instance instance of the dataloader in a background process. That enables
        you to better control what samples are pushed through the pipeline (for example if you want to iterate once
        over your dataset). Also, this augmenter will process samples separately and therefore has better synergy with
        SpatialTransform which will either deform all samples in a batch or no samples at all.
        Advantages:
        1) you can better control what samples go through the pipeline (iterate once over your training data)
        2) samples are processed one at a time: SpatialTransform etc are not applied to either all of no samples within
        a batch
        Disatvantage:
        1) is is slower (sometimes substantially!)
        2) if you dataloader is slow, the entire Pipeline will be slow. Make sure your dataloader is lightning fast!!
        :param dataloader: must have batch size 1, batch size is controlled via AlternativeMultiThreadedAugmenter
        :param num_processes:
        :param num_raw_cached:
        :param num_transformed_cached:
        :param batch_size:
        :param transform:
        :param batch_joiner: this function needs to fuse samples back together into a batch. See default_joiner for an
        example
        :param seeds:
        :param verbose:
        """
        warn("This is deprecated. Please use MultiThreadedAugmenter instead. As of 0.18, all augmentations have a "
             "p_per_sample keyword, making augmentations on a per-sample basis in the multithreaded augmenter "
             "superfluous. Also, this one is much slower than MultiThreadedAugmenter",
             DeprecationWarning)
        self.verbose = verbose
        self.batch_joiner = batch_joiner
        self.num_transformed_cached = num_transformed_cached
        self.num_raw_cached = num_raw_cached
        self.transform = transform
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.num_processes = num_processes
        if seeds is None:
            seeds = [np.random.randint(999999) for i in range(num_processes)]
        self.seeds = seeds # TODO
        assert self.dataloader.BATCH_SIZE == 1, "batch size of dataloader must be 1!"
        self.was_started = False
        self.sample_queues = self.sample_generating_process = self.ready_queues = self.transformed_queues = None
        self.joiners = None
        self.q_ctr = None
        self.transformers = None
        self.joining_process = None

    def start(self):
        print("started")

        def producer(target_queues, data_loader):
            num_queues = len(target_queues)
            ctr = 0
            for item in data_loader:
                q = ctr % num_queues
                target_queues[q].put(item)
                ctr += 1
            [target_queue.put("end") for target_queue in target_queues]

        def transformer(target_queue, source_queue, transform, seed):
            np.random.seed(seed)
            item = source_queue.get()
            while item != "end":
                target_queue.put(transform(**item))
                item = source_queue.get()
            target_queue.put("end")

        def joiner(transformed_queues, ready_queues, join_method, batch_size):
            # collects and joins samples to batches
            stop = False
            num_queues = len(transformed_queues)
            ctr = 0
            num_rdy = len(ready_queues)
            rdy_ctr = 0
            while not stop:
                items = []
                for _ in range(batch_size):
                    q = ctr % num_queues
                    item = transformed_queues[q].get()
                    ctr += 1
                    if item == "end":
                        stop = True
                        break
                    items.append(item)
                if stop:
                    break
                else:
                    joined = join_method(items)
                    rdy_q = rdy_ctr % num_rdy
                    ready_queues[rdy_q].put(joined)
                    rdy_ctr += 1
            [ready_queue.put("end") for ready_queue in ready_queues]

        self.sample_queues = [Queue(self.num_raw_cached) for i in range(self.num_processes)]
        self.transformed_queues = [Queue(self.num_transformed_cached) for i in range(self.num_processes)]
        self.ready_queues = [Queue(2) for i in range(2)]

        self.sample_generating_process = ProcessTerminateOnJoin(target=producer, args=(self.sample_queues, self.dataloader))
        self.sample_generating_process.daemon = True
        self.sample_generating_process.start()

        self.joining_process = ProcessTerminateOnJoin(target=joiner, args=(self.transformed_queues, self.ready_queues, self.batch_joiner, self.batch_size))
        self.joining_process.daemon = True
        self.joining_process.start()

        self.q_ctr = 0

        self.transformers = []
        for i in range(self.num_processes):
            p = ProcessTerminateOnJoin(target=transformer, args=(self.transformed_queues[i], self.sample_queues[i], self.transform, self.seeds[i]))
            p.daemon = True
            p.start()
            self.transformers.append(p)

        self.was_started = True

    def __next__(self):
        if not self.was_started:
            self.start()
        if self.verbose:
            print("sample_queues", [i.qsize() for i in self.sample_queues])
            print("transformed_queues", [i.qsize() for i in self.transformed_queues])
            print("ready_queues", [i.qsize() for i in self.ready_queues])
        item = self.ready_queues[self.q_ctr % len(self.ready_queues)].get()
        self.q_ctr += 1
        if item == "end":
            raise StopIteration
        return item

    def __del__(self):
        self.sample_generating_process.join()
        self.joining_process.join()
        [i.join() for i in self.transformers]


if __name__ == "__main__":
    from batchgenerators.dataloading import DataLoaderBase
    from batchgenerators.transforms import SpatialTransform

    class DummyDL(DataLoaderBase):
        def __init__(self, num_threads_in_mt=8):
            super(DummyDL, self).__init__(None, None, None, False)
            self.num_threads_in_mt = num_threads_in_mt
            self._data = list(range(1000))
            self.current_position = 0

        def reset(self):
            super(DummyDL, self).reset()
            self.current_position = self.thread_id

        def generate_train_batch(self):
            idx = self.current_position
            if idx < len(self._data):
                self.current_position = idx + self.num_threads_in_mt
                return self._data[idx]
            else:
                raise StopIteration

    dl = DummyDL(num_threads_in_mt=3)
    mt = MultiThreadedAugmenter(dl, None, 3, 1, None)




    # ignore this code. this is work in progress
    from BraTS2018.dataset_loading.load_dataset import load_dataset
    from meddec.dataloading.dataset_loading import DataLoader3D
    import matplotlib.pyplot as plt
    import os
    dataset = load_dataset(os.path.join(os.environ['BraTS_2018_BASE'], "BraTS2018"))
    dl = DataLoader3D(dataset, (128, 128, 128), (64, 64, 64), 2)
    #tr = GaussianBlurTransform(3)
    tr = SpatialTransform((128, 128, 128), (64, 64, 64), False, do_rotation=False, do_scale=True, scale=(0.6, 0.60000001))
    from time import time
    num_batches = 10
    num_threads = 8

    mt = MultiThreadedAugmenter(dl, tr, num_threads, 2)

    # warm up
    warum_up_times_old = []
    for _ in range(6):
        a = time()
        b = next(mt)
        warum_up_times_old.append(time() - a)

    start = time()
    times_old = []
    for _ in range(num_batches):
        a = time()
        b = next(mt)
        times_old.append(time() - a)
    end = time()
    time_old = end - start

    dl = DataLoader3D(dataset, (128, 128, 128), (64, 64, 64), 1)
    mt = AlternativeMultiThreadedAugmenter(dl, num_threads, 3, 6, 2, tr, verbose=True)

    # warm up
    warum_up_times_new5 = []
    for _ in range(6):
        a = time()
        b = next(mt)
        warum_up_times_new5.append(time() - a)

    start = time()
    times_new5 = []
    for _ in range(num_batches):
        a = time()
        b = next(mt)
        times_new5.append(time() - a)
    end = time()
    time_new5 = end - start


    plt.ion()
    plt.plot(range(len(times_old)), times_old, color="r", ls="-")
    plt.plot(range(len(times_new5)), times_new5, color="black", ls="-")
    plt.title("time per example")
    plt.legend(["old, total: %f s" % time_old, "new, total: %f s" % time_new5])


