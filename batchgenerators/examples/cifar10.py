import re
import torch
from collections import OrderedDict
from urllib.request import urlretrieve
from batchgenerators.dataloading import SlimDataLoaderBase, MultiThreadedAugmenter
import numpy as np
import os
import tarfile
import shutil
from abc import ABCMeta, abstractmethod
from batchgenerators.transforms.spatial_transforms import SpatialTransform

from batchgenerators.transforms import NumpyToTensor, Compose
from torch._six import int_classes, string_classes, container_abcs
from torch.utils.data.dataloader import numpy_type_map


_use_shared_memory = False


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size

    this is a copy default_collate from pytorch. The reason this is here is because we need torch.cat instead of
    torch.stack in if isinstance(batch[0], torch.Tensor)"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def unpickle(file):
    '''
    taken from http://www.cs.toronto.edu/~kriz/cifar.html
    :param file:
    :return:
    '''
    import pickle

    with open(file, 'rb') as fo:
        dc = pickle.load(fo, encoding='bytes')
    return dc


def maybe_download_and_prepare_cifar10(target_dir):
    '''
    Checks if cifar10 is already present in target_dir and downloads it if not.
    CIFAR10 comes in 5 batches that need to be unpickled. What a mess.
    We stack all 5 batches together to one single npy array. No idea why they are being so complicated
    :param target_dir:
    :return:
    '''
    if not os.path.isfile(os.path.join(target_dir, 'cifar10_test_data.npz')) or not \
            os.path.isfile(os.path.join(target_dir, 'cifar10_training_data.npz')):
        print('downloading CIFAR10...')
        urlretrieve('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', target_dir)

        tar = tarfile.open(os.path.join(target_dir, 'cifar-10-python.tar.gz'), "r:gz")
        tar.extractall()
        tar.close()

        data = []
        labels = []
        filenames = []

        for batch in range(1, 6):
            loaded = unpickle(os.path.join(target_dir, 'cifar-10-batches-py', 'data_batch_%d' % batch))
            data.append(loaded[b'data'].reshape((loaded[b'data'].shape[0], 3, 32, 32)).astype(np.uint8))
            labels += [int(i) for i in loaded[b'labels']]
            filenames += [str(i) for i in loaded[b'filenames']]

        data = np.vstack(data)
        labels = np.array(labels)
        filenames = np.array(filenames)

        np.savez_compressed(os.path.join(target_dir, 'cifar10_training_data.npz'), data=data, labels=labels,
                            filenames=filenames)

        test = unpickle(os.path.join(target_dir, 'cifar-10-batches-py', 'test_batch'))
        data = test[b'data'].reshape((test[b'data'].shape[0], 3, 32, 32)).astype(np.uint8)
        labels = [int(i) for i in test[b'labels']]
        filenames = [i for i in test[b'filenames']]

        np.savez_compressed(os.path.join(target_dir, 'cifar10_test_data.npz'), data=data, labels=labels,
                            filenames=filenames)

        # clean up
        shutil.rmtree(os.path.join(target_dir, 'cifar-10-batches-py'))
        os.remove(os.path.join(target_dir, 'cifar-10-python.tar.gz'))


class Dataset(object):
    def __init__(self):
        __metaclass__ = ABCMeta

    @abstractmethod
    def __getitem__(self, item):
        '''
        needs to return a data_dict for the sample at the position item
        :param item:
        :return:
        '''
        pass

    @abstractmethod
    def __len__(self):
        '''
        returns how many items the dataset has
        :return:
        '''
        pass


class Cifar10Dataset(Dataset):
    def __init__(self, dataset_directory, train=True, transform=None):
        super().__init__()
        self.transform = transform
        maybe_download_and_prepare_cifar10(dataset_directory)

        self.train = train

        # load appropriate data
        if train:
            fname = os.path.join(dataset_directory, 'cifar10_training_data.npz')
        else:
            fname = os.path.join(dataset_directory, 'cifar10_test_data.npz')

        dataset = np.load(fname)

        self.data = dataset['data']
        self.labels = dataset['labels']
        self.filenames = dataset['filenames']

    def __getitem__(self, item):
        data_dict = {'data': self.data[item:item+1], 'labels': self.labels[item], 'filenames': self.filenames[item]}
        if self.transform is not None:
            data_dict = self.transform(**data_dict)
        return data_dict

    def __len__(self):
        return len(self.data)


class DataLoader(SlimDataLoaderBase):
    def __init__(self, data, batch_size, num_threads_in_multithreaded, seed_for_shuffle=1):
        super().__init__(data, batch_size, num_threads_in_multithreaded)
        self.rs = np.random.RandomState(seed_for_shuffle)
        self.current_position = None
        self.was_initialized = False

        # when you derive, make sure to set this! We can't set it here because we don't know what data will be like
        self.indices = None

    def reset(self):
        assert self.indices is not None

        global _use_shared_memory  # this is default_collate specific stuff
        _use_shared_memory = True

        self.current_position = self.thread_id
        self.was_initialized = True
        self.rs.seed(self.rs.randint(0, 999999999))
        self.rs.shuffle(self.indices)

    def get_indices(self):
        if not self.was_initialized:
            self.reset()

        indices = []

        for b in range(self.batch_size):
            if self.current_position < len(self.indices):
                indices.append(self.current_position)

                self.current_position += self.number_of_threads_in_multithreaded
            else:
                self.reset()
                raise StopIteration
        return indices

    @abstractmethod
    def generate_train_batch(self):
        '''
        make use of self.get_indices() to know what indices to work on!
        :return:
        '''
        pass


class HighPerformanceCIFAR10Loader(DataLoader):
    def __init__(self, data, batch_size, num_threads_in_multithreaded, seed_for_shuffle=1):
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle)
        self.indices = np.arange(len(data[0]))

    def generate_train_batch(self):
        indices = self.get_indices()

        data = self._data[0][indices]
        labels = self._data[1][indices]
        filenames = self._data[2][indices]

        return {'data': data, 'labels': labels, 'filenames': filenames}


"""def default_collate(batch):
    '''
    heavily inspired by the default_collate function of pytorch
    :param batch:
    :return:
    '''
    if isinstance(batch[0], np.ndarray):
        return np.vstack(batch)
    elif isinstance(batch[0], (int, np.int64)):
        return np.array(batch).astype(np.int32)
    elif isinstance(batch[0], (float, np.float32)):
        return np.array(batch).astype(np.float32)
    elif isinstance(batch[0], (np.float64,)):
        return np.array(batch).astype(np.float64)
    elif isinstance(batch[0], (dict, OrderedDict)):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    elif isinstance(batch[0], str):
        return batch
    else:
        raise TypeError('unknown type for batch:', type(batch))"""


class DataLoaderFromDataset(DataLoader):
    def __init__(self, data, batch_size, num_threads_in_multithreaded, seed_for_shuffle=1):
        '''
        A simple dataloader that can take a Dataset as data.
        It is not super efficient because I cannot make too many hard assumptions about what data_dict will contain.
        If you know what you need, implement your own!
        :param data:
        :param batch_size:
        :param num_threads_in_multithreaded:
        :param seed_for_shuffle:
        '''
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle)
        assert isinstance(self._data, Dataset)
        self.indices = np.arange(len(data))

    def generate_train_batch(self):
        indices = self.get_indices()

        batch = [self._data[i] for i in indices]

        return default_collate(batch)


if __name__ == '__main__':
    from time import time
    batch_size = 50
    num_workers = 8
    pin_memory = True
    num_epochs = 3
    dataset_dir = '/media/fabian/data/data/cifar10'
    numpy_to_tensor = NumpyToTensor(['data', 'labels'], cast_to=None)
    fname = os.path.join(dataset_dir, 'cifar10_training_data.npz')
    dataset = np.load(fname)
    cifar_dataset_as_arrays = (dataset['data'], dataset['labels'], dataset['filenames'])
    print('batch_size', batch_size)
    print('num_workers', num_workers)
    print('pin_memory', pin_memory)
    print('num_epochs', num_epochs)

    tr_transforms = [SpatialTransform((32, 32))] * 1  # SpatialTransform is computationally expensive and we need some
    # load on CPU so we just stack 5 of them on top of each other
    tr_transforms.append(numpy_to_tensor)
    tr_transforms = Compose(tr_transforms)

    cifar_dataset = Cifar10Dataset(dataset_dir, train=True, transform=tr_transforms)

    dl = DataLoaderFromDataset(cifar_dataset, batch_size, num_workers, 1)
    mt = MultiThreadedAugmenter(dl, None, num_workers, 1, None, pin_memory)

    batches = 0
    for _ in mt:
        batches += 1
    assert len(_['data'].shape) == 4

    assert batches == len(cifar_dataset) / batch_size  # this assertion only holds if len(datset) is divisible by
    # batch size

    start = time()
    for _ in range(num_epochs):
        batches = 0
        for _ in mt:
            batches += 1
    stop = time()
    print('batchgenerators took %03.4f seconds' % (stop - start))

    # The best I can do:

    dl = HighPerformanceCIFAR10Loader(cifar_dataset_as_arrays, batch_size, num_workers, 1) # this circumvents the
    # default_collate function, just to see if that is slowing things down
    mt = MultiThreadedAugmenter(dl, tr_transforms, num_workers, 1, None, pin_memory)

    batches = 0
    for _ in mt:
        batches += 1
    assert len(_['data'].shape) == 4

    assert batches == len(cifar_dataset_as_arrays[0]) / batch_size  # this assertion only holds if len(datset) is
    # divisible by batch size

    start = time()
    for _ in range(num_epochs):
        batches = 0
        for _ in mt:
            batches += 1
    stop = time()
    print('high performance batchgenerators %03.4f seconds' % (stop - start))


    from torch.utils.data import DataLoader as TorchDataLoader

    trainloader = TorchDataLoader(cifar_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory, collate_fn=default_collate)

    batches = 0
    for _ in iter(trainloader):
        batches += 1
    assert len(_['data'].shape) == 4

    start = time()
    for _ in range(num_epochs):
        batches = 0
        for _ in trainloader:
            batches += 1
    stop = time()
    print('pytorch took %03.4f seconds' % (stop - start))
