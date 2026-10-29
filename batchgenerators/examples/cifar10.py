import argparse
import numpy as np
import os
from batchgenerators.dataloading.data_loader import DataLoaderFromDataset
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.datasets.cifar import HighPerformanceCIFARLoader, CifarDataset, maybe_download_and_prepare_cifar
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from torch.utils.data import default_collate


def collate(batch):
    """
    CifarDataset returns each sample with a batch dimension of 1 (SpatialTransform needs one), so default_collate
    stacks the images to (b, 1, c, x, y). Used by both DataLoaders so that they do the same work.
    """
    batch = default_collate(batch)
    batch['data'] = batch['data'][:, 0]
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarks batchgenerators against the pytorch DataLoader on CIFAR-10')
    parser.add_argument('dataset_dir', help='folder for CIFAR-10, it is downloaded there if not present')
    args = parser.parse_args()

    from time import time
    batch_size = 50
    num_workers = 8
    pin_memory = False
    num_epochs = 3
    dataset_dir = args.dataset_dir
    maybe_download_and_prepare_cifar(dataset_dir)
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

    cifar_dataset = CifarDataset(dataset_dir, train=True, transform=tr_transforms)

    dl = DataLoaderFromDataset(cifar_dataset, batch_size, num_workers, 1, collate_fn=collate)
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

    dl = HighPerformanceCIFARLoader(cifar_dataset_as_arrays, batch_size, num_workers, 1) # this circumvents the
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
                                  pin_memory=pin_memory, collate_fn=collate)

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
