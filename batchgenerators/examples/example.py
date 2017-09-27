from __future__ import print_function

from builtins import range

import numpy as np

from batchgenerators.generators.channel_selection_generators import data_channel_selection_generator
from batchgenerators.generators.crop_and_pad_generators import pad_generator, center_crop_generator, \
    random_crop_generator
from batchgenerators.generators.data_generator_base import BatchGeneratorBase
from batchgenerators.generators.multi_threaded_generator import MultiThreadedGenerator
from batchgenerators.generators.spatial_transform_generators import mirror_axis_generator


class MyEasyBatchGenerator(BatchGeneratorBase):
    def generate_train_batch(self):
        # img_data is here (data, target) (see __main__)
        idx = np.random.choice(self._data[0].shape[0], self.BATCH_SIZE, False, None)
        # copy data to ensure that we are not modifying the original dataset with subsequeng augmentation techniques!
        x = np.array(self._data[0][idx])
        y = np.array(self._data[1][idx])
        data_dict = {"data": x,
                     "target": y}
        return data_dict


if __name__ == "__main__":
    # easy example (clasification, no data augmentation):
    # generate some random data
    data = np.random.random((10, 3, 100, 100)).astype(np.float32)
    target = np.random.choice([0, 1], 100, True).astype(np.int32)
    data_slice_0 = np.array(np.meshgrid(list(range(100)), list(range(100)))).sum(0)
    data_slice_1 = data_slice_0.transpose()
    data_slice_2 = data_slice_0[::-1, ::-1] - 50
    for i in range(data.shape[0]):
        data[i, 0] = data_slice_0
        data[i, 1] = data_slice_1
        data[i, 2] = data_slice_2

    # declare batch generator
    batch_gen = MyEasyBatchGenerator((data, target), BATCH_SIZE=10, num_batches=100, seed=None)
    ctr = 0
    for data_dict in batch_gen:
        print("got batch ", ctr)
        ctr += 1

    # more complex example (classification, data augmentation):
    # generate some random data

    # declare batch generator
    num_threads = 4
    number_of_batches = 100
    # when going multi-threaded we need to adapt the number of batches for the generator because it will be cloned num_threads times
    batch_gen = MyEasyBatchGenerator((data, target), BATCH_SIZE=10, num_batches=number_of_batches / num_threads,
                                     seed=None)
    # now stack data augmentation
    batch_gen = pad_generator(batch_gen, (150, 150), 0)  # pads x form size (100, 100) to (150, 150)
    batch_gen = data_channel_selection_generator(batch_gen, [1, 2])  # selects channel 1 and 2 of img (discards 0)
    batch_gen = rotation_generator(batch_gen)  # rotate around center
    batch_gen = center_crop_generator(batch_gen, (
    125, 125))  # crop borders (do this after rotation so that there are no black borders)
    batch_gen = mirror_axis_generator(batch_gen)  # just some mirroring
    batch_gen = elastric_transform_generator(batch_gen, 100,
                                             10)  # do some elastic deformations (the parameters need to be tuned carefully), beware of image border artifacts
    batch_gen = random_crop_generator(batch_gen, (100, 100))  # select a random patch from x
    batch_gen = MultiThreadedGenerator(batch_gen, 4, 16)

    ctr = 0
    for data_dict in batch_gen:
        print("got batch ", ctr)
        ctr += 1
        # simple, right?
