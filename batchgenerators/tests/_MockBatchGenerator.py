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


import numpy as np
from batchgenerators.dataloading import DataLoaderBase


class MockBatchGenerator(DataLoaderBase):

    def generate_train_batch(self):

        #Sample randomly from data
        idx = np.random.choice(self._data[0].shape[0], self.BATCH_SIZE, False, None)
        # copy data to ensure that we are not modifying the original dataset with subsequeng augmentation techniques!
        x = np.array(self._data[0][idx])
        y = np.array(self._data[1][idx])

        data_dict = {"data": x,
                     "seg": y}
        return data_dict


class MockRepeatBatchGenerator(DataLoaderBase):
    def generate_train_batch(self):

        # copy data to ensure that we are not modifying the original dataset with subsequeng augmentation techniques!
        x = np.repeat(self._data[0], repeats=self.BATCH_SIZE, axis=0)
        y = np.repeat(self._data[1], repeats=self.BATCH_SIZE, axis=0)

        data_dict = {"data": x,
                     "seg": y}
        return data_dict


class DummyGenerator(DataLoaderBase):
    def __init__(self, dataset_size, batch_size, fill_data='random', fill_seg='ones'):
        if fill_data == "random":
            data = np.random.random(dataset_size)
        else:
            raise NotImplementedError

        if fill_seg == "ones":
            seg = np.ones(dataset_size)
        else:
            raise NotImplementedError

        super(DummyGenerator, self).__init__((data, seg), batch_size, None, False)

    def generate_train_batch(self):
        idx = np.random.choice(self._data[0].shape[0])

        data = self._data[0][idx]
        seg = self._data[1][idx]
        return {'data': data, 'seg': seg}
