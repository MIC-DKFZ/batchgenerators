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
from batchgenerators.dataloading import SlimDataLoaderBase


class BasicDataLoader(SlimDataLoaderBase):
    """
    data is a tuple of images (b,c,x,y(,z)) and segmentations (b,c,x,y(,z))
    """

    def generate_train_batch(self):
        #Sample randomly from data
        idx = np.random.choice(self._data[0].shape[0], self.batch_size, True, None)
        # copy data to ensure that we are not modifying the original dataset with subsequeng augmentation techniques!
        x = np.array(self._data[0][idx])
        y = np.array(self._data[1][idx])
        data_dict = {"data": x,
                     "seg": y}
        return data_dict


class DummyGenerator(SlimDataLoaderBase):
    """
    creates random data and seg of shape dataset_size and returns those.
    """
    def __init__(self, dataset_size, batch_size, fill_data='random', fill_seg='ones'):
        if fill_data == "random":
            data = np.random.random(dataset_size)
        elif fill_data == "ones":
            data = np.ones(dataset_size)
        else:
            raise NotImplementedError

        if fill_seg == "ones":
            seg = np.ones(dataset_size)
        else:
            raise NotImplementedError

        super(DummyGenerator, self).__init__((data, seg), batch_size, None)

    def generate_train_batch(self):
        idx = np.random.choice(self._data[0].shape[0], self.batch_size)

        data = self._data[0][idx]
        seg = self._data[1][idx]
        return {'data': data, 'seg': seg}


class OneDotDataLoader(SlimDataLoaderBase):
    def __init__(self, dataset_size, batch_size, coord_of_voxel):
        """
        creates both data and seg with only one voxel being = 1 and the rest zero. This will allow easy tracking of
        spatial transformations
        :param data_size: (b,c,x,y(,z))
        :param coord_of_voxel: (x, y(, z)))
        """
        super(OneDotDataLoader, self).__init__(None, batch_size, None)

        self.data = np.zeros(dataset_size)
        self.seg = np.zeros(dataset_size)
        self.data[:, :][coord_of_voxel] = 1
        self.seg[:, :][coord_of_voxel] = 1

    def generate_train_batch(self):
        idx = np.random.choice(self.data.shape[0], self.batch_size)

        data = self.data[idx]
        seg = self.data[idx]
        return {'data': data, 'seg': seg}
