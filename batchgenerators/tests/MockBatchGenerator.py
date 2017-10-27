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


from DeepLearningBatchGeneratorUtils.DataGeneratorBase import BatchGeneratorBase
import numpy as np

class MockBatchGenerator(BatchGeneratorBase):

    def generate_train_batch(self):

        #Sample randomly from data
        idx = np.random.choice(self._data[0].shape[0], self.BATCH_SIZE, False, None)
        # copy data to ensure that we are not modifying the original dataset with subsequeng augmentation techniques!
        x = np.array(self._data[0][idx])
        y = np.array(self._data[1][idx])

        data_dict = {"data": x,
                     "seg": y}
        return data_dict


class MockRepeatBatchGenerator(BatchGeneratorBase):
    def generate_train_batch(self):

        # copy data to ensure that we are not modifying the original dataset with subsequeng augmentation techniques!
        x = np.repeat(self._data[0], repeats=self.BATCH_SIZE, axis=0)
        y = np.repeat(self._data[1], repeats=self.BATCH_SIZE, axis=0)

        data_dict = {"data": x,
                     "seg": y}
        return data_dict


