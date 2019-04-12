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


import unittest
from time import sleep

import numpy as np
from batchgenerators.dataloading import SlimDataLoaderBase, MultiThreadedAugmenter
from batchgenerators.examples.multithreaded_dataloading import DummyDL, DummyDLWithShuffle
from skimage.data import camera, checkerboard, astronaut, binary_blobs, coins
from skimage.transform import resize
from copy import deepcopy


class DummyDL2DImage(SlimDataLoaderBase):
    def __init__(self, batch_size, num_threads=8):
        data = []
        target_shape = (224, 224)

        c = camera()
        c = resize(c.astype(np.float64), target_shape, 1, anti_aliasing=False, clip=True, mode='reflect').astype(np.float32)
        data.append(c[None])

        c = checkerboard()
        c = resize(c.astype(np.float64), target_shape, 1, anti_aliasing=False, clip=True, mode='reflect').astype(np.float32)
        data.append(c[None])

        c = astronaut().mean(-1)
        c = resize(c.astype(np.float64), target_shape, 1, anti_aliasing=False, clip=True, mode='reflect').astype(np.float32)
        data.append(c[None])

        c = binary_blobs()
        c = resize(c.astype(np.float64), target_shape, 1, anti_aliasing=False, clip=True, mode='reflect').astype(np.float32)
        data.append(c[None])

        c = coins()
        c = resize(c.astype(np.float64), target_shape, 1, anti_aliasing=False, clip=True, mode='reflect').astype(np.float32)
        data.append(c[None])
        data = np.stack(data)
        super(DummyDL2DImage, self).__init__(data, batch_size, num_threads)

    def generate_train_batch(self):
        idx = np.random.choice(len(self._data), self.batch_size)
        res = []
        for i in idx:
            res.append(self._data[i:i+1])
        res = np.vstack(res)
        return {'data': res}


class TestMultiThreadedAugmenter(unittest.TestCase):
    """
    This test is inspired by the multithreaded example I did a while back
    """
    def setUp(self):
        np.random.seed(1234)
        self.num_threads = 4
        self.dl = DummyDL(self.num_threads)
        self.dl_with_shuffle = DummyDLWithShuffle(self.num_threads)
        self.dl_images = DummyDL2DImage(4, self.num_threads)

    def test_no_crash(self):
        """
        This one should just not crash, that's all
        :return:
        """
        dl = self.dl_images
        mt_dl = MultiThreadedAugmenter(dl, None, self.num_threads, 1, None, False)

        for _ in range(20):
            _ = mt_dl.next()

    def test_DummyDL(self):
        """
        DummyDL must return numbers from 0 to 99 in ascending order
        :return:
        """
        dl = DummyDL(1)
        res = []
        for i in dl:
            res.append(i)

        assert len(res) == 100
        res_copy = deepcopy(res)
        res.sort()
        assert all((i == j for i, j in zip(res, res_copy)))
        assert all((i == j for i, j in zip(res, np.arange(0, 100))))

    def test_order(self):
        """
        Coordinating workers in a multiprocessing envrionment is difficult. We want DummyDL in a multithreaded
        environment to still give us the numbers from 0 to 99 in ascending order
        :return:
        """
        dl = self.dl
        mt = MultiThreadedAugmenter(dl, None, self.num_threads, 1, None, False)

        res = []
        for i in mt:
            res.append(i)

        assert len(res) == 100
        res_copy = deepcopy(res)
        res.sort()
        assert all((i == j for i, j in zip(res, res_copy)))
        assert all((i == j for i, j in zip(res, np.arange(0, 100))))

    def test_restart_and_order(self):
        """
        Coordinating workers in a multiprocessing envrionment is difficult. We want DummyDL in a multithreaded
        environment to still give us the numbers from 0 to 99 in ascending order.

        We want the MultiThreadedAugmenter to restart and return the same result in each run
        :return:
        """
        dl = self.dl
        mt = MultiThreadedAugmenter(dl, None, self.num_threads, 1, None, False)

        res = []
        for i in mt:
            res.append(i)

        assert len(res) == 100
        res_copy = deepcopy(res)
        res.sort()
        assert all((i == j for i, j in zip(res, res_copy)))
        assert all((i == j for i, j in zip(res, np.arange(0, 100))))

        res = []
        for i in mt:
            res.append(i)

        assert len(res) == 100
        res_copy = deepcopy(res)
        res.sort()
        assert all((i == j for i, j in zip(res, res_copy)))
        assert all((i == j for i, j in zip(res, np.arange(0, 100))))

        res = []
        for i in mt:
            res.append(i)

        assert len(res) == 100
        res_copy = deepcopy(res)
        res.sort()
        assert all((i == j for i, j in zip(res, res_copy)))
        assert all((i == j for i, j in zip(res, np.arange(0, 100))))

    def test_image_pipeline_and_pin_memory(self):
        '''
        This just should not crash
        :return:
        '''
        try:
            import torch
        except ImportError:
            '''dont test if torch is not installed'''
            return

        from batchgenerators.transforms import MirrorTransform, NumpyToTensor, TransposeAxesTransform, Compose

        tr_transforms = []
        tr_transforms.append(MirrorTransform())
        tr_transforms.append(TransposeAxesTransform(transpose_any_of_these=(0, 1), p_per_sample=0.5))
        tr_transforms.append(NumpyToTensor(keys='data', cast_to='float'))

        composed = Compose(tr_transforms)

        dl = self.dl_images
        mt = MultiThreadedAugmenter(dl, composed, 4, 1, None, True)

        for _ in range(50):
            res = mt.next()

        assert isinstance(res['data'], torch.Tensor)
        assert res['data'].is_pinned()

        # let mt finish caching, otherwise it's going to print an error (which is not a problem and will not prevent
        # the success of the test but it does not look pretty)
        sleep(2)

    def test_image_pipeline(self):
        '''
        This just should not crash
        :return:
        '''
        from batchgenerators.transforms import MirrorTransform, TransposeAxesTransform, Compose

        tr_transforms = []
        tr_transforms.append(MirrorTransform())
        tr_transforms.append(TransposeAxesTransform(transpose_any_of_these=(0, 1), p_per_sample=0.5))

        composed = Compose(tr_transforms)

        dl = self.dl_images
        mt = MultiThreadedAugmenter(dl, composed, 4, 1, None, False)

        for _ in range(50):
            res = mt.next()

        # let mt finish caching, otherwise it's going to print an error (which is not a problem and will not prevent
        # the success of the test but it does not look pretty)
        sleep(2)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    unittest.main()
