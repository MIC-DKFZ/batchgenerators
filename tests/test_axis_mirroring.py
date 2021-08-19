# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
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
import unittest2
import numpy as np
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from skimage import data

from tests.DataGenerators import BasicDataLoader
from batchgenerators.transforms.spatial_transforms import MirrorTransform


class TestMirrorAxis(unittest2.TestCase):
    def setUp(self):
        self.seed = 1234

        self.batch_size = 10
        self.num_batches = 1000

        np.random.seed(self.seed)

        ### 2D initialiazations

        cam = data.camera()
        self.cam = cam[np.newaxis, np.newaxis, :, :]

        self.cam_left = self.cam[:, :, :, ::-1]
        self.cam_updown = self.cam[:, :, ::-1, :]
        self.cam_updown_left = self.cam[:, :, ::-1, ::-1]

        self.x_2D = self.cam
        self.y_2D = self.cam

        ### 3D initialiazations

        self.cam_3D = np.random.rand(20, 20, 20)[np.newaxis, np.newaxis, :, :, :]

        self.cam_3D_left = self.cam_3D[:, :, :, ::-1, :]
        self.cam_3D_updown = self.cam_3D[:, :, ::-1, :, :]
        self.cam_3D_updown_left = self.cam_3D[:, :, ::-1, ::-1, :]

        self.cam_3D_left_z = self.cam_3D_left[:, :, :, :, ::-1]
        self.cam_3D_updown_z = self.cam_3D_updown[:, :, :, :, ::-1]
        self.cam_3D_updown_left_z = self.cam_3D_updown_left[:, :, :, :, ::-1]
        self.cam_3D_z = self.cam_3D[:, :, :, :, ::-1]

        self.x_3D = self.cam_3D
        self.y_3D = self.cam_3D


    def test_random_distributions_2D(self):
        ### test whether all 4 possible mirrorings occur in approximately equal frquencies in 2D

        batch_gen = BasicDataLoader((self.x_2D, self.y_2D), self.batch_size, number_of_threads_in_multithreaded=None)
        batch_gen = SingleThreadedAugmenter(batch_gen, MirrorTransform((0, 1)))

        counts = np.zeros(shape=(4,))

        for b in range(self.num_batches):
            batch = next(batch_gen)

            for ix in range(self.batch_size):
                if (batch['data'][ix, :, :, :] == self.cam_left).all():
                    counts[0] = counts[0] + 1

                elif (batch['data'][ix, :, :, :] == self.cam_updown).all():
                    counts[1] = counts[1] + 1

                elif (batch['data'][ix, :, :, :] == self.cam_updown_left).all():
                    counts[2] = counts[2] + 1

                elif (batch['data'][ix, :, :, :] == self.cam).all():
                    counts[3] = counts[3] + 1

        self.assertTrue([1 if (2200 < c < 2800) else 0 for c in counts] == [1]*4, "2D Images were not mirrored along "
                                                                                  "all axes with equal probability. "
                                                                                  "This may also indicate that "
                                                                                  "mirroring is not working")


    def test_segmentations_2D(self):
        ### test whether segmentations are mirrored coherently with images

        batch_gen = BasicDataLoader((self.x_2D, self.y_2D), self.batch_size, number_of_threads_in_multithreaded=None)
        batch_gen = SingleThreadedAugmenter(batch_gen, MirrorTransform((0, 1)))

        equivalent = True

        for b in range(self.num_batches):
            batch = next(batch_gen)
            for ix in range(self.batch_size):
                if (batch['data'][ix] != batch['seg'][ix]).all():
                    equivalent = False

        self.assertTrue(equivalent, "2D images and seg were not mirrored in the same way (they should though because "
                                    "seg needs to match the corresponding data")


    def test_random_distributions_3D(self):
        ### test whether all 8 possible mirrorings occur in approximately equal frquencies in 3D case

        batch_gen = BasicDataLoader((self.x_3D, self.y_3D), self.batch_size, number_of_threads_in_multithreaded=None)
        batch_gen = SingleThreadedAugmenter(batch_gen, MirrorTransform((0, 1, 2)))

        counts = np.zeros(shape=(8,))

        for b in range(self.num_batches):
            batch = next(batch_gen)
            for ix in range(self.batch_size):
                if (batch['data'][ix, :, :, :, :] == self.cam_3D_left).all():
                    counts[0] = counts[0] + 1

                elif (batch['data'][ix, :, :, :, :] == self.cam_3D_updown).all():
                    counts[1] = counts[1] + 1

                elif (batch['data'][ix, :, :, :, :] == self.cam_3D_updown_left).all():
                    counts[2] = counts[2] + 1

                elif (batch['data'][ix, :, :, :, :] == self.cam_3D).all():
                    counts[3] = counts[3] + 1

                elif (batch['data'][ix, :, :, :, :] == self.cam_3D_left_z).all():
                    counts[4] = counts[1] + 1

                elif (batch['data'][ix, :, :, :, :] == self.cam_3D_updown_z).all():
                    counts[5] = counts[1] + 1

                elif (batch['data'][ix, :, :, :, :] == self.cam_3D_updown_left_z).all():
                    counts[6] = counts[2] + 1

                elif (batch['data'][ix, :, :, :, :] == self.cam_3D_z).all():
                    counts[7] = counts[3] + 1

        self.assertTrue([1 if (1000 < c < 1400) else 0 for c in counts] == [1]*8, "3D Images were not mirrored along "
                                                                                  "all axes with equal probability. "
                                                                                  "This may also indicate that "
                                                                                  "mirroring is not working")


    def test_segmentations_3D(self):
        ### test whether segmentations are rotated coherently with images

        batch_gen = BasicDataLoader((self.x_3D, self.y_3D), self.batch_size, number_of_threads_in_multithreaded=None)
        batch_gen = SingleThreadedAugmenter(batch_gen, MirrorTransform((0, 1, 2)))

        equivalent = True

        for b in range(self.num_batches):
            batch = next(batch_gen)
            for ix in range(self.batch_size):
                if (batch['data'][ix] != batch['seg'][ix]).all():
                    equivalent = False

        self.assertTrue(equivalent, "3D images and seg were not mirrored in the same way (they should though because "
                                    "seg needs to match the corresponding data")


if __name__ == '__main__':
    unittest.main()

