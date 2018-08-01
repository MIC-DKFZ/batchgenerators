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


from builtins import object, range

import numpy as np
from copy import deepcopy

from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, interpolate_img, rotate_coords_2d, \
    rotate_coords_3d, uncenter_coords


class InvertibleRotationGenerator(object):
    """
    Enables rotatation of batch data received from a generator object and the inverse roatation
    Use-case: Use rotations in the prediction step and rotate back the results
    Use like:
    batch_gen = SomeBatchGenerator(SomeData)
    inv_rot_batch_gen = InvertibleRotationGenerator(batch_gen)
    batch_gen = inv_rot_batch_gen.generate()
    batch, rotated_batch = next(batch_gen)
    inverse_rotated_batch = inv_rot_batch_gen.invert(rotated_batch)
    """

    def __init__(self, generator, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, seed=42):

        np.random.seed(seed)
        self.generator = generator
        self.params = {'ax': angle_x, 'ay': angle_y, 'az': angle_z,
                       'bmode_data': border_mode_data, 'bmode_seg': border_mode_seg,
                       'bcval_data': border_cval_data, 'bcval_seg': border_cval_seg,
                       'order_data': order_data, 'order_seg': order_seg}

        self.rand_params = {}

    def rotate(self, data_dict):
        data = data_dict["data"]
        do_seg = False
        seg = None
        if "seg" in list(data_dict.keys()):
            seg = data_dict["seg"]
            do_seg = True
        shape = np.array(data.shape[2:])
        dim = len(shape)
        for sample_id in range(data.shape[0]):
            coords = create_zero_centered_coordinate_mesh(shape)

            if dim == 3:
                coords = rotate_coords_3d(coords,
                                          self.rand_params['ax'][sample_id],
                                          self.rand_params['ay'][sample_id],
                                          self.rand_params['az'][sample_id])
            else:
                coords = rotate_coords_2d(coords, self.rand_params['ax'][sample_id])
            coords = uncenter_coords(coords)
            for channel_id in range(data.shape[1]):
                data[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords,
                                                              self.params['order_data'], self.params['bmode_data'],
                                                              cval=self.params['bcval_data'])
            if do_seg:
                for channel_id in range(seg.shape[1]):
                    seg[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords,
                                                                 self.params['order_seg'], self.params['bmode_seg'],
                                                                 cval=self.params['bcval_seg'], is_seg=True)

        return {'data': data, 'seg': seg}

    def generate(self):

        for data_dict in self.generator:
            assert "data" in list(
                data_dict.keys()), "your data generator needs to return a python dictionary with at least a 'data' key value pair"

            self.shape = np.array(data_dict["data"].shape[2:])
            self.dim = len(self.shape)

            self.rand_params['ax'] = np.random.uniform(self.params['ax'][0], self.params['ax'][1], size=self.shape[0])
            if self.dim == 3:
                self.rand_params['ay'] = np.random.uniform(self.params['ay'][0], self.params['ay'][1],
                                                           size=self.shape[0])
                self.rand_params['az'] = np.random.uniform(self.params['az'][0], self.params['az'][1],
                                                           size=self.shape[0])

            initial_data_dict = deepcopy(data_dict)
            rotated_data_dict = self.rotate(data_dict)
            yield initial_data_dict, rotated_data_dict

    def invert(self, data_dict):

        rotated_data_dict = deepcopy(data_dict)
        self.rand_params['ax'] = -self.rand_params['ax']
        if self.dim == 3:
            self.rand_params['ay'] = -self.rand_params['ay']
            self.rand_params['az'] = -self.rand_params['az']

        return self.rotate(rotated_data_dict)
