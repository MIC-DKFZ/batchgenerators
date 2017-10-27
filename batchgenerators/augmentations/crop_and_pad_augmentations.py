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

from builtins import range

import numpy as np


def center_crop(data, output_size, seg=None):
    seg_return = None
    if type(output_size) not in (tuple, list):
        center_crop = [int(output_size)] * (len(data.shape) - 2)
    else:
        center_crop = output_size
        assert len(center_crop) == len(
            data.shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your data (2d/3d)"
    center = np.array(data.shape[2:]) / 2
    if len(data.shape) == 5:
        data_return = data[:, :, int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
                      int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.),
                      int(center[2] - center_crop[2] / 2.):int(center[2] + center_crop[2] / 2.)]
        if seg is not None:
            seg_return = seg[:, :, int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
                         int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.),
                         int(center[2] - center_crop[2] / 2.):int(center[2] + center_crop[2] / 2.)]
    elif len(data.shape) == 4:
        data_return = data[:, :, int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
                      int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]
        if seg is not None:
            seg_return = seg[:, :, int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
                         int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]
    else:
        raise Exception(
            "Invalid dimension for seg. seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
    return data_return, seg_return


def center_crop_seg(seg, output_size):
    if type(output_size) not in (tuple, list):
        center_crop = [int(output_size)] * (len(seg.shape) - 2)
    else:
        center_crop = output_size
        assert len(center_crop) == len(
            seg.shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your data (2d/3d)"
    center = np.array(seg.shape[2:]) / 2
    if len(seg.shape) == 4:
        seg_return = seg[:, :, int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
                     int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]
    elif len(seg.shape) == 5:
        seg_return = seg[:, :, int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
                     int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.),
                     int(center[2] - center_crop[2] / 2.):int(center[2] + center_crop[2] / 2.)]
    else:
        raise Exception(
            "Invalid dimension for seg. seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")

    return seg_return


def random_crop(data, seg=None, crop_size=128, margins=[0, 0, 0]):
    seg_return = None
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(data.shape) - 2)
    else:
        assert len(crop_size) == len(
            data.shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your data (2d/3d)"

    if crop_size[0] < data.shape[2]:
        lb_x = np.random.randint(margins[0], data.shape[2] - crop_size[0] - margins[0])
    elif crop_size[0] == data.shape[2]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < data.shape[3]:
        lb_y = np.random.randint(margins[1], data.shape[3] - crop_size[1] - margins[1])
    elif crop_size[1] == data.shape[3]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    if len(data.shape) == 4:
        data_return = data[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]
        if seg is not None:
            seg_return = seg[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]
    elif len(data.shape) == 5:
        if crop_size[2] < data.shape[4]:
            lb_z = np.random.randint(margins[2], data.shape[4] - crop_size[2] - margins[2])
        elif crop_size[2] == data.shape[4]:
            lb_z = 0
        else:
            raise ValueError("crop_size[2] must be smaller or equal to the images z dimension")
        data_return = data[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]
        if seg is not None:
            seg_return = seg[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]
    else:
        raise ValueError("Invalid data/seg dimension")
    return data_return, seg_return


def pad(data, new_size, seg=None, pad_value_data=None, pad_value_seg=None):
    shape = tuple(list(data.shape)[2:])
    start = np.array(new_size) / 2. - np.array(shape) / 2.
    res_data = np.ones([data.shape[0], data.shape[1]] + list(new_size), dtype=data.dtype)
    res_seg = None
    if seg is not None:
        res_seg = np.ones([seg.shape[0], seg.shape[1]] + list(new_size), dtype=seg.dtype) * pad_value_seg
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if pad_value_data is None:
                if len(shape) == 2:
                    pad_value_tmp = data[i, j, 0, 0]
                elif len(shape) == 3:
                    pad_value_tmp = data[i, j, 0, 0, 0]
                else:
                    raise Exception(
                        "Invalid dimension for data and seg. data and seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
            else:
                pad_value_tmp = pad_value_data
            res_data[i, j] = pad_value_tmp
            if len(shape) == 2:
                res_data[i, j, int(start[0]):int(start[0]) + int(shape[0]),
                int(start[1]):int(start[1]) + int(shape[1])] = data[i, j]
            elif len(shape) == 3:
                res_data[i, j, int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
                int(start[2]):int(start[2]) + int(shape[2])] = data[i, j]
            else:
                raise Exception(
                    "Invalid dimension for data and seg. data and seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
            if seg is not None:
                for j in range(seg.shape[1]):
                    if pad_value_seg is None:
                        if len(shape) == 2:
                            pad_value_tmp = seg[i, j, 0, 0]
                        elif len(shape) == 3:
                            pad_value_tmp = seg[i, j, 0, 0, 0]
                        else:
                            raise Exception(
                                "Invalid dimension for data and seg. data and seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
                    else:
                        pad_value_tmp = pad_value_seg
                    res_seg[i, j] = pad_value_tmp
                    if len(shape) == 2:
                        res_seg[i, j, int(start[0]):int(start[0]) + int(shape[0]),
                        int(start[1]):int(start[1]) + int(shape[1])] = seg[i, j]
                    elif len(shape) == 3:
                        res_seg[i, j, int(start[0]):int(start[0]) + int(shape[0]),
                        int(start[1]):int(start[1]) + int(shape[1]), int(start[2]):int(start[2]) + int(shape[2])] = seg[
                            i, j]
    return res_data, res_seg
