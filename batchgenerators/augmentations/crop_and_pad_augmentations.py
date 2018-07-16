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


def center_crop(data, crop_size, seg=None):
    if isinstance(data, np.ndarray):
        is_list = False
        data_shape = tuple(list(data.shape))
    elif isinstance(data, (list, tuple)):
        is_list = True
        assert len(data) > 0 and isinstance(data[0], np.ndarray)
        data_shape = tuple([len(data)] + list(data[0].shape))
    else:
        raise TypeError("Data has to be either a numpy array or a list")
    if isinstance(seg, np.ndarray):
        seg_shape = tuple(list(seg.shape))
    elif isinstance(seg, (list, tuple)):
        assert len(data) > 0 and isinstance(data[0], np.ndarray)
        seg_shape = tuple([len(seg)] + list(seg[0].shape))
    elif seg is not None:
        raise TypeError("Data has to be either a numpy array or a list")

    seg_return = None
    if type(crop_size) not in (tuple, list, np.ndarray):
        center_crop_size = [int(crop_size)] * (len(data_shape) - 2)
    else:
        center_crop_size = crop_size
        assert len(crop_size) == len(
            data_shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your data (2d/3d)"
    center = np.array(data_shape[2:]) / 2
    if len(data_shape) == 5:
        if not is_list:
            data_return = data[:, :,
                          int(center[0] - center_crop_size[0] / 2.):int(center[0] + center_crop_size[0] / 2.),
                          int(center[1] - center_crop_size[1] / 2.):int(center[1] + center_crop_size[1] / 2.),
                          int(center[2] - center_crop_size[2] / 2.):int(center[2] + center_crop_size[2] / 2.)]
            if seg is not None:
                seg_return = seg[:, :,
                             int(center[0] - center_crop_size[0] / 2.):int(center[0] + center_crop_size[0] / 2.),
                             int(center[1] - center_crop_size[1] / 2.):int(center[1] + center_crop_size[1] / 2.),
                             int(center[2] - center_crop_size[2] / 2.):int(center[2] + center_crop_size[2] / 2.)]
        else:
            data_return = np.zeros([data_shape[0], data_shape[1]] + list(center_crop_size), dtype=data[0].dtype)
            if seg is not None:
                seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(center_crop_size), dtype=data[0].dtype)
            for i, data_smpl in enumerate(data):
                center = np.array(data_smpl.shape[1:]) / 2
                data_return[i,] = data_smpl[:,
                                  int(center[0] - center_crop_size[0] / 2.):int(center[0] + center_crop_size[0] / 2.),
                                  int(center[1] - center_crop_size[1] / 2.):int(center[1] + center_crop_size[1] / 2.),
                                  int(center[2] - center_crop_size[2] / 2.):int(center[2] + center_crop_size[2] / 2.)]
                if seg is not None:
                    seg_return[i,] = seg[i][:,
                                     int(center[0] - center_crop_size[0] / 2.):int(
                                         center[0] + center_crop_size[0] / 2.),
                                     int(center[1] - center_crop_size[1] / 2.):int(
                                         center[1] + center_crop_size[1] / 2.),
                                     int(center[2] - center_crop_size[2] / 2.):int(
                                         center[2] + center_crop_size[2] / 2.)]
    elif len(data_shape) == 4:
        if not is_list:
            data_return = data[:, :,
                          int(center[0] - center_crop_size[0] / 2.):int(center[0] + center_crop_size[0] / 2.),
                          int(center[1] - center_crop_size[1] / 2.):int(center[1] + center_crop_size[1] / 2.)]
            if seg is not None:
                seg_return = seg[:, :,
                             int(center[0] - center_crop_size[0] / 2.):int(center[0] + center_crop_size[0] / 2.),
                             int(center[1] - center_crop_size[1] / 2.):int(center[1] + center_crop_size[1] / 2.)]
        else:
            data_return = np.zeros([data_shape[0], data_shape[1]] + list(center_crop_size), dtype=data[0].dtype)
            if seg is not None:
                seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(center_crop_size), dtype=data[0].dtype)
            for i, data_smpl in enumerate(data):
                center = np.array(data_smpl.shape[1:]) / 2

                data_return[i,] = data_smpl[:,
                                  int(center[0] - center_crop_size[0] / 2.):int(center[0] + center_crop_size[0] / 2.),
                                  int(center[1] - center_crop_size[1] / 2.):int(center[1] + center_crop_size[1] / 2.)]
                if seg is not None:
                    seg_return[i,] = seg[i][:,
                                     int(center[0] - center_crop_size[0] / 2.):int(
                                         center[0] + center_crop_size[0] / 2.),
                                     int(center[1] - center_crop_size[1] / 2.):int(
                                         center[1] + center_crop_size[1] / 2.)]
    else:
        raise Exception(
            "Invalid dimension for seg. seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
    return data_return, seg_return


def center_crop_seg(seg, output_size):
    if type(output_size) not in (tuple, list, np.ndarray):
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


###

def get_rnd_vals(crop_size, data_shape, margins):
    if crop_size[0] < data_shape[2]:
        lb_x = np.random.randint(margins[0], data_shape[2] - crop_size[0] - margins[0])
    elif crop_size[0] == data_shape[2]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < data_shape[3]:
        lb_y = np.random.randint(margins[1], data_shape[3] - crop_size[1] - margins[1])
    elif crop_size[1] == data_shape[3]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    return lb_x, lb_y


def random_crop(data, seg=None, crop_size=128, margins=[0, 0, 0]):
    if isinstance(data, np.ndarray):
        is_list = False
        data_shape = tuple(list(data.shape))
    elif isinstance(data, (list, tuple)):
        is_list = True
        assert len(data) > 0 and isinstance(data[0], np.ndarray)
        data_shape = tuple([len(data)] + list(data[0].shape))
    else:
        raise TypeError("Data has to be either a numpy array or a list")
    if seg is None:
        pass
    elif isinstance(seg, np.ndarray):
        seg_shape = tuple(list(seg.shape))
    elif isinstance(seg, (list, tuple)):
        assert len(data) > 0 and isinstance(data[0], np.ndarray)
        seg_shape = tuple([len(seg)] + list(seg[0].shape))
    else:
        raise TypeError("Data has to be either a numpy array or a list")

    seg_return = None
    if type(crop_size) not in (tuple, list, np.ndarray):
        crop_size = [crop_size] * (len(data_shape) - 2)
    else:
        assert len(crop_size) == len(
            data_shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your data (2d/3d)"

    lb_x, lb_y = get_rnd_vals(crop_size, data_shape, margins)

    if len(data_shape) == 4:
        if not is_list:
            data_return = data[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]
            if seg is not None:
                seg_return = seg[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]
        else:
            data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data[0].dtype)
            if seg is not None:
                seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=data[0].dtype)
            for i, data_smpl in enumerate(data):
                lb_x, lb_y = get_rnd_vals(crop_size, tuple([1] + list(data_smpl.shape)), margins)
                data_return[i,] = data_smpl[:, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]
                if seg is not None:
                    seg_return[i,] = seg[i][:, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]
    elif len(data_shape) == 5:
        if not is_list:
            if crop_size[2] < data_shape[4]:
                lb_z = np.random.randint(margins[2], data_shape[4] - crop_size[2] - margins[2])
            elif crop_size[2] == data_shape[4]:
                lb_z = 0
            else:
                raise ValueError("crop_size[2] must be smaller or equal to the images z dimension")
            data_return = data[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]
            if seg is not None:
                seg_return = seg[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]
        else:
            data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data[0].dtype)
            if seg is not None:
                seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=data[0].dtype)

            for i, data_smpl in enumerate(data):
                if crop_size[2] < data_smpl.shape[3]:
                    lb_z = np.random.randint(margins[2], data_smpl.shape[3] - crop_size[2] - margins[2])
                elif crop_size[2] == data_smpl.shape[3]:
                    lb_z = 0
                else:
                    raise ValueError("crop_size[2] must be smaller or equal to the images z dimension")
                lb_x, lb_y = get_rnd_vals(crop_size, tuple([1] + list(data_smpl.shape)), margins)
                data_return[i,] = data_smpl[:, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1],
                                  lb_z:lb_z + crop_size[2]]
                if seg is not None:
                    seg_return[i,] = seg[i][:, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1],
                                     lb_z:lb_z + crop_size[2]]
    else:
        raise ValueError("Invalid data/seg dimension")
    return data_return, seg_return


def pad_to_multiple(data, multiple, seg=None, pad_value_data=None, pad_value_seg=None):
    if isinstance(data, np.ndarray):
        cur_size = data.shape[2:]
        target_size = [i if i % multiple == 0 else (int(i // multiple) + 1) * multiple for i in cur_size]

        return pad(data=data, new_size=target_size, seg=seg, pad_value_data=pad_value_data, pad_value_seg=pad_value_seg)

    elif isinstance(data, (list, tuple)):
        ret_data = []
        ret_seg = []
        for i, data_smpl in enumerate(data):
            cur_size = data_smpl.shape[1:]
            target_size = [i if i % multiple == 0 else (int(i // multiple) + 1) * multiple for i in cur_size]

            seg_smpl = [seg[i]] if i < len(seg) else None

            res_data, res_seg = pad([data_smpl], target_size, seg_smpl, pad_value_data=pad_value_data,
                                    pad_value_seg=pad_value_seg)

            ret_data.append(res_data)
            ret_seg.append(res_seg)

        return ret_data, ret_seg


def pad_to_ratio_2d(data, ratio, seg=None, pad_value_data=None, pad_value_seg=None):

    assert ratio != 0

    if isinstance(data, np.ndarray):
        cur_size = data.shape[2:]

        size_1 = (int(cur_size[1] * ratio), int(cur_size[1]))
        size_2 = (int(cur_size[0]), int(cur_size[0] * (1. / ratio)))
        target_size = list(cur_size)

        if size_1[0] > size_2[0]:
            target_size[0:2] = size_1[:]
        else:
            target_size[0:2] = size_2[:]

        return pad(data=data, new_size=target_size, seg=seg, pad_value_data=pad_value_data, pad_value_seg=pad_value_seg)

    elif isinstance(data, (list, tuple)):
        ret_data = []
        ret_seg = []
        for i, data_smpl in enumerate(data):
            cur_size = data_smpl.shape[1:]

            size_1 = (int(cur_size[1] * ratio), int(cur_size[1]))
            size_2 = (int(cur_size[0]), int(cur_size[0] * (1. / ratio)))
            target_size = list(cur_size)


            if size_1[0] > size_2[0]:
                target_size[0:2] = size_1[:]
            else:
                target_size[0:2] = size_2[:]

            seg_smpl = [seg[i]] if i < len(seg) else None

            res_data, res_seg = pad([data_smpl], target_size, seg_smpl, pad_value_data=pad_value_data,
                                    pad_value_seg=pad_value_seg)

            ret_data.append(res_data)
            ret_seg.append(res_seg)

        return ret_data, ret_seg



def fillup_pad(data, min_size, seg=None, pad_value_data=None, pad_value_seg=None):
    if isinstance(data, np.ndarray):
        data_shape = tuple(list(data.shape))  #

        if type(min_size) not in (tuple, list):
            min_size = [min_size] * (len(data_shape) - 2)
        else:
            assert len(min_size) == len(
                data_shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your " \
                                 "data (2d/3d)"

        if np.min(np.asarray(data_shape[2:]) - np.asarray(min_size)) < 0:
            return pad(data, min_size, seg, pad_value_data, pad_value_seg)
        else:
            return data, seg
    elif isinstance(data, (list, tuple)):

        if type(min_size) not in (tuple, list, np.ndarray):
            min_size = [min_size] * (len(data[0].shape) - 1)
        else:
            assert len(min_size) == len(
                data[0].shape) - 1, "If you provide a list/tuple as center crop make sure it has the same dimension as " \
                                    "your " \
                                    "data (2d/3d)"

        res_data = []
        res_seg = []
        for i in range(len(data)):
            data_smpl = data[i]
            seg_smpl = None
            if seg is not None:
                seg_smpl = [seg[i]]
            new_shp = np.max(np.vstack((np.array(data_smpl.shape[1:])[None], np.array(min_size)[None])), 0)

            res_d, res_s = pad([data_smpl], new_shp, seg_smpl, pad_value_data=None, pad_value_seg=None)

            res_data.append(res_d[0])
            if seg is not None:
                res_seg.append(res_s[0])
            else:
                res_seg = None
        return res_data, res_seg
    else:
        raise TypeError("Data has to be either a numpy array or a list")


def pad(data, new_size, seg=None, pad_value_data=None, pad_value_seg=None):
    if isinstance(data, np.ndarray):
        data_shape = tuple(list(data.shape))
    elif isinstance(data, (list, tuple)):
        assert len(data) > 0 and isinstance(data[0], np.ndarray)
        data_shape = tuple([len(data)] + list(data[0].shape))
    else:
        raise TypeError("Data has to be either a numpy array or a list")

    if isinstance(seg, np.ndarray):
        seg_shape = tuple(list(seg.shape))
    elif isinstance(seg, (list, tuple)):
        assert len(data) > 0 and isinstance(data[0], np.ndarray)
        seg_shape = tuple([len(seg)] + list(seg[0].shape))
    elif seg is not None:
        raise TypeError("Data has to be either a numpy array or a list")

    if type(new_size) not in (tuple, list, np.ndarray):
        new_size = [new_size] * (len(data_shape) - 2)
    else:
        assert len(new_size) == len(
            data_shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your " \
                             "data (2d/3d)"

    shape = tuple(data_shape[2:])
    start = np.array(new_size) / 2. - np.array(shape) / 2.

    res_data = np.ones([data_shape[0], data_shape[1]] + list(new_size), dtype=data[0].dtype)
    res_seg = None
    if seg is not None:
        res_seg = np.zeros([seg_shape[0], seg_shape[1]] + list(new_size), dtype=seg[0].dtype)
    if pad_value_seg is not None:
        res_seg *= pad_value_seg
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            if pad_value_data is None:
                if len(shape) == 2:
                    pad_value_tmp = data[i][j, 0, 0]
                elif len(shape) == 3:
                    pad_value_tmp = data[i][j, 0, 0, 0]
                else:
                    raise Exception(
                        "Invalid dimension for data and seg. data and seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
            else:
                pad_value_tmp = pad_value_data
            res_data[i, j] = pad_value_tmp
            if len(shape) == 2:
                res_data[i, j, int(start[0]):int(start[0]) + int(shape[0]),
                int(start[1]):int(start[1]) + int(shape[1])] = data[i][j]
            elif len(shape) == 3:
                res_data[i, j, int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
                int(start[2]):int(start[2]) + int(shape[2])] = data[i][j]
            else:
                raise Exception(
                    "Invalid dimension for data and seg. data and seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
            if seg is not None:
                for j in range(seg_shape[1]):
                    if pad_value_seg is None:
                        if len(shape) == 2:
                            pad_value_tmp = seg[i][j, 0, 0]
                        elif len(shape) == 3:
                            pad_value_tmp = seg[i][j, 0, 0, 0]
                        else:
                            raise Exception(
                                "Invalid dimension for data and seg. data and seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
                    else:
                        pad_value_tmp = pad_value_seg
                    res_seg[i, j] = pad_value_tmp
                    if len(shape) == 2:
                        res_seg[i, j, int(start[0]):int(start[0]) + int(shape[0]),
                        int(start[1]):int(start[1]) + int(shape[1])] = seg[i][j]
                    elif len(shape) == 3:
                        res_seg[i, j, int(start[0]):int(start[0]) + int(shape[0]),
                        int(start[1]):int(start[1]) + int(shape[1]), int(start[2]):int(start[2]) + int(shape[2])] = \
                            seg[i][j]

    return res_data, res_seg
