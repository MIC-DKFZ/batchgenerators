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
import warnings
from warnings import warn
import numpy as np
warnings.simplefilter("once", UserWarning)


def center_crop(data, crop_size, seg=None):
    return crop(data, seg, crop_size, 0, 'center')


def get_lbs_for_random_crop(crop_size, data_shape, margins):
    """

    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the whole thing!
    :param margins:
    :return:
    """
    lbs = []
    for i in range(len(data_shape) - 2):
        if crop_size[i] > data_shape[i + 2]:
            warn("Crop_size > data_shape. data: %s, crop: %s. Data will be padded to accomodate crop_size" % (str(data_shape), str(crop_size)), UserWarning)

        if data_shape[i+2] - crop_size[i] - margins[i] >= margins[i]:
            lbs.append(np.random.randint(margins[i], data_shape[i+2] - crop_size[i] - margins[i]))
        else:
            warn("Random crop is falling back to center crop because the crop along with the desired margin does "
                 "not fit the data. "
                 "data: %s, crop_size: %s, margin: %s" % (str(data_shape), str(crop_size), str(margins)), UserWarning)
            lbs.append((data_shape[i+2] - crop_size[i]) // 2)
    return lbs


def get_lbs_for_center_crop(crop_size, data_shape):
    """
    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the whole thing!
    :return:
    """
    lbs = []
    for i in range(len(data_shape) - 2):
        if crop_size[i] > data_shape[i + 2]:
            warn("Crop_size > data_shape. data: %s, crop: %s. Data will be padded to accomodate crop_size" % (str(data_shape), str(crop_size)), UserWarning)
        lbs.append((data_shape[i + 2] - crop_size[i]) // 2)
    return lbs


def crop(data, seg=None, crop_size=128, margins=(0, 0, 0), crop_type="center"):
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError("data has to be either a numpy array or a list")

    data_shape = tuple([len(data)] + list(data[0].shape))
    data_dtype = data[0].dtype
    dim = len(data_shape) - 2

    if seg is not None:
        seg_shape = tuple([len(seg)] + list(seg[0].shape))
        seg_dtype = seg[0].dtype

        if not isinstance(seg, (list, tuple, np.ndarray)):
            raise TypeError("data has to be either a numpy array or a list")

        assert all([i == j for i, j in zip(seg_shape[2:], data_shape[2:])]), "data and seg must have the same spatial " \
                                                                             "dimensions. Data: %s, seg: %s" % \
                                                                             (str(data_shape), str(seg_shape))

    if type(crop_size) not in (tuple, list, np.ndarray):
        crop_size = [crop_size] * dim
    else:
        assert len(crop_size) == len(
            data_shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your " \
                             "data (2d/3d)"

    if not isinstance(margins, (np.ndarray, tuple, list)):
        margins = [margins] * dim

    if any([crop_size[d] > (data_shape[d+2] + 2*abs(min(0, margins[d]))) for d in range(dim)]):
        warn("Crop_size > data_shape (considering margin). Data will be padded to accomodate crop_size")

    if dim == 2:
        data_return = np.zeros((data_shape[0], data_shape[1], crop_size[0], crop_size[1]), dtype=data_dtype)
    else:
        data_return = np.zeros((data_shape[0], data_shape[1], crop_size[0], crop_size[1], crop_size[2]),
                               dtype=data_dtype)
    if seg is not None:
        if dim == 2:
            seg_return = np.zeros((seg_shape[0], seg_shape[1], crop_size[0], crop_size[1]), dtype=seg_dtype)
        else:
            seg_return = np.zeros((seg_shape[0], seg_shape[1], crop_size[0], crop_size[1], crop_size[2]),
                                  dtype=seg_dtype)
    else:
        seg_return = None

    for b in range(data_shape[0]):
        data_shape_here = [data_shape[0]] + list(data[b].shape)

        if crop_type == "center":
            lbs = get_lbs_for_center_crop(crop_size, data_shape_here)
        elif crop_type == "random":
            lbs = get_lbs_for_random_crop(crop_size, data_shape_here, margins)
        else:
            raise NotImplementedError("crop_type must be either center or random")

        need_to_pad = [[0, 0]] + [[abs(min(0, lbs[d])),
                                   abs(min(0, data_shape_here[d + 2] - (lbs[d] + crop_size[d])))]
                                  for d in range(dim)]

        if any([i > 0 for j in need_to_pad for i in j]):
            data_2 = np.pad(data[b], need_to_pad, 'constant', constant_values=0)
            if seg_return is not None:
                seg_2 = np.pad(seg[b], need_to_pad, 'constant', constant_values=0)
            else:
                seg_2 = None
        else:
            data_2 = data[b]
            if seg_return is not None:
                seg_2 = seg[b]
            else:
                seg_2 = None

        lbs = [lbs[d] + need_to_pad[d+1][0] for d in range(dim)]
        assert all([i >= 0 for i in lbs]), "just a failsafe"
        slicer = [slice(0, data_shape_here[1])] + [slice(lbs[d], lbs[d]+crop_size[d]) for d in range(dim)]
        data_return[b] = data_2[slicer]
        if seg_return is not None:
            seg_return[b] = seg_2[slicer]

    return data_return, seg_return


def random_crop(data, seg=None, crop_size=128, margins=[0, 0, 0]):
    return crop(data, seg, crop_size, margins, 'random')


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


def pad_to_aspect_ratio_2d(data, ratio, seg=None, pad_value_data=None, pad_value_seg=None):

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
