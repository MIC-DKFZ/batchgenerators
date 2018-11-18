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

from batchgenerators.augmentations.utils import pad_nd_image

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

        if data_shape[i+2] - crop_size[i] - margins[i] > margins[i]:
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
    """
    crops data and seg (seg may be None) to crop_size. Whether this will be achieved via center or random crop is
    determined by crop_type. Margin will be respected only for random_crop and will prevent the crops form being closer
    than margin to the respective image border. crop_size can be larger than data_shape - margin -> data/seg will be
    padded with zeros in that case. margins can be negative -> results in padding of data/seg followed by cropping with
    margin=0 for the appropriate axes

    :param data:
    :param seg:
    :param crop_size:
    :param margins:
    :param crop_type:
    :return:
    """
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
        warn("Crop_size + margin > data_shape. Data will be padded to accomodate crop_size")

    data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)
    if seg is not None:
        seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype)
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


def pad_nd_image_and_seg(data, seg, new_shape=None, must_be_divisible_by=None, pad_mode_data='constant',
                         np_pad_kwargs_data=None, pad_mode_seg='constant', np_pad_kwargs_seg=None):
    assert len(new_shape) == len(data.shape), "data_shape and new_shape must have the same dimensionality"
    sample_data = pad_nd_image(data, new_shape, mode=pad_mode_data, kwargs=np_pad_kwargs_data,
                               return_slicer=False, shape_must_be_divisible_by=must_be_divisible_by)
    if seg is not None:
        sample_seg = pad_nd_image(seg, new_shape, mode=pad_mode_seg, kwargs=np_pad_kwargs_seg,
                                  return_slicer=False, shape_must_be_divisible_by=must_be_divisible_by)
    else:
        sample_seg = None
    return sample_data, sample_seg

