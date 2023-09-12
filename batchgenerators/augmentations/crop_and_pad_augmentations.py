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

import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image
from typing import Union, Sequence


def center_crop(data, crop_size, seg=None):
    return crop(data, seg, crop_size, 0, 'center')


def get_lbs_for_random_crop(crop_size, data_shape, margins):
    """

    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the whole thing!
    :param margins:
    :return:
    """
    new_shape = data_shape - crop_size
    mask = new_shape > 2 * margins
    new_shape[mask] = np.random.randint(margins[mask], new_shape[mask] - margins[mask])
    new_shape[~mask] //= 2
    return new_shape


def get_lbs_for_center_crop(crop_size, data_shape):
    """
    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the only x,y(,z)!
    :return:
    """
    return (data_shape - crop_size) // 2


def crop(data: Union[Sequence[np.ndarray], np.ndarray], seg: Union[Sequence[np.ndarray], np.ndarray] = None,
         crop_size=128, margins=(0, 0, 0), crop_type="center",
         pad_mode='constant', pad_kwargs={'constant_values': 0},
         pad_mode_seg='constant', pad_kwargs_seg={'constant_values': 0}):
    """
    crops data and seg (seg may be None) to crop_size. Whether this will be achieved via center or random crop is
    determined by crop_type. Margin will be respected only for random_crop and will prevent the crops form being closer
    than margin to the respective image border. crop_size can be larger than data_shape - margin -> data/seg will be
    padded with zeros in that case. margins can be negative -> results in padding of data/seg followed by cropping with
    margin=0 for the appropriate axes

    :param data: b, c, x, y(, z)
    :param seg:
    :param crop_size:
    :param margins: distance from each border, can be int or list/tuple of ints (one element for each dimension).
    Can be negative (data/seg will be padded if needed)
    :param crop_type: random or center
    :return:
    """
    data_shape = (len(data),) + data[0].shape
    data_dtype = data[0].dtype
    dim = len(data_shape) - 2

    if seg is not None:
        seg_shape = (len(seg),) + seg[0].shape
        seg_dtype = seg[0].dtype

        assert np.array_equal(seg_shape[2:], data_shape[2:]), "data and seg must have the same spatial dimensions. " \
                                                              f"Data: {data_shape}, seg: {seg_shape}"

    if type(crop_size) not in (tuple, list, np.ndarray):
        crop_size = (crop_size,) * dim
    else:
        assert len(crop_size) == dim, ("If you provide a list/tuple as center crop make sure it has the same dimension "
                                       "as your data (2d/3d)")
    crop_size = np.asarray(crop_size)

    if not isinstance(margins, (np.ndarray, tuple, list)):
        margins = (margins,) * dim
    margins = np.asarray(margins)

    data_return = np.zeros((data_shape[0], data_shape[1], *crop_size), dtype=data_dtype)
    if seg is not None:
        seg_return = np.zeros((seg_shape[0], seg_shape[1], *crop_size), dtype=seg_dtype)
    else:
        seg_return = None

    for b in range(data_shape[0]):
        data_first_dim = data[b].shape[0]
        data_shape_here = np.array(data[b].shape[1:])
        if seg is not None:
            seg_first_dim = seg[b].shape[0]

        if crop_type == "center":
            lbs = get_lbs_for_center_crop(crop_size, data_shape_here)
        elif crop_type == "random":
            lbs = get_lbs_for_random_crop(crop_size, data_shape_here, margins)
        else:
            raise NotImplementedError("crop_type must be either center or random")

        zero = np.zeros(dim, dtype=int)
        temp1 = np.abs(np.minimum(lbs, zero))
        lbs_plus_crop_size = lbs + crop_size
        temp2 = np.abs(np.minimum(zero, data_shape_here - lbs_plus_crop_size))
        need_to_pad = ((0, 0),) + tuple(zip(temp1, temp2))
        need_to_pad = np.array(need_to_pad)

        # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
        ubs = np.minimum(data_shape_here, lbs_plus_crop_size)
        lbs = np.maximum(zero, lbs)

        slicer_data = (slice(0, data_first_dim), *[slice(lbs[d], ubs[d]) for d in range(dim)])
        data_cropped = data[b][slicer_data]

        if seg_return is not None:
            slicer_data = (slice(0, seg_first_dim),) + slicer_data[1:]
            seg_cropped = seg[b][slicer_data]

        if np.any(need_to_pad):
            data_return[b] = np.pad(data_cropped, need_to_pad, pad_mode, **pad_kwargs)
            if seg_return is not None:
                seg_return[b] = np.pad(seg_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
        else:
            data_return[b] = data_cropped
            if seg_return is not None:
                seg_return[b] = seg_cropped

    return data_return, seg_return


def random_crop(data, seg=None, crop_size=128, margins=[0, 0, 0]):
    return crop(data, seg, crop_size, margins, 'random')


def pad_nd_image_and_seg(data, seg, new_shape=None, must_be_divisible_by=None, pad_mode_data='constant',
                         np_pad_kwargs_data=None, pad_mode_seg='constant', np_pad_kwargs_seg=None):
    """
    Pads data and seg to new_shape. new_shape is thereby understood as min_shape (if data/seg is already larger then
    new_shape the shape stays the same for the dimensions this applies)
    :param data:
    :param seg:
    :param new_shape: if none then only must_be_divisible_by is applied
    :param must_be_divisible_by: UNet like architectures sometimes require the input to be divisibly by some number. This
    will modify new_shape if new_shape is not divisibly by this (by increasing it accordingly).
    must_be_divisible_by should be a list of int (one for each spatial dimension) and this list must have the same
    length as new_shape
    :param pad_mode_data: see np.pad
    :param np_pad_kwargs_data:see np.pad
    :param pad_mode_seg:see np.pad
    :param np_pad_kwargs_seg:see np.pad
    :return:
    """
    sample_data = pad_nd_image(data, new_shape, mode=pad_mode_data, kwargs=np_pad_kwargs_data,
                               return_slicer=False, shape_must_be_divisible_by=must_be_divisible_by)
    if seg is not None:
        sample_seg = pad_nd_image(seg, new_shape, mode=pad_mode_seg, kwargs=np_pad_kwargs_seg,
                                  return_slicer=False, shape_must_be_divisible_by=must_be_divisible_by)
    else:
        sample_seg = None
    return sample_data, sample_seg
