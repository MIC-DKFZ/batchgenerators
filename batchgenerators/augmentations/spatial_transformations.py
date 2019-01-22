#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug


def augment_rot90(sample_data, sample_seg, num_rot=(1, 2, 3), axes=(0, 1, 2)):
    """

    :param sample_data:
    :param sample_seg:
    :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
    :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
    :return:
    """
    num_rot = np.random.choice(num_rot)
    axes = np.random.choice(axes, size=2, replace=False)
    axes.sort()
    axes = [i + 1 for i in axes]
    sample_data = np.rot90(sample_data, num_rot, axes)
    if sample_seg is not None:
        sample_seg = np.rot90(sample_seg, num_rot, axes)
    return sample_data, sample_seg


def augment_resize(sample_data, sample_seg, target_size, order=3, order_seg=1, cval_seg=0):
    """
    Reshapes data (and seg) to target_size
    :param sample_data: np.ndarray or list/tuple of np.ndarrays, must be (c, x, y(, z))) (if list/tuple then each entry
    must be of this shape!)
    :param target_size: int or list/tuple of int
    :param order: interpolation order for data (see skimage.transform.resize)
    :param order_seg: interpolation order for seg (see skimage.transform.resize)
    :param cval_seg: cval for segmentation (see skimage.transform.resize)
    :param sample_seg: can be None, if not None then it will also be resampled to target_size. Can also be list/tuple of
    np.ndarray (just like data). Must also be (c, x, y(, z))
    :return:
    """
    dimensionality = len(sample_data.shape) - 1
    if not isinstance(target_size, (list, tuple)):
        target_size_here = [target_size] * dimensionality
    else:
        assert len(target_size) == dimensionality, "If you give a tuple/list as target size, make sure it has " \
                                                   "the same dimensionality as data!"
        target_size_here = list(target_size)

    sample_data = resize_multichannel_image(sample_data, target_size_here, order)

    if sample_seg is not None:
        target_seg = np.ones([sample_seg.shape[0]] + target_size_here)
        for c in range(sample_seg.shape[0]):
            target_seg[c] = resize_segmentation(sample_seg[c], target_size_here, order_seg, cval_seg)
    else:
        target_seg = None

    return sample_data, target_seg


def augment_zoom(sample_data, sample_seg, zoom_factors, order=3, order_seg=1, cval_seg=0):
    """
    zooms data (and seg) by factor zoom_factors
    :param sample_data: np.ndarray or list/tuple of np.ndarrays, must be (c, x, y(, z))) (if list/tuple then each entry
    must be of this shape!)
    :param zoom_factors: int or list/tuple of int (multiplication factor for the input size)
    :param order: interpolation order for data (see skimage.transform.resize)
    :param order_seg: interpolation order for seg (see skimage.transform.resize)
    :param cval_seg: cval for segmentation (see skimage.transform.resize)
    :param sample_seg: can be None, if not None then it will also be zoomed by zoom_factors. Can also be list/tuple of
    np.ndarray (just like data). Must also be (c, x, y(, z))
    :return:
    """

    dimensionality = len(sample_data.shape) - 1
    shape = np.array(sample_data.shape[1:])
    if not isinstance(zoom_factors, (list, tuple)):
        zoom_factors_here = np.array([zoom_factors] * dimensionality)
    else:
        assert len(zoom_factors) == dimensionality, "If you give a tuple/list as target size, make sure it has " \
                                                    "the same dimensionality as data!"
        zoom_factors_here = np.array(zoom_factors)
    target_shape_here = list(np.round(shape * zoom_factors_here).astype(int))

    sample_data = resize_multichannel_image(sample_data, target_shape_here, order)

    if sample_seg is not None:
        target_seg = np.ones([sample_seg.shape[0]] + target_shape_here)
        for c in range(sample_seg.shape[0]):
            target_seg[c] = resize_segmentation(sample_seg[c], target_shape_here, order_seg, cval_seg)
    else:
        target_seg = None

    return sample_data, target_seg


def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg


def augment_channel_translation(data, const_channel=0, max_shifts=None):
    if max_shifts is None:
        max_shifts = {'z': 2, 'y': 2, 'x': 2}

    shape = data.shape

    const_data = data[:, [const_channel]]
    trans_data = data[:, [i for i in range(shape[1]) if i != const_channel]]

    # iterate the batch dimension
    for j in range(shape[0]):

        slice = trans_data[j]

        ixs = {}
        pad = {}

        if len(shape) == 5:
            dims = ['z', 'y', 'x']
        else:
            dims = ['y', 'x']

        # iterate the image dimensions, randomly draw shifts/translations
        for i, v in enumerate(dims):
            rand_shift = np.random.choice(list(range(-max_shifts[v], max_shifts[v], 1)))

            if rand_shift > 0:
                ixs[v] = {'lo': 0, 'hi': -rand_shift}
                pad[v] = {'lo': rand_shift, 'hi': 0}
            else:
                ixs[v] = {'lo': abs(rand_shift), 'hi': shape[2 + i]}
                pad[v] = {'lo': 0, 'hi': abs(rand_shift)}

        # shift and pad so as to retain the original image shape
        if len(shape) == 5:
            slice = slice[:, ixs['z']['lo']:ixs['z']['hi'], ixs['y']['lo']:ixs['y']['hi'],
                    ixs['x']['lo']:ixs['x']['hi']]
            slice = np.pad(slice, ((0, 0), (pad['z']['lo'], pad['z']['hi']), (pad['y']['lo'], pad['y']['hi']),
                                   (pad['x']['lo'], pad['x']['hi'])),
                           mode='constant', constant_values=(0, 0))
        if len(shape) == 4:
            slice = slice[:, ixs['y']['lo']:ixs['y']['hi'], ixs['x']['lo']:ixs['x']['hi']]
            slice = np.pad(slice, ((0, 0), (pad['y']['lo'], pad['y']['hi']), (pad['x']['lo'], pad['x']['hi'])),
                           mode='constant', constant_values=(0, 0))

        trans_data[j] = slice

    data_return = np.concatenate([const_data, trans_data], axis=1)
    return data_return


def augment_spatial(data, seg, patch_size, patch_center_dist_from_border=30,
                    do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
                    p_scale_per_sample=1, p_rot_per_sample=1):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]
    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False
        if np.random.uniform() < p_el_per_sample and do_elastic_deform:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = True
        if np.random.uniform() < p_rot_per_sample and do_rotation:
            if angle_x[0] == angle_x[1]:
                a_x = angle_x[0]
            else:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            if dim == 3:
                if angle_y[0] == angle_y[1]:
                    a_y = angle_y[0]
                else:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                if angle_z[0] == angle_z[1]:
                    a_z = angle_z[0]
                else:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True
        if np.random.uniform() < p_scale_per_sample and do_scale:
            if np.random.random() < 0.5 and scale[0] < 1:
                sc = np.random.uniform(scale[0], 1)
            else:
                sc = np.random.uniform(max(scale[0], 1), scale[1])
            coords = scale_coords(coords, sc)
            modified_coords = True
        # now find a nice center location
        if modified_coords:
            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(patch_center_dist_from_border[d],
                                            data.shape[d + 2] - patch_center_dist_from_border[d])
                else:
                    ctr = int(np.round(data.shape[d + 2] / 2.))
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                     border_mode_data, cval=border_cval_data)
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                        border_mode_seg, cval=border_cval_seg, is_seg=True)
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id:sample_id + 1]
            if random_crop:
                margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
            else:
                d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]
    return data_result, seg_result


def augment_transpose_axes(data_sample, seg_sample, axes=(0, 1, 2)):
    """

    :param data_sample: c,x,y(,z)
    :param seg_sample: c,x,y(,z)
    :param axes: list/tuple
    :return:
    """
    axes = list(np.array(axes) + 1)  # need list to allow shuffle; +1 to accomodate for color channel

    assert np.max(axes) <= len(data_sample.shape), "axes must only contain valid axis ids"
    static_axes = list(range(len(data_sample.shape)))
    for i in axes: static_axes[i] = -1
    np.random.shuffle(axes)

    ctr = 0
    for j, i in enumerate(static_axes):
        if i == -1:
            static_axes[j] = axes[ctr]
            ctr += 1

    data_sample = data_sample.transpose(*static_axes)
    if seg_sample is not None:
        seg_sample = seg_sample.transpose(*static_axes)
    return data_sample, seg_sample


def flip_vector_axis(data):
    data = np.copy(data)
    if (len(data.shape) != 4) and (len(data.shape) != 5) or data.shape[1] != 9:
        raise Exception("Invalid dimension for data. Data should be either [BATCH_SIZE, 9, x, y] or [BATCH_SIZE, 9, x, y, z]")
    axis = np.random.choice(["x", "y", "z"])   #chose axes to flip
    BATCH_SIZE = data.shape[0]
    for id in np.arange(BATCH_SIZE):
        if np.random.uniform() < 0.5:
            if axis == "x":
                data[id, 0] *= -1
                data[id, 3] *= -1
                data[id, 6] *= -1
            elif axis == "y":
                data[id, 1] *= -1
                data[id, 4] *= -1
                data[id, 7] *= -1
            elif axis == "z":
                data[id, 2] *= -1
                data[id, 5] *= -1
                data[id, 8] *= -1

    return data