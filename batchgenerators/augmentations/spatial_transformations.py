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
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation
from skimage.transform import resize


def augment_resize(data, target_size, order=3, order_seg=1, cval_seg=0, seg=None, concatenate_list=False):
    """
    Reshapes data (and seg) to target_size
    :param data: np.ndarray or list/tuple of np.ndarrays, must be (b, c, x, y(, z))) (if list/tuple then each entry
    must be of this shape!)
    :param target_size: int or list/tuple of int
    :param order: interpolation order for data (see skimage.transform.resize)
    :param order_seg: interpolation order for seg (see skimage.transform.resize)
    :param cval_seg: cval for segmentation (see skimage.transform.resize)
    :param seg: can be None, if not None then it will also be resampled to target_size. Can also be list/tuple of
    np.ndarray (just like data). Must also be (b, c, x, y(, z))
    :param concatenate_list: if you give list/tuple of data/seg and set concatenate_list=True then the result will be
    concatenated into one large ndarray (once again b, c, x, y(, z))
    :return:
    """
    if isinstance(data, np.ndarray):
        is_list = False
        assert (seg is None) or isinstance(seg, np.ndarray), "if data is ndarray then seg must be ndarray as well"
    elif isinstance(data, (list, tuple)):
        is_list = True
        assert (seg is None) or isinstance(seg,  (list, tuple)), "if data is list/tuple then seg must be list/tuple as well"
    else:
        raise TypeError("Data has to be either a numpy array or a list")

    if not is_list:
        data = [data]
        if seg is not None:
            seg = [seg]
        concatenate_list = True

    result_data = []
    for i in range(len(data)):
        dimensionality = len(data[i].shape) - 2
        if not isinstance(target_size, (list, tuple)):
            target_size_here = [target_size] * dimensionality
        else:
            assert len(target_size) == dimensionality, "If you give a tuple/list as target size, make sure it has " \
                                                       "the same dimensionality as data!"
            target_size_here = list(target_size)

        # resize only supports 3d images. And it makes sense to treat each color channel of each sample separately
        result_this_data = []
        for b in range(data[i].shape[0]):
            result_this_sample = []
            for c in range(data[i].shape[1]):
                result_this_sample.append(
                    resize(data[i][b, c].astype(float), target_size_here, order).astype(data[i].dtype)[None])
            result_this_sample = np.vstack(result_this_sample)
            result_this_data.append(result_this_sample[None])
        result_this_data = np.vstack(result_this_data)
        result_data.append(result_this_data)

    if concatenate_list:
        result_data = np.vstack(result_data)

    if seg is not None:
        result_seg = []
        for i in range(len(seg)):
            dimensionality = len(seg[i].shape) - 2
            if not isinstance(target_size, (list, tuple)):
                target_size_here = [target_size] * dimensionality
            else:
                assert len(target_size) == dimensionality, "If you give a tuple/list as target size, make sure it has " \
                                                           "the same dimensionality as seg!"
                target_size_here = list(target_size)

            # resize only supports 3d images. And it makes sense to treat each color channel of each sample separately
            result_this_seg = []
            for b in range(seg[i].shape[0]):
                result_this_sample = []
                for c in range(seg[i].shape[1]):
                    result_this_sample.append(
                        resize_segmentation(seg[i][b, c].astype(float), target_size_here, order_seg, cval_seg)[None])
                result_this_sample = np.vstack(result_this_sample)
                result_this_seg.append(result_this_sample[None])
            result_this_seg = np.vstack(result_this_seg)
            result_seg.append(result_this_seg)

        if concatenate_list:
            result_seg = np.vstack(result_seg)
    else:
        result_seg = None

    return result_data, result_seg


def augment_zoom(data, zoom_factors, order=3, order_seg=1, cval_seg=0, seg=None, concatenate_list=False):
    """
    zooms data (and seg) by factor zoom_factors
    :param data: np.ndarray or list/tuple of np.ndarrays, must be (b, c, x, y(, z))) (if list/tuple then each entry
    must be of this shape!)
    :param zoom_factors: int or list/tuple of int
    :param order: interpolation order for data (see skimage.transform.resize)
    :param order_seg: interpolation order for seg (see skimage.transform.resize)
    :param cval_seg: cval for segmentation (see skimage.transform.resize)
    :param seg: can be None, if not None then it will also be zoomed by zoom_factors. Can also be list/tuple of
    np.ndarray (just like data). Must also be (b, c, x, y(, z))
    :param concatenate_list: if you give list/tuple of data/seg and set concatenate_list=True then the result will be
    concatenated into one large ndarray (once again b, c, x, y(, z))
    :return:
    """
    if isinstance(data, np.ndarray):
        is_list = False
    elif isinstance(data, (list, tuple)):
        is_list = True
        assert len(data) > 0 and all([isinstance(i, np.ndarray) for i in data])
    else:
        raise TypeError("Data has to be either a numpy array or a list")

    if seg is not None:
        if is_list:
            assert isinstance(seg, (list, tuple)), "if data is list/tuple then seg must be, too"
            assert len(seg) > 0 and all([isinstance(i, np.ndarray) for i in seg])
        else:
            assert isinstance(seg, np.ndarray)

    if not is_list:
        data = [data]
        if seg is not None:
            seg = [seg]
        concatenate_list = True

    result_data = []
    for i in range(len(data)):
        dimensionality = len(data[i].shape) - 2
        shape = np.array(data[i].shape[2:])
        if not isinstance(zoom_factors, (list, tuple)):
            zoom_factors_here = np.array([zoom_factors] * dimensionality)
        else:
            assert len(zoom_factors) == dimensionality, "If you give a tuple/list as target size, make sure it has " \
                                                        "the same dimensionality as data!"
            zoom_factors_here = np.array(zoom_factors)
        target_shape_here = np.round(shape * zoom_factors_here).astype(int)

        # resize only supports 3d images. And it makes sense to treat each color channel of each sample separately
        result_this_data = []
        for b in range(data[i].shape[0]):
            result_this_sample = []
            for c in range(data[i].shape[1]):
                result_this_sample.append(
                    resize(data[i][b, c].astype(float), target_shape_here, order).astype(data[i].dtype)[None])
            result_this_sample = np.vstack(result_this_sample)
            result_this_data.append(result_this_sample[None])
        result_this_data = np.vstack(result_this_data)
        result_data.append(result_this_data)

    if concatenate_list:
        result_data = np.vstack(result_data)

    if seg is not None:
        result_seg = []
        for i in range(len(seg)):
            dimensionality = len(seg[i].shape) - 2
            shape = np.array(seg[i].shape[2:])
            if not isinstance(zoom_factors, (list, tuple)):
                zoom_factors_here = np.array([zoom_factors] * dimensionality)
            else:
                assert len(zoom_factors) == dimensionality, "If you give a tuple/list as target size, make sure it has " \
                                                            "the same dimensionality as seg!"
                zoom_factors_here = np.array(zoom_factors)
            target_shape_here = np.round(shape * zoom_factors_here).astype(int)

            # resize only supports 3d images. And it makes sense to treat each color channel of each sample separately
            result_this_seg = []
            for b in range(seg[i].shape[0]):
                result_this_sample = []
                for c in range(seg[i].shape[1]):
                    result_this_sample.append(
                        resize_segmentation(seg[i][b, c].astype(float), target_shape_here, order_seg, cval_seg).astype(seg[i].dtype)[None])
                result_this_sample = np.vstack(result_this_sample)
                result_this_seg.append(result_this_sample[None])
            result_this_seg = np.vstack(result_this_seg)
            result_seg.append(result_this_seg)

        if concatenate_list:
            result_seg = np.vstack(result_seg)
    else:
        result_seg = None

    return result_data, result_seg


def augment_mirroring(data, seg=None, axes=(2, 3, 4)):
    data = np.copy(data)
    if seg is not None:
        seg = np.copy(seg)
    if (len(data.shape) != 4) and (len(data.shape) != 5):
        raise Exception(
            "Invalid dimension for data and seg. data and seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
    BATCH_SIZE = data.shape[0]
    idx = np.arange(BATCH_SIZE)
    for id in idx:
        if 2 in axes and np.random.uniform() < 0.5:
            data[id, :, :] = data[id, :, ::-1]
            if seg is not None:
                seg[id, :, :] = seg[id, :, ::-1]
        if 3 in axes and np.random.uniform() < 0.5:
            data[id, :, :, :] = data[id, :, :, ::-1]
            if seg is not None:
                seg[id, :, :, :] = seg[id, :, :, ::-1]
        if 4 in axes and len(data.shape) == 5:
            if np.random.uniform() < 0.5:
                data[id, :, :, :, :] = data[id, :, :, :, ::-1]
                if seg is not None:
                    seg[id, :, :, :, :] = seg[id, :, :, :, ::-1]
    return data, seg


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
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True):
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
        if do_elastic_deform:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
        if do_rotation:
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
        if do_scale:
            if np.random.random() < 0.5 and scale[0] < 1:
                sc = np.random.uniform(scale[0], 1)
            else:
                sc = np.random.uniform(max(scale[0], 1), scale[1])
            coords = scale_coords(coords, sc)
        # now find a nice center location
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
    return data_result, seg_result


def augment_transpose_axes(data, seg, axes=(2, 3, 4)):
    axes = list(np.array(axes) - 1)  # need list to allow shuffle; -1 because we iterate over samples in batch
    data_res = np.copy(data)
    seg_res = None
    if seg is not None:
        seg_res = np.copy(seg)

    assert np.max(axes) <= len(data.shape), "axes must only contain valid axis ids"
    for b in range(data.shape[0]):
        np.random.shuffle(axes)
        data_res[b] = data_res[b].transpose(*([0] + axes))
        if seg is not None:
            seg_res[b] = seg_res[b].transpose(*([0] + axes))
    return data_res, seg_res


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