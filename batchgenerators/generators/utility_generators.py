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


from builtins import range, str

import numpy as np
from skimage.transform import resize


def convert_seg_flat_to_binary_label_indicator_array(seg_flat, num_classes=5):
    seg2 = np.zeros((len(seg_flat), num_classes))
    for i in range(seg2.shape[0]):
        seg2[i, int(seg_flat[i])] = 1
    return seg2


def create_one_hot_encoding_generator(generator, num_classes, output_key='seg', input_key='seg'):
    '''
    If you want to keep the original segmentation, use another output_key
    :param generator:
    :param num_classes:
    :param output_key:
    :return:
    '''
    for data_dict in generator:
        old_seg = data_dict[input_key]
        new_seg = np.zeros((old_seg.shape[0], old_seg.shape[1], np.prod(old_seg.shape[2:]), num_classes),
                           dtype=np.float32)
        for b in range(data_dict[input_key].shape[0]):
            for ch in range(data_dict[input_key].shape[1]):
                new_seg[b, ch] = convert_seg_flat_to_binary_label_indicator_array(data_dict[input_key][b][ch].flatten(),
                                                                                  num_classes)
        data_dict[output_key] = new_seg
        yield data_dict


def soft_rescale_seg_for_deep_supervision_generator(generator, rescaling_factors, output_key='seg_rescaled',
                                                    input_key='seg', unique_vals=None, convert_to_oneHot=True):
    # generates a soft segmentation. That means that provided we have a hard segmentation map (one image containing 1,
    # 2, 3, 4 etc) we first create a one hot encoding [(x, y, z) becomes (x, y, z, c) where c is the number of different labels], then smooth each
    # c separately
    # unique_vals gives all labels that exist in the dataset, including background!
    for data_dict in generator:
        seg = data_dict[input_key]  # seg must be shape (b, t, x, y(, z))
        data_dict[output_key] = {}
        if unique_vals is None:
            unique_vals = np.unique(seg)
        for r in rescaling_factors:
            new_shape = np.array(seg.shape[2:]) * r
            if (new_shape % 1).sum() != 0:
                raise ValueError(
                    "If rescaling_factor < 1 then the shape of data_dict[input_key] times slace factor must yield an int. shape: %s, rescaling_factor: %f" % (
                    str(seg.shape[2:]), r))
            new_shape = new_shape.astype(int)
            res = np.zeros(list(seg.shape[:2]) + [len(unique_vals)] + list(new_shape),
                           dtype=np.float32)  # (b, t, c, x, y, (z))
            for b in range(seg.shape[0]):
                for t in range(seg.shape[1]):
                    for i, c in enumerate(unique_vals):
                        if len(seg.shape) == 4:  # seg is 2d
                            res[b, t, i, :, :] = resize((seg[b, t] == c).astype(float), new_shape, 3,
                                                        preserve_range=True)
                        elif len(seg.shape) == 5:  # seg is 3d
                            res[b, t, i, :, :, :] = resize((seg[b, t] == c).astype(float), new_shape, 3,
                                                           preserve_range=True)
                        else:
                            raise ValueError("Invalid shape of seg: %s" % str(seg.shape))
            if convert_to_oneHot:
                data_dict[output_key][r] = res.reshape(seg.shape[0], seg.shape[1], len(unique_vals),
                                                       np.prod(new_shape)).transpose(
                    (0, 1, 3, 2))  # are you confused already?
            else:
                data_dict[output_key][r] = res
        yield data_dict
