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

import copy

from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding
from batchgenerators.augmentations.utils import convert_seg_to_bounding_box_coordinates
from batchgenerators.augmentations.utils import transpose_channels

import numpy as np

class NumpyToTensor(AbstractTransform):
    """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
    """
    def __call__(self, **data_dict):
        import torch

        for key, val in data_dict.items():
            if isinstance(val, np.ndarray):
                data_dict[key] = torch.from_numpy(val)

        return data_dict

class ListToNumpy(AbstractTransform):
    """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
    """
    def __call__(self, **data_dict):

        for key, val in data_dict.items():
            if isinstance(val, (list, tuple)):
                data_dict[key] = np.asarray(val)

        return data_dict

class ListToTensor(AbstractTransform):
    """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
    """
    def __call__(self, **data_dict):
        import torch

        for key, val in data_dict.items():
            if isinstance(val, (list, tuple)):
                data_dict[key] = [torch.from_numpy(smpl) for smpl in val]

        return data_dict



class ConvertSegToOnehotTransform(AbstractTransform):
    """Creates a one hot encoding of one of the seg channels. Important when using our soft dice loss.

    Args:
        classes (tuple of int): All the class labels that are in the dataset

        seg_channel (int): channel of seg to convert to onehot

        output_key (string): key to use for output of the one hot encoding. Default is 'seg' but that will override any
        other existing seg channels. Therefore you have the option to change that. BEWARE: Any non-'seg' segmentations
        will not be augmented anymore. Use this only at the very end of your pipeline!
    """
    def __init__(self, classes, seg_channel=0, output_key="seg"):
        self.output_key = output_key
        self.seg_channel = seg_channel
        self.classes = classes

    def __call__(self, **data_dict):
        seg = data_dict.get("seg")
        if seg is not None:
            new_seg = np.zeros([seg.shape[0], len(self.classes)] + list(seg.shape[2:]), dtype=seg.dtype)
            for b in range(seg.shape[0]):
                new_seg[b] = convert_seg_image_to_one_hot_encoding(seg[b, self.seg_channel], self.classes)
            data_dict[self.output_key] = new_seg
        else:
            from warnings import warn
            warn("calling ConvertSegToOnehotTransform but there is no segmentation")

        return data_dict




class ConvertSegToBoundingBoxCoordinates(AbstractTransform):
    """ Converts segmentation masks into bounding box coordinates. Works only for one object per image
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, **data_dict):
        data_dict['bb_target'] = convert_seg_to_bounding_box_coordinates(data_dict['seg'], data_dict['pid'], self.dim)

        return data_dict

class TransposeChannels(AbstractTransform):
    """ Converts segmentation masks into bounding box coordinates. Works only for one object per image
    """

    def __call__(self, **data_dict):
        data_dict['data'] = transpose_channels(data_dict['data'])
        data_dict['seg'] = transpose_channels(data_dict['seg'])

        return data_dict


class RemoveLabelTransform(AbstractTransform):
    '''
    Replaces all pixels in data_dict[input_key] that have value remove_label with replace_with and saves the result to
    data_dict[output_key]
    '''
    def __init__(self, remove_label, replace_with=0, input_key="seg", output_key="seg"):
        self.output_key = output_key
        self.input_key = input_key
        self.replace_with = replace_with
        self.remove_label = remove_label

    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]
        seg[seg == self.remove_label] = self.replace_with
        data_dict[self.output_key] = seg
        return data_dict


class RenameTransform(AbstractTransform):
    '''
    Saves the value of data_dict[in_key] to data_dict[out_key]. Does not remove data_dict[in_key] from the dict.
    '''
    def __init__(self, in_key, out_key):
        self.out_key = out_key
        self.in_key = in_key

    def __call__(self, **data_dict):
        data_dict[self.out_key] = data_dict[self.in_key]
        return data_dict


class CopyTransform(AbstractTransform):
    """Renames some attributes of the data_dict (e. g. transformations can be applied on different dict items).

    Args:
        re_dict: Dict with the key=origin name, val=new name.
        copy: Copy (and not move (cp vs mv)) to new target val and leave the old ones in place

    Example:
        >>> transforms.CopyTransform({"data": "data2", "seg": "seg2"})
    """

    def __init__(self, re_dict, copy=False):
        self.re_dict = re_dict
        self.copy = copy

    def __call__(self, **data_dict):
        new_dict = {}
        for key, val in data_dict.items():
            if key in self.re_dict:
                n_key = self.re_dict[key]
                if isinstance(n_key, (tuple, list)):
                    for k in n_key:
                        if self.copy:
                            new_dict[k] = copy.deepcopy(val)
                        else:
                            new_dict[k] = val
                else:
                    if self.copy:
                        new_dict[n_key] = copy.deepcopy(val)
                    else:
                        new_dict[n_key] = val
            if key not in self.re_dict:
                new_dict[key] = val

            if self.copy:
                    new_dict[key] = copy.deepcopy(val)

        return new_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"


class ReshapeTransform(AbstractTransform):

    def __init__(self, new_shape, key="data"):
        self.key = key
        self.new_shape = new_shape

    def __call__(self, **data_dict):

        data_arr = data_dict[self.key]
        data_shape = data_arr.shape
        c, h, w = data_shape[-3:]

        target_shape = []
        for val in self.new_shape:
            if val == "c":
                target_shape.append(c)
            elif val == "h":
                target_shape.append(h)
            elif val == "w":
                target_shape.append(w)
            else:
                target_shape.append(val)

        data_dict[self.key] = np.reshape(data_dict[self.key], target_shape)

        return data_dict