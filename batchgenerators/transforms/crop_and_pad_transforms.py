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


from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop, center_crop_seg, random_crop, pad, \
    random_crop


class CenterCropTransform(AbstractTransform):
    """ Crops data and seg (if available) in the center

    Args:
        output_size (int or tuple of int): Output patch size

    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        seg = data_dict.get("seg")
        data, seg = center_crop(data, self.crop_size, seg)

        data_dict["data"] = data
        if seg is not None:
            data_dict["seg"] = seg

        return data_dict


class CenterCropSegTransform(AbstractTransform):
    """ Crops seg in the center (required if you are using unpadded convolutions in a segmentation network).
    Leaves data as it is

    Args:
        output_size (int or tuple of int): Output patch size

    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, **data_dict):
        seg = data_dict.get("seg")

        if seg is not None:
            data_dict["seg"] = center_crop_seg(seg, self.output_size)
        else:
            from warnings import warn
            warn("You shall not pass data_dict without seg: Used CenterCropSegTransform, but there is no seg", Warning)
        return data_dict


class RandomCropTransform(AbstractTransform):
    """ Randomly crops data and seg (if available)

    Args:
        crop_size (int or tuple of int): Output patch size

        margins (tuple of int): how much distance should the patch border have to the image broder (bilaterally)?

    """

    def __init__(self, crop_size=128, margins=(0, 0, 0)):
        self.margins = margins
        self.crop_size = crop_size

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        seg = data_dict.get("seg")

        data, seg = random_crop(data, seg, self.crop_size, self.margins)

        data_dict["data"] = data
        if seg is not None:
            data_dict["seg"] = seg

        return data_dict


class PadTransform(AbstractTransform):
    """Pads data and seg

    Args:
        new_size (tuple of int): Size after padding

        pad_value_data: constant value with which to pad data. If None it uses the image value of [0, 0(, 0)] for each
        sample and channel

        pad_value_seg: constant value with which to pad segIf None it uses the seg value of [0, 0(, 0)] for each sample
        and channel
    """

    def __init__(self, new_size, pad_value_data=None, pad_value_seg=None):
        self.pad_value_seg = pad_value_seg
        self.pad_value_data = pad_value_data
        self.new_size = new_size

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        seg = data_dict.get("seg")

        data, seg = pad(data, self.new_size, seg, self.pad_value_data, self.pad_value_seg)

        data_dict["data"] = data
        if seg is not None:
            data_dict["seg"] = seg

        return data_dict


class FillupPadTransform(AbstractTransform):
    """Pads data and seg if not already having the minimal given size

    Args:
        min_size (tuple of int): Size after padding

        pad_value_data: constant value with which to pad data. If None it uses the image value of [0, 0(, 0)] for each
        sample and channel

        pad_value_seg: constant value with which to pad segIf None it uses the seg value of [0, 0(, 0)] for each sample
        and channel
    """

    def __init__(self, min_size, pad_value_data=None, pad_value_seg=None):
        self.pad_value_seg = pad_value_seg
        self.pad_value_data = pad_value_data
        self.new_size = min_size

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        seg = data_dict.get("seg")

        data, seg = fillup_pad(data, self.new_size, seg, self.pad_value_data, self.pad_value_seg)

        data_dict["data"] = data
        if seg is not None:
            data_dict["seg"] = seg

        return data_dict
