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
from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding
import numpy as np

class NumpyToTensor(AbstractTransform):
    """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
    """
    def __call__(self, **data_dict):
        import torch

        data = data_dict.get("data")
        seg = data_dict.get("seg")

        assert isinstance(data, np.ndarray)

        data_dict["data"] = torch.from_numpy(data)
        if seg is not None:
            data_dict["seg"] = torch.from_numpy(seg)

        return data_dict

class ListToTensor(AbstractTransform):
    """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
    """
    def __call__(self, **data_dict):
        import torch

        data = data_dict.get("data")
        seg = data_dict.get("seg")

        assert isinstance(data, (list, tuple))

        data_ret = [torch.from_numpy(data_smpl) for data_smpl in data]
        data_dict["data"] = data_ret
        if seg is not None:
            seg_ret = [torch.from_numpy(seg_smpl) for seg_smpl in seg]
            data_dict["seg"] = seg_ret

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

class RenameTransform(AbstractTransform):
    def __init__(self, key, rename_to):
        self.rename_to = rename_to
        self.key = key

    def __call__(self, **data_dict):
        data_dict[self.rename_to] = data_dict[self.key]
        _ = data_dict.pop(self.key)
        return data_dict