import torch
from abstract_transform import AbstractTransform
from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding
import numpy as np

class DictToTensor(AbstractTransform):

    def __call__(self, **data_dict):

        data = data_dict.get("data")
        seg = data_dict.get("seg")

        data_dict["data"] = torch.from_numpy(data)
        if seg is not None:
            data_dict["seg"] = torch.from_numpy(seg)

        return data_dict


class ConvertSegToOnehotTransform(AbstractTransform):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, **data_dict):
        seg = data_dict.get("seg")
        if seg is not None:
            assert seg.shape[1] == 1, "only supports one segmentation channel"
            new_seg = np.zeros([seg.shape[0], len(self.classes)] + list(seg.shape[2:]), dtype=seg.dtype)
            for b in range(seg.shape[0]):
                new_seg[b] = convert_seg_image_to_one_hot_encoding(seg[b, 0], self.classes)
            seg = new_seg
        else:
            from warnings import warn
            warn("calling ConvertSegToOnehotTransform but there is no segmentation")
        return seg