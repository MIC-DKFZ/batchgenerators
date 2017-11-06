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


from warnings import warn
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class DataChannelSelectionTransform(AbstractTransform):
    """Selects color channels from the batch and discards the others.

    Args:
        channels (list of int): List of channels to be kept.

    """
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, **data_dict):
        data_dict["data"] = data_dict["data"][:, self.channels]
        return data_dict


class SegChannelSelectionTransform(AbstractTransform):
    """Segmentations may have more than one channel. This transform selects segmentation channels

    Args:
        channels (list of int): List of channels to be kept.

    """
    def __init__(self, channels, keep_discarded_seg=False):
        self.channels = channels
        self.keep_discarded = keep_discarded_seg

    def __call__(self, **data_dict):
        seg = data_dict.get("seg")

        if seg is None:
            warn("You used SegChannelSelectionTransform but there is no 'seg' key in your data_dict, returning data_dict unmodified", Warning)
        else:
            if self.keep_discarded:
                discarded_seg_idx = [i for i in range(len(seg[0])) if i not in self.channels]
                data_dict['discarded_seg'] = seg[:, discarded_seg_idx]
            data_dict["seg"] = seg[:, self.channels]
        return data_dict
