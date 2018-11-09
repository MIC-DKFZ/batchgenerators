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
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
from batchgenerators.augmentations.resample_augmentations import augment_downsampling_upsampling
import numpy as np


class ResampleTransform(AbstractTransform):
    """Downsamples each sample (linearly) by a random factor and upsamples to original resolution again
    (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel:

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)
    """

    def __init__(self, zoom_range=(0.5, 1), data_key="data", per_channel=False, p_per_sample=1, p_per_channel=1,
                 channels=None):
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.zoom_range = zoom_range

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key] = augment_linear_downsampling_scipy(data_dict[self.data_key],
                                                                             zoom_range=self.zoom_range,
                                                                             per_channel=self.per_channel,
                                                                             p_per_channel=self.p_per_channel,
                                                                             channels=self.channels)
        return data_dict

