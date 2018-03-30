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


class ResampleTransform(AbstractTransform):
    '''Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from linear_downsampling_generator_nilearn)

    Args:
        zoom_range (tuple of float): Random downscaling factor in this range. (e.g.: 0.5 halfs the resolution)
    '''

    def __init__(self, zoom_range=(0.5, 1), data_key="data"):
        self.data_key = data_key
        self.zoom_range = zoom_range

    def __call__(self, **data_dict):
        data_dict[self.data_key] = augment_linear_downsampling_scipy(data_dict[self.data_key], zoom_range=self.zoom_range)
        return data_dict


class SimulateLowResTransform(AbstractTransform):
    def __init__(self, sampling_range_per_axes=None, order_down=0, order_up=1, per_channel=True, channels=None):
        """
        Downsamples along specified axes with factor randomly drawn from rampling_range. Factor 5 hereby means that the output of
        that axis is downsampled to a fifth of the original size and then sampled back up again. You can control the order
        of interpolation by using order_down and _up for down/upsampling separately. per_chanel=True will randomly select
        a downsampling separately for each color channel. If you would like to augment only specific color channels,
        you can specify to do so via channels.
        :param data: np array: b x c x x x y( x z)
        :param sampling_range_per_axes: dict with key:value axes:(tuple) as key_value pairs. Axes is as they are in the batch (2, 3(, 4) for spatial dimensions)
        :param order_down: int
        :param order_up: int
        :param per_channel: bool
        :return:
        """
        self.channels = channels
        self.per_channel = per_channel
        self.order_up = order_up
        self.order_down = order_down
        self.sampling_range_per_axes = sampling_range_per_axes

    def __call__(self, **data_dict):
        data = data_dict.get('data')
        data = augment_downsampling_upsampling(data, self.sampling_range_per_axes, self.order_down, self.order_up,
                                               self.per_channel, self.channels)
        data_dict['data'] = data
        return data_dict
