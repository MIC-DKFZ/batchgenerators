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

class ResampleTransform(AbstractTransform):
    '''Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from linear_downsampling_generator_nilearn)

    Args:
        zoom_range (tuple of float): Random downscaling factor in this range. (e.g.: 0.5 halfs the resolution)
    '''

    def __init__(self, zoom_range=(0.5, 1)):
        self.zoom_range = zoom_range

    def __call__(self, **data_dict):
        data_dict['data'] = augment_linear_downsampling_scipy(data_dict['data'], zoom_range=self.zoom_range)
        return data_dict


