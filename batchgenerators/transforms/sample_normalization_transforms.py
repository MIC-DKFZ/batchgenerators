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


from abstract_transform import AbstractTransform
from batchgenerators.augmentations.normalizations import range_normalization, cut_off_outliers, \
    zero_mean_unit_variance_normalization


class RangeTransform(AbstractTransform):
    '''Rescales data into the specified range

    Args:
        rnge (tuple of float): The range to which the data is scaled

        per_channel (bool): determines whether the min and max values used for the rescaling are computed over the whole
        sample or separately for each channel

    '''
    def __init__(self, rnge=(0, 1), per_channel=True):
        self.per_channel = per_channel
        self.rnge = rnge

    def __call__(self, **data_dict):
        data_dict['data'] = range_normalization(data_dict['data'], self.rnge, per_channel=self.per_channel)
        return data_dict


class CutOffOutliersTransform(AbstractTransform):
    """ Removes outliers from data

    Args:
        percentile_lower (float between 0 and 100): Lower cutoff percentile

        percentile_upper (float between 0 and 100): Upper cutoff percentile

        per_channel (bool): determines whether percentiles are computed for each color channel separately
    """
    def __init__(self, percentile_lower=0.2, percentile_upper=99.8, per_channel=False):
        self.per_channel = per_channel
        self.percentile_upper = percentile_upper
        self.percentile_lower = percentile_lower

    def __call__(self, **data_dict):
        data_dict['data'] = cut_off_outliers(data_dict['data'], self.percentile_lower, self.percentile_upper,
                                             per_channel=self.per_channel)
        return data_dict


class ZeroMeanUnitVarianceTransform(AbstractTransform):
    """ Zero mean unit variance transform

    Args:
        per_channel (bool): determines whether mean and std are computed for and applied to each color channel
        separately

        epsilon (float): prevent nan if std is zero, keep at 1e-7
    """
    def __init__(self, per_channel=True, epsilon=1e-7):
        self.epsilon = epsilon
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        data_dict['data'] = zero_mean_unit_variance_normalization(data_dict["data"], self.per_channel, self.epsilon)
        return data_dict

