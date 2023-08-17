# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
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

import numpy as np

from batchgenerators.augmentations.utils import get_broadcast_axes, reverse_broadcast


def range_normalization(data, rnge=(0, 1), per_channel=True, eps=1e-8):
    if per_channel:
        axes = tuple(range(2, len(data.shape)))
    else:
        axes = tuple(range(1, len(data.shape)))

    data_normalized = min_max_normalization_batched(data, eps, axes)
    data_normalized *= (rnge[1] - rnge[0])
    data_normalized += rnge[0]
    return data_normalized


def min_max_normalization_batched(data, eps, axes):
    mn = data.min(axis=axes)
    mx = data.max(axis=axes)
    old_range = mx - mn + eps

    data_normalized = ((data.T - mn.T) / old_range.T).T
    # broadcast_axes = get_broadcast_axes(len(data.shape))
    # mn = reverse_broadcast(mn, broadcast_axes)
    # old_range = reverse_broadcast(old_range, broadcast_axes)
    # data_normalized = (data - mn) / old_range
    return data_normalized


def min_max_normalization(data, eps):
    mn = data.min()
    mx = data.max()
    old_range = mx - mn + eps
    data_normalized = (data - mn) / old_range
    return data_normalized


def zero_mean_unit_variance_normalization(data, per_channel=True, epsilon=1e-8):
    if per_channel:
        axes = tuple(range(2, len(data.shape)))
    else:
        axes = tuple(range(1, len(data.shape)))

    mean = np.mean(data, axis=axes)
    std = np.std(data, axis=axes) + epsilon
    data_normalized = ((data.T - mean.T) / std.T).T
    return data_normalized


def mean_std_normalization(data, mean, std, per_channel=True):
    if per_channel:
        channel_dimension = data[0].shape[0]
        if isinstance(mean, float) and isinstance(std, float):
            mean = (mean,) * channel_dimension
            std = (std,) * channel_dimension
        else:
            assert len(mean) == channel_dimension
            assert len(std) == channel_dimension

        broadcast_axes = tuple(range(2, len(data.shape)))
        mean = np.expand_dims(np.broadcast_to(mean, (len(data), len(mean))), axis=broadcast_axes)
        std = np.expand_dims(np.broadcast_to(std, (len(data), len(std))), axis=broadcast_axes)

    data_normalized = (data - mean) / std
    return data_normalized


def cut_off_outliers(data, percentile_lower=0.2, percentile_upper=99.8, per_channel=False):
    if per_channel:
        axes = tuple(range(2, len(data.shape)))
    else:
        axes = tuple(range(1, len(data.shape)))

    cut_off_lower, cut_off_upper = np.percentile(data, (percentile_lower, percentile_upper), axis=axes)
    cut_off_lower = np.expand_dims(cut_off_lower, axis=axes)
    cut_off_upper = np.expand_dims(cut_off_upper, axis=axes)
    np.clip(data, cut_off_lower, cut_off_upper, out=data)
    return data
