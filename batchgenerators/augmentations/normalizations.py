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
    if isinstance(data, np.ndarray):
        data_shape = data.shape
    elif isinstance(data, (list, tuple)):
        assert len(data) > 0 and isinstance(data[0], np.ndarray)
        data_shape = (len(data),) + data[0].shape
    else:
        raise TypeError("Data has to be either a numpy array or a list")

    if per_channel and isinstance(mean, float) and isinstance(std, float):
        mean = [mean] * data_shape[1]
        std = [std] * data_shape[1]
    elif per_channel and isinstance(mean, (tuple, list, np.ndarray)):
        assert len(mean) == data_shape[1]
    elif per_channel and isinstance(std, (tuple, list, np.ndarray)):
        assert len(std) == data_shape[1]

    if per_channel:
        mean = np.broadcast_to(mean, (len(data), len(mean)))
        std = np.broadcast_to(std, (len(data), len(std)))
        data_normalized = ((data.T - mean.T) / std.T).T
    else:
        data_normalized = (data - mean) / std
    return data_normalized


def cut_off_outliers(data, percentile_lower=0.2, percentile_upper=99.8, per_channel=False):
    if per_channel:
        axes = tuple(range(2, len(data.shape)))
    else:
        axes = tuple(range(1, len(data.shape)))

    cut_off_lower, cut_off_upper = np.percentile(data, (percentile_lower, percentile_upper), axis=axes)
    np.clip(data.T, cut_off_lower.T, cut_off_upper.T, out=data.T)
    return data
