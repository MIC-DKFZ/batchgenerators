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

from typing import Union, Tuple, Callable, Any
import numpy as np

ScalarType = Union[Union[int, float], Tuple[float, float], Callable[[Any, ...], Union[float, int]]]


def sample_scalar(scalar_type: ScalarType, *args):
    if isinstance(scalar_type, (int, float)):
        return scalar_type
    elif isinstance(scalar_type, (list, tuple)):
        assert len(scalar_type) == 2, 'if list is provided, its length must be 2'
        assert scalar_type[0] < scalar_type[1], 'if list is provided, first entry must be smaller than second entry, ' \
                                                'otherwise we cannot sample using np.random.uniform'
        return np.random.uniform(*scalar_type)
    elif callable(scalar_type):
        return scalar_type(*args)
    else:
        raise RuntimeError('Unknown type: %s. Expected: int, float, list, tuple, callable', type(scalar_type))


if __name__ == '__main__':
    sample_scalar(0.5)
    sample_scalar((0, 1))
    sample_scalar(lambda: np.random.uniform(-1, 2))
    sample_scalar(lambda x, y: np.random.uniform(x, y), 0.5, 2)
