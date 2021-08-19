from typing import Union, Tuple, Callable
import numpy as np

ScalarType = Union[Union[int, float], Tuple[float, float], Callable[[], float]]


def sample_scalar(scalar_type: ScalarType):
    if isinstance(scalar_type, (int, float)):
        return scalar_type
    elif isinstance(scalar_type, (list, tuple)):
        assert len(scalar_type) == 2, 'if list is provided, its length must be 2'
        assert scalar_type[0] < scalar_type[1], 'if list is provided, first entry must be smaller than second entry, ' \
                                                'otherwise we cannot sample using np.random.uniform'
        return np.random.uniform(scalar_type)
    elif callable(scalar_type):
        return scalar_type()
    else:
        raise RuntimeError('Unknown type: %s. Expected: int, float, list, tuple, callable', type(scalar_type))
