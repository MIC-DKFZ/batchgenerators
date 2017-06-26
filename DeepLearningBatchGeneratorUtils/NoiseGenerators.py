import numpy as np
import random

def rician_noise_generator(generator, noise_variance=(0, 0.1)):
    '''
    adds rician noise with the given
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"

        data = data_dict['data']
        shape = data[0].shape
        for sample_idx in range(data.shape[0]):

            sample = data[sample_idx]

            variance = random.uniform(noise_variance[0], noise_variance[1])
            for val in np.nditer(sample, op_flags=['readwrite']):
                val[...] = np.sqrt(np.float_power(val + np.random.normal(0.0, variance), 2) + np.float_power(np.random.normal(0.0, variance), 2))

            data[sample_idx] = sample

        data_dict["data"] = data
        yield data_dict
