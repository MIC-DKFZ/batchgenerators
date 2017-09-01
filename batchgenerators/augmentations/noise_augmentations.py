from builtins import range

import numpy as np
import random


def augment_rician_noise(data, noise_variance=(0, 0.1)):

    ret_data = []
    for sample_idx in range(data.shape[0]):
        sample = data[sample_idx]
        variance = random.uniform(noise_variance[0], noise_variance[1])
        sample = np.sqrt((sample + np.random.normal(0.0, variance, size=sample.shape)) ** 2 +
                         np.random.normal(0.0, variance, size=sample.shape) ** 2)
        ret_data.append(sample)

    return ret_data


def augment_gaussian_noise(data, noise_variance=(0, 0.1)):

    ret_data = []
    for sample_idx in range(data.shape[0]):
        sample = data[sample_idx]
        variance = random.uniform(noise_variance[0], noise_variance[1])
        sample = sample + np.random.normal(0.0, variance, size=sample.shape)
        ret_data.append(sample)

    return ret_data
