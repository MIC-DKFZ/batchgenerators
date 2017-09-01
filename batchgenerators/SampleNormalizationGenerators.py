import numpy as np
import random


def zero_one_normalization_generator(generator):
    '''
    normalizes each sample to zero mean and std one
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"

        data = data_dict['data']
        shape = data[0].shape
        for sample_idx in range(data.shape[0]):

            mean = np.mean(data[sample_idx])
            std = np.std(data[sample_idx])
            data[sample_idx] -= mean
            if std > 0.0001:
                data[sample_idx] /= std

        data_dict["data"] = data
        yield data_dict
