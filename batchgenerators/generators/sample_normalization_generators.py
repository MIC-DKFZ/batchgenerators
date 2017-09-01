from builtins import range

import numpy as np


def zero_one_normalization_generator(generator):
    '''
    normalizes each sample to zero mean and std one
    '''
    for data_dict in generator:
        assert "data" in list(
            data_dict.keys()), "your data generator needs to return a python dictionary with at least a 'data' key value pair"

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


def normalize_data_generator(gen):
    '''
    normalizes all data to zero mean unis variance (done separately for each channel in each training instance)
    :param gen:
    :return:
    '''
    for data_dict in gen:
        for b in range(data_dict['data'].shape[0]):
            for c in range(data_dict['data'][b].shape[0]):
                mn = data_dict['data'][b][c].mean()
                sd = data_dict['data'][b][c].std()
                if sd == 0:
                    sd = 1.
                data_dict['data'][b][c] = (data_dict['data'][b][c] - mn) / sd
        yield data_dict


def cut_off_outliers_generator(generator, percentile_lower=0.2, percentile_upper=99.8):
    for data_dict in generator:
        for b in range(data_dict['data'].shape[0]):
            for c in range(data_dict['data'][b].shape[0]):
                img = data_dict['data'][b][c].ravel()
                cut_off_lower = np.percentile(img, percentile_lower)
                cut_off_upper = np.percentile(img, percentile_upper)
                data_dict['data'][b][c][data_dict['data'][b][c] < cut_off_lower] = cut_off_lower
                data_dict['data'][b][c][data_dict['data'][b][c] > cut_off_upper] = cut_off_upper
        yield data_dict
