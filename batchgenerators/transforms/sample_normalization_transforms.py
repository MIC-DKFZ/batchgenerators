from abstract_transform import AbstractTransform
from batchgenerators.augmentations.normalizations import range_normalization, cut_off_outliers, \
    zero_mean_unit_variance_normalization


class RangeTransform(AbstractTransform):
    def __init__(self, rnge=(0, 1), per_channel=True):
        self.per_channel = per_channel
        self.rnge = rnge

    def __call__(self, **data_dict):
        data_dict['data'] = range_normalization(data_dict['data'], self.rnge, per_channel=self.per_channel)
        return data_dict


class CutOffOutliersTransform(AbstractTransform):
    def __init__(self, percentile_lower=0.2, percentile_upper=99.8, per_channel=False):
        self.per_channel = per_channel
        self.percentile_upper = percentile_upper
        self.percentile_lower = percentile_lower

    def __call__(self, **data_dict):
        data_dict['data'] = cut_off_outliers(data_dict['data'], self.percentile_lower, self.percentile_upper,
                                             per_channel=self.per_channel)
        return data_dict


class ZeroMeanUnitVarianceTransform(AbstractTransform):
    def __init__(self, per_channel=True, epsilon=1e-7):
        self.epsilon = epsilon
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        data_dict['data'] = zero_mean_unit_variance_normalization(data_dict["data"], self.per_channel, self.epsilon)
        return data_dict

