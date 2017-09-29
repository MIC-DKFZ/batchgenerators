import numpy as np

def range_normalization(data, rnge=(0, 1), per_channel=True):
    for b in data.shape[0]:
        if per_channel:
            for c in data.shape[1]:
                mn = data[b, c].min()
                mx = data[b, c].max()
                data[b, c] -= mn
                data[b, c] /= (mx - mn)
                data[b, c] *= (rnge[1] - rnge[0])
                data[b, c] += rnge[0]
        else:
            mn = data[b].min()
            mx = data[b].max()
            data[b] -= mn
            data[b] /= (mx - mn)
            data[b] *= (rnge[1] - rnge[0])
            data[b] += rnge[0]
    return data


def zero_mean_unit_variance_normalization(data, per_channel=True, epsilon=1e-7):
    for b in data.shape[0]:
        if per_channel:
            for c in data.shape[1]:
                mean = data[b, c].mean()
                std = data[b, c].std() + epsilon
                data[b, c] = (data[b, c] - mean) / std
        else:
            mean = data[b].mean()
            std = data[b].std() + epsilon
            data[b] = (data[b] - mean) / std
    return data


def cut_off_outliers(data, percentile_lower=0.2, percentile_upper=99.8, per_channel=False):
    for b in range(len(data)):
        if not per_channel:
            cut_off_lower = np.percentile(data[b], percentile_lower)
            cut_off_upper = np.percentile(data[b], percentile_upper)
            data[b][data[b] < cut_off_lower] = cut_off_lower
            data[b][data[b] > cut_off_upper] = cut_off_upper
        else:
            for c in range(data.shape[1]):
                cut_off_lower = np.percentile(data[b, c], percentile_lower)
                cut_off_upper = np.percentile(data[b, c], percentile_upper)
                data[b, c][data[b, c] < cut_off_lower] = cut_off_lower
                data[b, c][data[b, c] > cut_off_upper] = cut_off_upper
    return data