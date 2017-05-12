import numpy as np

def contrast_augmentation_generator(generator, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
    # if contrast_range[0] is <1 and contrast_range[2] is >1 then it will do 50:50 (reduce, increase contrast)
    # preserve_range will cut off values that due to contrast enhancement get over/below the original minimum/maximum of the data
    # per_channel does the contrast enhancement separately for each channel
    for data_dict in generator:
        data = data_dict['data']
        for sample in range(data.shape[0]):
            if not per_channel:
                mn = data[sample].mean()
                if preserve_range:
                    minm = data[sample].min()
                    maxm = data[sample].max()
                if np.random.random() < 0.5 and contrast_range[0] < 1:
                    factor = np.random.uniform(contrast_range[0], 1)
                else:
                    factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
                data[sample] = (data[sample]-mn) * factor + mn
                if preserve_range:
                    data[sample][data[sample]<minm] = minm
                    data[sample][data[sample]>maxm] = maxm
            else:
                for c in range(data[sample].shape[0]):
                    mn = data[sample][c].mean()
                    if preserve_range:
                        minm = data[sample][c].min()
                        maxm = data[sample][c].max()
                    if np.random.random() < 0.5 and contrast_range[0] < 1:
                        factor = np.random.uniform(contrast_range[0], 1)
                    else:
                        factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
                    data[sample][c] = (data[sample][c] - mn) * factor + mn
                    if preserve_range:
                        data[sample][c][data[sample][c] < minm] = minm
                        data[sample][c][data[sample][c] > maxm] = maxm
        data_dict['data'] = data
        yield data_dict