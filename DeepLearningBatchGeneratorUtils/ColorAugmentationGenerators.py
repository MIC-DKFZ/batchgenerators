import numpy as np
from utils import general_cc_var_num_channels, illumination_jitter

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


def brightness_augmentation_generator(generator, mu, sigma, per_channel=True):
    # adds a randomly sampled (gaussian with mu and sigma) offset
    # this is done separately for each channel if per_channel is set to True
    for data_dict in generator:
        data = data_dict['data']
        for sample in range(data.shape[0]):
            tmp = np.array(data[sample])
            brain_mask = data_dict['seg'][sample, 1:]!=0
            brain_mask = np.vstack([brain_mask] * len(tmp))
            if not per_channel:
                rnd_nb = np.random.normal(mu, sigma)
                tmp[brain_mask] += rnd_nb
            else:
                for c in range(tmp.shape[0]):
                    rnd_nb = np.random.normal(mu, sigma)
                    tmp[c][brain_mask[c]] += rnd_nb
            tmp[tmp < 0] = 0
            tmp[tmp > 1] = 1
            data[sample] = tmp
        yield data_dict


def gamma_augmentation_generator(generator, gamma_range=(0.5, 2), invert_image=False):
    # augments by shifting the gamma value as in gamma correction (https://en.wikipedia.org/wiki/Gamma_correction)
    for data_dict in generator:
        data = data_dict['data']
        for sample in range(data.shape[0]):
            if invert_image:
                data = - data
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data[sample].min()
            rnge = data[sample].max() - minm
            data[sample] = np.power(((data[sample]-minm)/float(rnge)), gamma) * rnge + minm
        data_dict['data'] = data
        yield data_dict


def illumintaion_augmentation_generator(generator, white_rgb):
    '''This generator implements illumination color augmentation. The idea here is that we can estimate the
    illumination of all training samples and then transfer that to other training samples, thereby augmenting the data.
    In order for this to work properly, your images have to be on the range of [0, 255] and they must be natural images
    (meaning for instance that one of the color channels does not exploit the whole range of [0, 255]. This technique
    does not work properly with images that have to be manually rescaled to this range (such as MRI images) due to the
    way the color constancy method works.
    white_rgb: list of illuminations to choose from for the augmentation. Can be generated using the white_colors
    returned by utils.general_cc_var_num_channels'''
    for data_dict in generator:
        data = data_dict['data']
        idx = np.random.choice(len(white_rgb), data.shape[0])
        for sample in range(data.shape[0]):
            _,img = general_cc_var_num_channels(data[sample], 0, 5, 0, None, 1., 7, False)
            rgb = np.array(white_rgb[idx[sample]]) * np.sqrt(3)
            for c in range(data[sample].shape[0]):
                data[sample, c] = img[c] * rgb[c]
            # data[sample][data[sample] < 0] = 0
            # data[sample][data[sample] > 1] = 1
        data_dict['data'] = data
        yield data_dict

def fancy_color_augmentation_generator(generator, U, s, sigma=0.2):
    '''Implements the fancy color augmentation used in AlexNet. u is the matrix of eigenvalues of your training data,
    s are the eigenvectors. Augmentation is done by sampling one random number rnd (gaussian distribution with  parameter
    sigma) for each eigenvalue (vector of random numbers is denoted r) and then adding Ux(s*r) to each pixel of the
    image (x is matrix vector multiplication, * is elementwise multiplication)'''
    for data_dict in generator:
        data = data_dict['data']
        for sample in range(data.shape[0]):
            data[sample] = illumination_jitter(data[sample], U, s, sigma)
            data[sample] -= data[sample].min()
            data[sample] /= data[sample].max()
        data_dict['data'] = data
        yield data_dict

