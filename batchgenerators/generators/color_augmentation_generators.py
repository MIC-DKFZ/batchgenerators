from batchgenerators.augmentations.color_augmentations import (augment_contrast,
augment_brightness_additive, augment_brightness_multiplicative, augment_gamma, augment_illumination, augment_PCA_shift)


def contrast_augmentation_generator(generator, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
    # if contrast_range[0] is <1 and contrast_range[2] is >1 then it will do 50:50 (reduce, increase contrast)
    # preserve_range will cut off values that due to contrast enhancement get over/below the original minimum/maximum of the data
    # per_channel does the contrast enhancement separately for each channel
    for data_dict in generator:
        data_dict['data'] = augment_contrast(data_dict['data'], contrast_range=contrast_range, preserve_range=preserve_range, per_channel=per_channel)
        yield data_dict



def brightness_augmentation_generator(generator, mu, sigma, per_channel=True):
    '''
    Adds a randomly sampled offset (gaussian with mean mu and std sigma).
    This is done separately for each channel if per_channel is set to True.
    '''
    print "Warning (for Fabian): This should no longer be used for brain tumor segmentation (brain mask support dropped)"
    for data_dict in generator:
        data_dict['data'] = augment_brightness_additive(data_dict['data'], mu, sigma, per_channel)
        yield data_dict



def brightness_augmentation_by_multiplication_generator(generator, multiplier_range=(0.5,2), per_channel=True):
    '''
    Multiplies each voxel with a randomly sampled multiplier.
    This is done separately for each channel if per_channel is set to True.
    '''
    for data_dict in generator:
        data_dict['data'] = augment_brightness_multiplicative(data_dict['data'], multiplier_range, per_channel)
        yield data_dict




def gamma_augmentation_generator(generator, gamma_range=(0.5, 2), invert_image=False):
    # augments by shifting the gamma value as in gamma correction (https://en.wikipedia.org/wiki/Gamma_correction)
    for data_dict in generator:
        data_dict['data'] = augment_gamma(data_dict['data'], gamma_range, invert_image)
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
        data_dict['data'] = augment_illumination(data_dict['data'], white_rgb)
        yield data_dict



def fancy_color_augmentation_generator(generator, U, s, sigma=0.2):
    '''Implements the fancy color augmentation used in AlexNet. U is the matrix of eigenvalues of your training data,
    s are the eigenvectors. Augmentation is done by sampling one random number rnd (gaussian distribution with  parameter
    sigma) for each eigenvalue (vector of random numbers is denoted r) and then adding Ux(s*r) to each pixel of the
    image (x is matrix vector multiplication, * is elementwise multiplication)'''
    for data_dict in generator:
        data_dict['data'] = augment_PCA_shift(data_dict['data'], U, s, sigma)
        yield data_dict

