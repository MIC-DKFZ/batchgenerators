import random
import numpy as np


def linear_downsampling_generator(generator, max_downsampling_factor=2, isotropic=False):
    '''
    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor).

    Info:
    * Uses nilearn resample_img for resampling.
    * If isotropic=True:  Resamples all dimensions (channels, x, y, z) with same downsampling factor
    * If isotropic=False: Randomly choose new downsampling factor for each dimension
    * Does not resample "seg".
    '''
    import nibabel as nib
    from nilearn.image.resampling import resample_img, resample_to_img

    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"

        data = data_dict['data']    #shape of data must be: (batch_size, nr_of_channels, x, y, [z])  (z ist optional; nr_of_channels can be 1)
        dim = len(data.shape[2:])        #remove batch_size and nr_of_channels dimension
        for sample_idx in range(data.shape[0]):

            fact = random.uniform(1, max_downsampling_factor)

            for channel_idx in range(data.shape[1]):

                affine = np.identity(4)
                if dim == 3:
                    img_data = data[sample_idx, channel_idx]
                elif dim == 2:
                    tmp = data[sample_idx, channel_idx]
                    img_data = np.reshape(tmp, (1, tmp.shape[0], tmp.shape[1]))   #add third spatial dimension to make resample_img work
                else:
                    raise ValueError("Invalid dimension size")

                image = nib.Nifti1Image(img_data, affine)
                affine2 = affine
                if isotropic :
                    affine2[0, 0] = fact
                    affine2[1, 1] = fact
                    affine2[2, 2] = fact
                else:
                    affine2[0, 0] = random.uniform(1, max_downsampling_factor)
                    affine2[1, 1] = random.uniform(1, max_downsampling_factor)
                    affine2[2, 2] = random.uniform(1, max_downsampling_factor)
                affine2[3, 3] = 1
                image2 = resample_img(image, target_affine=affine2, interpolation='continuous')
                image3 = resample_to_img(image2, image, 'nearest')
                data[sample_idx,channel_idx] = np.squeeze(image3.get_data())

        data_dict["data"] = data
        yield data_dict


def linear_downsampling_generator_scipy(generator, zoom_range=(0.5, 1)):
    '''
    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from linear_downsampling_generator)
    * Does not resample "seg".
    '''
    import scipy.ndimage

    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"

        data = data_dict['data']    # shape of data must be: (batch_size, nr_of_channels, x, y, [z])  (z ist optional; nr_of_channels can be 1)
        dim = len(data.shape[2:])   # remove batch_size and nr_of_channels dimension

        for sample_idx in range(data.shape[0]):

            zoom = round(random.uniform(zoom_range[0], zoom_range[1]), 2)

            for channel_idx in range(data.shape[1]):
                img = data[sample_idx, channel_idx]
                img_down = scipy.ndimage.zoom(img, zoom, order=1)
                zoom_reverse = round(1. / zoom, 2)
                img_up = scipy.ndimage.zoom(img_down, zoom_reverse, order=0)

                if dim == 3:
                    # cut if dimension got too long
                    img_up = img_up[:img.shape[0], :img.shape[1], :img.shape[2]]

                    # pad with 0 if dimension too small
                    img_padded = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
                    img_padded[:img_up.shape[0],:img_up.shape[1],:img_up.shape[2]] = img_up

                    data[sample_idx,channel_idx] = img_padded

                elif dim == 2:
                    # cut if dimension got too long
                    img_up = img_up[:img.shape[0], :img.shape[1]]

                    # pad with 0 if dimension too small
                    img_padded = np.zeros((img.shape[0], img.shape[1]))
                    img_padded[:img_up.shape[0], :img_up.shape[1]] = img_up

                    data[sample_idx, channel_idx] = img_padded
                else:
                    raise ValueError("Invalid dimension size")

        data_dict["data"] = data
        yield data_dict




