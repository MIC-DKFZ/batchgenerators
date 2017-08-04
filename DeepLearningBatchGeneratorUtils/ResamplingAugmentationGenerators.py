import random
import numpy as np
import nibabel as nib

def linear_downsampling_generator(generator, max_downsampling_factor=3, isotropic=False):
    from nilearn.image.resampling import resample_img, resample_to_img
    '''
    downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"

        data = data_dict['data']
        shape = data[0].shape
        for sample_idx in range(data.shape[0]):

            if len(shape)>3 :
                for channel_idx in range(data.shape[1]):
                    affine = np.identity(4)
                    image = nib.Nifti1Image(data[sample_idx,channel_idx], affine)
                    affine2 = affine
                    if isotropic :
                        affine2[0, 0] = random.uniform(1, max_downsampling_factor)
                        affine2[1, 1] = random.uniform(1, max_downsampling_factor)
                        affine2[2, 2] = random.uniform(1, max_downsampling_factor)
                    else:
                        fact = random.uniform(1, max_downsampling_factor)
                        affine2[0, 0] = fact
                        affine2[1, 1] = fact
                        affine2[2, 2] = fact
                    affine2[3, 3] = 1
                    image2 = resample_img(image, target_affine=affine2, interpolation='continuous')
                    image3 = resample_to_img(image2, image, 'nearest')
                    data[sample_idx,channel_idx] = image3.get_data()

            else :
                affine = np.identity(4)
                image = nib.Nifti1Image(data[sample_idx], affine)
                affine2 = affine
                if isotropic :
                    fact = random.uniform(1, max_downsampling_factor)
                    affine2[1, 1] = fact
                    affine2[2, 2] = fact
                else:
                    affine2[1, 1] = random.uniform(1, max_downsampling_factor)
                    affine2[2, 2] = random.uniform(1, max_downsampling_factor)
                affine2[0, 0] = 1
                affine2[3, 3] = 1
                image2 = resample_img(image, target_affine=affine2, interpolation='continuous')
                image3 = resample_to_img(image2, image, 'nearest')
                data[sample_idx] = image3.get_data()

        data_dict["data"] = data
        yield data_dict
