# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from builtins import range
import numpy as np
import random
import scipy.ndimage

def augment_linear_downsampling_nilearn(data, max_downsampling_factor=2, isotropic=False):
    '''
    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor).

    Info:
    * Uses nilearn resample_img for resampling.
    * If isotropic=True:  Resamples all channels (channels, x, y, z) with same downsampling factor
    * If isotropic=False: Randomly choose new downsampling factor for each dimension
    '''
    import nibabel as nib
    from nilearn.image.resampling import resample_img, resample_to_img

    dim = len(data.shape[2:])  # remove batch_size and nr_of_channels dimension
    for sample_idx in range(data.shape[0]):

        fact = random.uniform(1, max_downsampling_factor)

        for channel_idx in range(data.shape[1]):

            affine = np.identity(4)
            if dim == 3:
                img_data = data[sample_idx, channel_idx]
            elif dim == 2:
                tmp = data[sample_idx, channel_idx]
                img_data = np.reshape(tmp, (
                1, tmp.shape[0], tmp.shape[1]))  # add third spatial dimension to make resample_img work
            else:
                raise ValueError("Invalid dimension size")

            image = nib.Nifti1Image(img_data, affine)
            affine2 = affine
            if isotropic:
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

        data[sample_idx, channel_idx] = np.squeeze(image3.get_data())
    return data


def augment_linear_downsampling_scipy(data, zoom_range=(0.5, 1)):
    '''
    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling. A bit faster than nilearn.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from linear_downsampling_generator_nilearn)
    '''
    zoom_range = list(zoom_range)
    zoom_range[1] += + 1e-6
    if zoom_range[0] >= zoom_range[1]:
        raise ValueError("First value of zoom_range must be smaller than second value.")

    dim = len(data.shape[2:])  # remove batch_size and nr_of_channels dimension
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
                img_padded[:img_up.shape[0], :img_up.shape[1], :img_up.shape[2]] = img_up

                data[sample_idx, channel_idx] = img_padded

            elif dim == 2:
                # cut if dimension got too long
                img_up = img_up[:img.shape[0], :img.shape[1]]

                # pad with 0 if dimension too small
                img_padded = np.zeros((img.shape[0], img.shape[1]))
                img_padded[:img_up.shape[0], :img_up.shape[1]] = img_up

                data[sample_idx, channel_idx] = img_padded
            else:
                raise ValueError("Invalid dimension size")

    return data