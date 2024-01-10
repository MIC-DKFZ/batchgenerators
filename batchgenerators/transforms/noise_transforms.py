# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
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


from batchgenerators.augmentations.noise_augmentations import augment_blank_square_noise, augment_gaussian_blur, \
    augment_gaussian_noise, augment_rician_noise
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
from typing import Union, Tuple

from scipy.ndimage import median_filter
from scipy.signal import convolve


class RicianNoiseTransform(AbstractTransform):
    """Adds rician noise with the given variance.
    The Noise of MRI data tends to have a rician distribution: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2254141/

    Args:
        noise_variance (tuple of float): samples variance of Gaussian distribution used to calculate
        the rician distribution from this interval

    CAREFUL: This transform will modify the value range of your data!
    """

    def __init__(self, noise_variance=(0, 0.1), data_key="data", label_key="seg", p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.noise_variance = noise_variance

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_rician_noise(data_dict[self.data_key][b],
                                                                   noise_variance=self.noise_variance)
        return data_dict


class GaussianNoiseTransform(AbstractTransform):
    def __init__(self, noise_variance=(0, 0.1), p_per_sample=1, p_per_channel: float = 1,
                 per_channel: bool = False, data_key="data"):
        """
        Adds additive Gaussian Noise

        :param noise_variance: variance is uniformly sampled from that range
        :param p_per_sample:
        :param p_per_channel:
        :param per_channel: if True, each channel will get its own variance sampled from noise_variance
        :param data_key:

        CAREFUL: This transform will modify the value range of your data!
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        mask = np.random.uniform(size=len(data_dict[self.data_key])) < self.p_per_sample
        if np.any(mask):
            data_dict[self.data_key][mask] = augment_gaussian_noise(data_dict[self.data_key][mask], self.noise_variance,
                                                                    self.p_per_channel, self.per_channel, batched=True)
        return data_dict


class GaussianBlurTransform(AbstractTransform):
    def __init__(self, blur_sigma: Tuple[float, float] = (1, 5), different_sigma_per_channel: bool = True,
                 different_sigma_per_axis: bool = False, p_isotropic: float = 0, p_per_channel: float = 1,
                 p_per_sample: float = 1, data_key: str = "data"):
        """

        :param blur_sigma:
        :param data_key:
        :param different_sigma_per_axis: if True, anisotropic kernels are possible
        :param p_isotropic: only applies if different_sigma_per_axis=True, p_isotropic is the proportion of isotropic
        kernels, the rest gets random sigma per axis
        :param different_sigma_per_channel: whether to sample a sigma for each channel or all channels at once
        :param p_per_channel: probability of applying gaussian blur for each channel. Default = 1 (all channels are
        blurred with prob 1)
        """
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.blur_sigma = blur_sigma
        self.different_sigma_per_axis = different_sigma_per_axis
        self.p_isotropic = p_isotropic

    def __call__(self, **data_dict):
        # TODO: Do batched gaussian blur
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gaussian_blur(data_dict[self.data_key][b], self.blur_sigma,
                                                                    self.different_sigma_per_channel,
                                                                    self.p_per_channel,
                                                                    different_sigma_per_axis=self.different_sigma_per_axis,
                                                                    p_isotropic=self.p_isotropic)
        return data_dict


class BlankSquareNoiseTransform(AbstractTransform):
    def __init__(self, squre_size=20, n_squres=1, noise_val=(0, 0), channel_wise_n_val=False, square_pos=None,
                 data_key="data", label_key="seg", p_per_sample=1):

        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.noise_val = noise_val
        self.n_squres = n_squres
        self.squre_size = squre_size
        self.channel_wise_n_val = channel_wise_n_val
        self.square_pos = square_pos

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_blank_square_noise(data_dict[self.data_key][b], self.squre_size,
                                                                         self.n_squres, self.noise_val,
                                                                         self.channel_wise_n_val, self.square_pos)
        return data_dict

class ColorFunctionExtractor:
    def __init__(self, rectangle_value):
        self.rectangle_value = rectangle_value

    def __call__(self, x):
        # TODO: Change this
        if np.isscalar(self.rectangle_value):
            return self.rectangle_value
        elif callable(self.rectangle_value):
            return self.rectangle_value(x)
        elif isinstance(self.rectangle_value, (tuple, list)):
            return np.random.uniform(*self.rectangle_value)
        else:
            raise RuntimeError("unrecognized format for rectangle_value")




class BlankRectangleTransform(AbstractTransform):
    def __init__(self, rectangle_size, rectangle_value, num_rectangles, force_square=False, p_per_sample=0.5,
                 p_per_channel=0.5, apply_to_keys=('data',)):
        """
        Currently under development. This will replace BlankSquareNoiseTransform soon

        Overwrites areas in tensors specified by apply_to_keys with rectangles of some intensity

        This transform supports nD data.

        Note that we say square/rectangle here but we really mean line/square/rectangle/cube/whatevs.

        :param rectangle_size: rectangle size range. Can be either
            - int: creates only squares with edge length rectangle_size
            - tuple/list of int: constant size for rectangles is used. List/Tuple must have the same length as the
              data has dimensions (so len=3 for 3D images)
            - tuple/list of tuple/list: must have the same length as the data has dimensions. internal tuple/list
            specify a range from wich rectangle size will be sampled uniformly, for example: ((5, 10), (7, 12)) will
            generate rectangles between edge length between 5 and 10 for x and 7 and 12 for the y axis.
            - IMPORTANT: if force_square=True then only the first entry of the list will be used. So in the previous
            example rectangle_size=((5, 10), (7, 12)) the (7, 12) entry will be ignored and only squares between edge
            length (5, 10) in all dimensions will be produced
        :param rectangle_value: Intensity value to overwrite the voxels within the square with. Can be int, tuple,
        string, or callable:
            - int: always use the value specified by  rectangle_value
            - tuple: for example (0, 10) uniformly samples intensity values from the given interval. Note that the
            first entry must be smaller than the second! (10, 0) is not valid.
            - callable: we call rectangle_value(x) for each rectangle and you decide what happens (where x is the
            patch to be replaced)
        :param num_rectangles: Specifies the number of rectangles produced per selected image (depends on p_per_sample
        and p_per_channel). Canbe either int or tuple (for example (1, 5)) specifying a range form which the number
        of rectangles is uniformly sampled (note that we use np.random.random_integers, so the upper value is never
        selected (5 in this case). You can give 5.1 or so to make sure 5 gets selected as well)
        :param force_square: If True, only produces squares. In that case, all but the first entry of rectangle_size
        is discarded (also see doc for rectangle_size)
        :param p_per_sample:
        :param p_per_image:
        :param apply_to_keys:
        """
        self.rectangle_size = rectangle_size
        self.num_rectangles = num_rectangles
        self.force_square = force_square
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.apply_to_keys = apply_to_keys
        self.color_fn = ColorFunctionExtractor(rectangle_value)

    def __call__(self, **data_dict):
        for k in self.apply_to_keys:
            workon = data_dict[k]
            img_shape = workon.shape[2:]
            img_dim = len(img_shape)
            for b in range(workon.shape[0]):
                if np.random.uniform(0, 1) < self.p_per_sample:
                    for c in range(workon.shape[1]):
                        if np.random.uniform(0, 1) < self.p_per_channel:
                            # number of rectangles
                            n_rect = self.num_rectangles if isinstance(self.num_rectangles, int) else \
                                np.random.random_integers(*self.num_rectangles)
                            for rect_id in range(n_rect):
                                if isinstance(self.rectangle_size, int):
                                    rectangle_size = [self.rectangle_size for d in img_shape]
                                elif isinstance(self.rectangle_size, (tuple, list)) and \
                                        all([isinstance(i, int) for i in self.rectangle_size]):
                                    rectangle_size = self.rectangle_size
                                elif isinstance(self.rectangle_size, (tuple, list)) and \
                                        all([isinstance(i, (tuple, list)) for i in self.rectangle_size]):
                                    if self.force_square:
                                        rectangle_size = [np.random.random_integers(*self.rectangle_size[0])] * img_dim
                                    else:
                                        rectangle_size = [np.random.random_integers(*self.rectangle_size[d])
                                                          for d in range(img_dim)]
                                else:
                                    raise RuntimeError("unrecognized format for rectangle_size")

                                lb = [np.random.random_integers(img_shape[i] - rectangle_size[i]) for i in
                                      range(img_dim)]
                                ub = [i + j for i, j in zip(lb, rectangle_size)]

                                my_slice = tuple([b, c, *[slice(i, j) for i, j in zip(lb, ub)]])

                                # figure out intensity value
                                intensity = self.color_fn(workon[my_slice])

                                workon[my_slice] = intensity
        return data_dict


class MedianFilterTransform(AbstractTransform):
    def __init__(self,
                 filter_size: Union[int, Tuple[int, int]],
                 same_for_each_channel: bool = False,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 data_key='data'
                 ):
        """

        :param filter_size:
        :param same_for_each_channel:
        :param p_per_sample:
        :param p_per_channel:
        :param data_key:
        """
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.filter_size = filter_size
        self.same_for_each_channel = same_for_each_channel

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        assert data is not None
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                if self.same_for_each_channel:
                    filter_size = self.filter_size if isinstance(self.filter_size, int) else np.random.randint(*self.filter_size)
                    for c in range(data.shape[1]):
                        if np.random.uniform() < self.p_per_channel:
                            data[b, c] = median_filter(data[b, c], filter_size)
                else:
                    for c in range(data.shape[1]):
                        if np.random.uniform() < self.p_per_channel:
                            filter_size = self.filter_size if isinstance(self.filter_size, int) else np.random.randint(*self.filter_size)
                            data[b, c] = median_filter(data[b, c], filter_size)
        return data_dict


class SharpeningTransform(AbstractTransform):
    filter_2d = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])
    filter_3d = np.array([[[0, 0, 0],
                           [0, -1, 0],
                           [0, 0, 0]],
                          [[0, -1, 0],
                           [-1, 6, -1],
                           [0, -1, 0]],
                          [[0, 0, 0],
                           [0, -1, 0],
                           [0, 0, 0]],
                          ])

    def __init__(self,
                 strength: Union[float, Tuple[float, float]] = 0.2,
                 same_for_each_channel: bool = False,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 data_key='data'):
        """
        :param strength:
        :param same_for_each_channel:
        :param p_per_sample:
        :param p_per_channel:
        :param data_key:
        """
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.strength = strength
        self.same_for_each_channel = same_for_each_channel

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        assert data is not None
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                if self.same_for_each_channel:
                    mn, mx = data[b].min(), data[b].max()
                    strength_here = self.strength if isinstance(self.strength, float) else np.random.uniform(
                        *self.strength)
                    if data.ndim == 4:
                        filter_here = self.filter_2d * strength_here
                        filter_here[1, 1] += 1
                    else:
                        filter_here = self.filter_3d * strength_here
                        filter_here[1, 1, 1] += 1
                    for c in range(data.shape[1]):
                        if np.random.uniform() < self.p_per_channel:
                            data[b, c] = convolve(data[b, c],
                                                  filter_here,
                                                  mode='same'
                                                  )
                            np.clip(data[b, c], mn, mx, out=data[b, c])
                else:
                    for c in range(data.shape[1]):
                        if np.random.uniform() < self.p_per_channel:
                            mn, mx = data[b, c].min(), data[b, c].max()
                            strength_here = self.strength if isinstance(self.strength, float) else np.random.uniform(
                                *self.strength)
                            if data.ndim == 4:
                                filter_here = self.filter_2d * strength_here
                                filter_here[1, 1] += 1
                            else:
                                filter_here = self.filter_3d * strength_here
                                filter_here[1, 1, 1] += 1
                            data[b, c] = convolve(data[b, c],
                                                  filter_here,
                                                  mode='same'
                                                  )
                            np.clip(data[b, c], mn, mx, out=data[b, c])
        return data_dict


if __name__ == '__main__':
    from copy import deepcopy
    from skimage.data import camera

    # just some playing around with BrightnessGradientAdditiveTransform
    data = {'data': np.vstack((camera()[None], camera()[None], camera()[None]))[None, None].astype(np.float32)}
    tr = MedianFilterTransform((1, 20), True)
    transformed = tr(**deepcopy(data))['data']
    from batchviewer import view_batch

    view_batch(*data['data'][0], *transformed[0])
