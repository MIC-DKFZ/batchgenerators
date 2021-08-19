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

from abc import ABC
from typing import Tuple

import numpy as np
import scipy.stats as st
from batchgenerators.utilities.custom_types import ScalarType, sample_scalar
from scipy.ndimage import gaussian_filter


class LocalTransform(ABC):
    def __init__(self,
                 scale: ScalarType,
                 loc: ScalarType = (-1, 2),
                 ):
        """
        Places a Gaussian in the image. This can be used to apply a variety of effects through creating a modified
        copy of the image (for example a smoothed copy) and then linearly interpolating between original and modified
        images with the Gaussian providing the weights
        """
        self.loc = loc
        self.scale = scale

    def _generate_kernel(self, img_shp: Tuple[int, ...]) -> np.ndarray:
        """
        returns an image of shape img_shp in which a Gaussian kernel is placed. The values of the kernel are
        normalized to be between 0 and 1

        kernel is 0 outside and 1 inside the Gaussian
        """
        assert len(img_shp) <= 3
        kernels = []
        for d in range(len(img_shp)):
            image_size_here = img_shp[d]
            loc = sample_scalar(self.loc, img_shp, d)
            scale = sample_scalar(self.scale, img_shp, d)

            loc_rescaled = loc * image_size_here
            x = np.arange(-0.5, image_size_here + 0.5)
            kernels.append(np.diff(st.norm.cdf(x, loc=loc_rescaled, scale=scale)))

        kernel_2d = kernels[0][:, None].dot(kernels[1][None])
        if len(kernels) > 2:
            # trial and error got me here lol
            kernel_image = kernel_2d[:, :, None].dot(kernels[2][None])
        else:
            kernel_image = kernel_2d

        # normalize to [0, 1]
        kernel_image = kernel_image - kernel_image.min()
        kernel_image = kernel_image / max(1e-8, kernel_image.max())
        return kernel_image

    def _generate_multiple_kernel_image(self, img_shp: Tuple[int, ...], num_kernels: int) -> np.ndarray:
        kernel_image = np.zeros(img_shp, dtype=np.float32)
        for n in range(num_kernels):
            kernel_image += self._generate_kernel(img_shp)
        # normalize to [0, 1]
        kernel_image = kernel_image - kernel_image.min()
        kernel_image = kernel_image / max(1e-8, kernel_image.max())
        return kernel_image

    @staticmethod
    def invert_kernel(kernel_image: np.ndarray) -> np.ndarray:
        """
        this function assumes that the kernel image is still in the range of [0, 1]!
        """
        return 1 - kernel_image

    @staticmethod
    def run_interpolation(original_image: np.ndarray, modified_image: np.ndarray, kernel_image: np.ndarray) -> np.ndarray:
        """
        this function assumes that the kernel image is still in the range of [0, 1]!
        Low values in kernel image mean original image is kept, high values mean that modified image is kept
        """
        return original_image * (1 - kernel_image) + modified_image * kernel_image


class BrightnessGradientAdditiveTransform(LocalTransform):
    def __init__(self,
                 scale: ScalarType,
                 loc: ScalarType = (-1, 2),
                 max_strength: ScalarType = 1.,
                 same_for_all_channels: bool = True,
                 mean_centered: bool = True,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 clip_intensities: bool = False,
                 data_key: str = "data"):
        """
        Applies an additive intensity gradient to the image. The intensity gradient is zero-centered (sum(add) = 0;
        will not shift the global mean of the image. Some pixels will be brighter, some darker after application)

        The gradient is implemented by placing a Gaussian distribution with sigma=scale somewhere in the image. The
        location of the kernel is selected independently for each image dimension. The location is encoded in % of the
        image size. The default value of (-1, 2) means that the location will be sampled uniformly from
        (-image.shape[i], 2* image.shape[i]). It is important to allow the center of the kernel to be outside of the image.

        IMPORTANT: Try this with different parametrizations and visualize the outcome to get a better feeling for how
        to use this!

        :param scale: scale of the gradient. Large values recommended!
            float: fixed value
            (float, float): will be sampled independently for each dimension from the interval [scale[0], scale[1]]
            callable: you get all the freedom you want. Will be called as scale(image.shape, dimension) where dimension
            is the index in image.shape we are requesting the scale for. Must return scalar (float).
        :param loc:
            (float, float): sample location uniformly from interval [scale[0], scale[1]] (see main description)
            callable: you get all the freedom you want. Will be called as loc(image.shape, dimension) where dimension
            is the index in image.shape we are requesting the location for. Must return a scalar value denoting a relative
            position along axis dimension (0 for index 0, 1 for image.shape[dimension]. Values beyond 0 and 1 are
            possible and even recommended)
        :param max_strength: scaling of the intensity gradient. Determines what max(abs(add_gauss)) is going to be
            float: fixed value
            (float, float): sampled from [max_strength[0], max_strength[1]]
            callable: you decide. Will be called as max_strength(image, gauss_add). Do not modify gauss_add.
            Must return a scalar.
        :param same_for_all_channels: If True, then the same gradient will be applied to all selected color
        channels of a sample (see p_per_channel). If False, each selected channel obtains its own random gradient.
        :param mean_centered: if True, the brightness addition will be done such that the mean intensity of the image
        does not change. So if a bright spot is added, other parts of the image will have something subtracted to keep
        the mean intensity the same as it was before
        :param p_per_sample:
        :param p_per_channel:
        :param clip_intensities:
        :param data_key:
        """
        super().__init__(scale, loc)
        self.max_strength = max_strength
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.same_for_all_channels = same_for_all_channels
        self.mean_centered = mean_centered
        self.clip_intensities = clip_intensities

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        assert data is not None, "Could not find data key '%s'" % self.data_key
        b, c, *img_shape = data.shape
        for bi in range(b):
            if np.random.uniform() < self.p_per_sample:
                if self.same_for_all_channels:
                    kernel = self._generate_kernel(img_shape)
                    if self.mean_centered:
                        # first center the mean of the kernel
                        kernel -= kernel.mean()
                    mx = max(np.max(np.abs(kernel)), 1e-8)
                    if not callable(self.max_strength):
                        strength = sample_scalar(self.max_strength, None, None)
                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            # now rescale so that the maximum value of the kernel is max_strength
                            strength = sample_scalar(self.max_strength, data[bi, ci], kernel) if callable(
                                self.max_strength) else strength
                            kernel_scaled = np.copy(kernel) / mx * strength
                            data[bi, ci] += kernel_scaled
                else:
                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            kernel = self._generate_kernel(img_shape)
                            if self.mean_centered:
                                kernel -= kernel.mean()
                            mx = max(np.max(np.abs(kernel)), 1e-8)
                            strength = sample_scalar(self.max_strength, data[bi, ci], kernel)
                            kernel = kernel / mx * strength
                            data[bi, ci] += kernel
        return data_dict


class LocalGammaTransform(LocalTransform):
    def __init__(self,
                 scale: ScalarType,
                 loc: ScalarType = (-1, 2),
                 gamma: ScalarType = (0.5, 1),
                 same_for_all_channels: bool = True,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 data_key: str = "data"):
        """
        This transform is weird and definitely experimental. Use at your own risk

        Applies gamma correction to the image. The gamma value varies locally using a gaussian kernel

        The local gamma values are implemented by placing a Gaussian distribution with sigma=scale somewhere in
        (or close to) the image. The location of the kernel is selected independently for each image dimension.
        The location is encoded in % of the image size. The default value of (-1, 2) means that the location will be
        sampled uniformly from (-image.shape[i], 2* image.shape[i]). It is important to allow the center of the kernel
        to be outside of the image.

        IMPORTANT: Try this with different parametrizations and visualize the outcome to get a better feeling for how
        to use this!

        :param scale: scale of the gradient. Large values recommended!
            float: fixed value
            (float, float): will be sampled independently for each dimension from the interval [scale[0], scale[1]]
            callable: you get all the freedom you want. Will be called as scale(image.shape, dimension) where dimension
            is the index in image.shape we are requesting the scale for. Must return scalar (float).
        :param loc:
            (float, float): sample location uniformly from interval [scale[0], scale[1]] (see main description)
            callable: you get all the freedom you want. Will be called as loc(image.shape, dimension) where dimension
            is the index in image.shape we are requesting the location for. Must return a scalar value denoting a relative
            position along axis dimension (0 for index 0, 1 for image.shape[dimension]. Values beyond 0 and 1 are
            possible and even recommended)
        :param gamma: gamma value at the peak of the gaussian distribution.
            Recommended: lambda: np.random.uniform(0.01, 1) if np.random.uniform() < 1 else np.random.uniform(1, 3)
            No, this is not a joke. Deal with it.
        :param same_for_all_channels: If True, then the same gradient will be applied to all selected color
        channels of a sample (see p_per_channel). If False, each selected channel obtains its own random gradient.
        :param allow_kernel_inversion:
        :param p_per_sample:
        :param p_per_channel:
        :param data_key:
        """
        super().__init__(scale, loc)
        self.gamma = gamma
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.same_for_all_channels = same_for_all_channels

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        assert data is not None, "Could not find data key '%s'" % self.data_key
        b, c, *img_shape = data.shape
        for bi in range(b):
            if np.random.uniform() < self.p_per_sample:
                if self.same_for_all_channels:
                    kernel = self._generate_kernel(img_shape)

                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            data[bi, ci] = self._apply_gamma_gradient(data[bi, ci], kernel)
                else:
                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            kernel = self._generate_kernel(img_shape)
                            data[bi, ci] = self._apply_gamma_gradient(data[bi, ci], kernel)
        return data_dict

    def _apply_gamma_gradient(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # store keep original image range
        mn, mx = img.min(), img.max()

        # rescale tp [0, 1]
        img = (img - mn) / (max(mx - mn, 1e-8))

        gamma = sample_scalar(self.gamma)
        img_modified = np.power(img, gamma)

        return self.run_interpolation(img, img_modified, kernel) * (mx - mn) + mn


class LocalSmoothingTransform(LocalTransform):
    def __init__(self,
                 scale: ScalarType,
                 loc: ScalarType = (-1, 2),
                 smoothing_strength: ScalarType = (0.5, 1),
                 kernel_size: ScalarType = (0.5, 1),
                 same_for_all_channels: bool = True,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 data_key: str = "data"):
        """
        Creates a local blurring of the image. This is achieved by creating a blurred copy of the image and then
        linearly interpolating between the original and smoothed images:
            result = image_orig * (1 - smoothing_strength) + smoothed_image * smoothing_strength
        The interpolation only happens where the local gaussian is placed (defined by scale and loc)
        strength of smoothing is determined by kernel_size in combination with smoothing_strength. You can set
        smoothing_strength=1 for simplicity
        :param scale:
        :param loc:
        :param smoothing_strength:
        :param kernel_size:
        :param same_for_all_channels:
        :param p_per_sample:
        :param p_per_channel:
        :param data_key:
        """
        super().__init__(scale, loc)
        self.smoothing_strength = smoothing_strength
        self.same_for_all_channels = same_for_all_channels
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.kernel_size = kernel_size

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        assert data is not None, "Could not find data key '%s'" % self.data_key
        b, c, *img_shape = data.shape
        for bi in range(b):
            if np.random.uniform() < self.p_per_sample:
                if self.same_for_all_channels:
                    kernel = self._generate_kernel(img_shape)

                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            data[bi, ci] = self._apply_local_smoothing(data[bi, ci], kernel)
                else:
                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            kernel = self._generate_kernel(img_shape)
                            data[bi, ci] = self._apply_local_smoothing(data[bi, ci], kernel)
        return data_dict

    def _apply_local_smoothing(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # copy kernel so that we don't modify it out of scope
        kernel = np.copy(kernel)

        smoothing = sample_scalar(self.smoothing_strength)
        assert 0 <= smoothing <= 1, 'smoothing_strength must be between 0 and 1, is %f' % smoothing

        # prepare kernel by rescaling it to gamma_range
        # kernel is already [0, 1]
        kernel *= smoothing

        smoothing_kernel_size = sample_scalar(self.kernel_size)
        img_smoothed = gaussian_filter(img, smoothing_kernel_size)

        return self.run_interpolation(img, img_smoothed, kernel)


class LocalContrastTransform(LocalTransform):
    def __init__(self,
                 scale: ScalarType,
                 loc: ScalarType = (-1, 2),
                 new_contrast: ScalarType = (0.5, 1),
                 same_for_all_channels: bool = True,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 data_key: str = "data"):
        super().__init__(scale, loc)
        self.new_contrast = new_contrast
        self.same_for_all_channels = same_for_all_channels
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.data_key = data_key

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        assert data is not None, "Could not find data key '%s'" % self.data_key
        b, c, *img_shape = data.shape
        for bi in range(b):
            if np.random.uniform() < self.p_per_sample:
                if self.same_for_all_channels:
                    kernel = self._generate_kernel(img_shape)

                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            data[bi, ci] = self._apply_local_smoothing(data[bi, ci], kernel)
                else:
                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            kernel = self._generate_kernel(img_shape)
                            data[bi, ci] = self._apply_local_smoothing(data[bi, ci], kernel)
        return data_dict

    def _apply_local_smoothing(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # copy kernel so that we don't modify it out of scope
        kernel = np.copy(kernel)

        new_contrast = sample_scalar(self.new_contrast)

        # we compute the mean within the kernel
        mean = (img * kernel).sum() / kernel.sum()
        img_modified = (img - mean) * new_contrast + mean

        return self.run_interpolation(img, img_modified, kernel)


if __name__ == '__main__':
    from copy import deepcopy
    from skimage.data import camera
    from batchviewer import view_batch  # https://github.com/FabianIsensee/BatchViewer

    """
    data = {'data': np.vstack((camera()[None], camera()[None], camera()[None], camera()[None]))[None].astype(np.float32)}
    
    tr = LocalSmoothingTransform(
        lambda x, y: np.random.uniform(x[y] // 6, x[y] // 2),
        (0, 1),
        1,
        13,
        same_for_all_channels=False
    )
    transformed = tr(**deepcopy(data))['data']
    data['data'][0][:, 0:2, 0] = np.array((0, 255))
    transformed[0][:, 0:2, 0] = np.array((0, 255))
    diff = [i - j for i, j in zip(data['data'][0], transformed[0])]
    [print(i[10,10]) for i in diff]
    view_batch(*data['data'][0], *transformed[0], *[i - j for i, j in zip(data['data'][0], transformed[0])])"""


    """
    data = {'data': np.vstack((camera()[None], camera()[None], camera()[None], camera()[None]))[None].astype(np.float32)}

    tr = LocalGammaTransform(
        lambda x, y: np.random.uniform(x[y] // 6, x[y] // 2),
        (0, 1),
        (0, 3),
        False,
        1,
        1
    )
    transformed = tr(**deepcopy(data))['data']
    data['data'][0][:, 0:2, 0] = np.array((0, 255))
    transformed[0][:, 0:2, 0] = np.array((0, 255))
    diff = [i - j for i, j in zip(data['data'][0], transformed[0])]
    [print(i[10,10]) for i in diff]
    view_batch(*data['data'][0], *transformed[0], *[i - j for i, j in zip(data['data'][0], transformed[0])])
    """

    data = {'data': np.vstack((camera()[None], camera()[None], camera()[None], camera()[None]))[None].astype(np.float32)}

    tr = LocalContrastTransform(
        lambda x, y: np.random.uniform(x[y] // 6, x[y] // 2),
        (0, 1),
        (0, 0.1),
        False,
        1,
        1
    )
    transformed = tr(**deepcopy(data))['data']
    data['data'][0][:, 0:2, 0] = np.array((0, 255))
    transformed[0][:, 0:2, 0] = np.array((0, 255))
    diff = [i - j for i, j in zip(data['data'][0], transformed[0])]
    [print(i[10,10]) for i in diff]
    view_batch(*data['data'][0], *transformed[0], *[i - j for i, j in zip(data['data'][0], transformed[0])])


    data = {'data': np.vstack((camera()[None], camera()[None], camera()[None], camera()[None]))[None].astype(np.float32)}

    tr = BrightnessGradientAdditiveTransform(
        lambda x, y: np.random.uniform(x[y] // 6, x[y] // 2),
        (0, 1),
        (-128, 128),
        False,
        1,
        1
    )
    transformed = tr(**deepcopy(data))['data']
    data['data'][0][:, 0:2, 0] = np.array((0, 255))
    transformed[0][:, 0:2, 0] = np.array((0, 255))
    diff = [i - j for i, j in zip(data['data'][0], transformed[0])]
    [print(i[10,10]) for i in diff]
    view_batch(*data['data'][0], *transformed[0], *[i - j for i, j in zip(data['data'][0], transformed[0])])


