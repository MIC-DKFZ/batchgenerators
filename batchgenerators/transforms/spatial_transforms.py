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

import numpy as np

from batchgenerators.augmentations.spatial_transformations import augment_spatial, augment_spatial_2, \
    augment_channel_translation, augment_mirroring_batched, augment_transpose_axes, augment_zoom, augment_resize, \
    augment_rot90, augment_anatomy_informed, augment_misalign
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class Rot90Transform(AbstractTransform):
    def __init__(self, num_rot=(1, 2, 3), axes=(0, 1, 2), data_key="data", label_key="seg", p_per_sample=0.3):
        """
        :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
        :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
        :param data_key:
        :param label_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.label_key = label_key
        self.data_key = data_key
        self.axes = axes
        self.num_rot = num_rot

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                d = data[b]
                if seg is not None:
                    s = seg[b]
                else:
                    s = None
                d, s = augment_rot90(d, s, self.num_rot, self.axes)
                data[b] = d
                if s is not None:
                    seg[b] = s

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg
        return data_dict


class ZoomTransform(AbstractTransform):
    def __init__(self, zoom_factors=1, order=3, order_seg=1, concatenate_list=False, data_key="data",
                 label_key="seg"):
        """
        Zooms 'data' (and 'seg') by zoom_factors
        :param zoom_factors: int or list/tuple of int
        :param order: interpolation order for data (see skimage.transform.resize)
        :param order_seg: interpolation order for seg (see skimage.transform.resize)
        :param cval_seg: cval for segmentation (see skimage.transform.resize)
        :param seg: can be None, if not None then it will also be zoomed by zoom_factors. Can also be list/tuple of
        np.ndarray (just like data). Must also be (b, c, x, y(, z))
        :param concatenate_list: if you give list/tuple of data/seg and set concatenate_list=True then the result will be
        concatenated into one large ndarray (once again b, c, x, y(, z))
        :param data_key:
        :param label_key:

        """
        self.concatenate_list = concatenate_list
        self.order_seg = order_seg
        self.data_key = data_key
        self.label_key = label_key
        self.order = order
        self.zoom_factors = zoom_factors

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if isinstance(data, np.ndarray):
            concatenate = True
        else:
            concatenate = self.concatenate_list

        if seg is not None:
            if isinstance(seg, np.ndarray):
                concatenate_seg = True
            else:
                concatenate_seg = self.concatenate_list
        else:
            concatenate_seg = None

        results = []
        for b in range(len(data)):
            sample_seg = None
            if seg is not None:
                sample_seg = seg[b]
            res_data, res_seg = augment_zoom(data[b], sample_seg, self.zoom_factors, self.order, self.order_seg)
            results.append((res_data, res_seg))

        if concatenate:
            data = np.vstack([i[0][None] for i in results])

        if concatenate_seg is not None and concatenate_seg:
            seg = np.vstack([i[1][None] for i in results])

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg
        return data_dict


class ResizeTransform(AbstractTransform):

    def __init__(self, target_size, order=3, order_seg=1, concatenate_list=False, data_key="data",
                 label_key="seg"):
        """
        Reshapes 'data' (and 'seg') to target_size
        :param target_size: int or list/tuple of int
        :param order: interpolation order for data (see skimage.transform.resize)
        :param order_seg: interpolation order for seg (see skimage.transform.resize)
        :param cval_seg: cval for segmentation (see skimage.transform.resize)
        :param seg: can be None, if not None then it will also be resampled to target_size. Can also be list/tuple of
        np.ndarray (just like data). Must also be (b, c, x, y(, z))
        :param concatenate_list: if you give list/tuple of data/seg and set concatenate_list=True then the result will be
        concatenated into one large ndarray (once again b, c, x, y(, z))
        :param data_key:
        :param label_key:

        """
        self.concatenate_list = concatenate_list
        self.order_seg = order_seg
        self.data_key = data_key
        self.label_key = label_key
        self.order = order
        self.target_size = target_size

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if isinstance(data, np.ndarray):
            concatenate = True
        else:
            concatenate = self.concatenate_list

        if seg is not None:
            if isinstance(seg, np.ndarray):
                concatenate_seg = True
            else:
                concatenate_seg = self.concatenate_list
        else:
            concatenate_seg = None

        results = []
        for b in range(len(data)):
            sample_seg = None
            if seg is not None:
                sample_seg = seg[b]
            res_data, res_seg = augment_resize(data[b], sample_seg, self.target_size, self.order, self.order_seg)
            results.append((res_data, res_seg))

        if concatenate:
            data = np.vstack([i[0][None] for i in results])

        if concatenate_seg is not None and concatenate_seg:
            seg = np.vstack([i[1][None] for i in results])

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg
        return data_dict


class MirrorTransform(AbstractTransform):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1, 2), data_key="data", label_key="seg", p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        seg = data_dict.get(self.label_key)

        mask = np.random.uniform(size=len(data)) < self.p_per_sample
        if np.any(mask):
            if seg is None:
                data[mask], _ = augment_mirroring_batched(data[mask], None, self.axes)
            else:
                data[mask], seg[mask] = augment_mirroring_batched(data[mask], seg[mask], self.axes)
                data_dict[self.label_key] = seg
            data_dict[self.data_key] = data

        return data_dict


class ChannelTranslation(AbstractTransform):
    """Simulates badly aligned color channels/modalities by shifting them against each other

    Args:
        const_channel: Which color channel is constant? The others are shifted

        max_shifts (dict {'x':2, 'y':2, 'z':2}): How many pixels should be shifted for each channel?

    """

    def __init__(self, const_channel=0, max_shifts=None, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.max_shift = max_shifts
        self.const_channel = const_channel

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)

        ret_val = augment_channel_translation(data=data, const_channel=self.const_channel, max_shifts=self.max_shift)

        data_dict[self.data_key] = ret_val[0]

        return data_dict


class SpatialTransform(AbstractTransform):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size

        independent_scale_for_each_axis: If True, a scale factor will be chosen independently for each axis.
    """

    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="data",
                 label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis: float = 1,
                 p_independent_scale_per_axis: int = 1):
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = tuple(patch_size)
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_rot_per_axis = p_rot_per_axis
        self.p_independent_scale_per_axis = p_independent_scale_per_axis

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        seg = data_dict.get(self.label_key)

        if self.patch_size is None:
            if data.ndim == 4:
                patch_size = data.shape[2:4]
            elif data.ndim == 5:
                patch_size = data.shape[2:5]
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial(data, seg, patch_size=patch_size,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop,
                                  p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                  p_rot_per_sample=self.p_rot_per_sample,
                                  independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                  p_rot_per_axis=self.p_rot_per_axis,
                                  p_independent_scale_per_axis=self.p_independent_scale_per_axis)
        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]

        return data_dict


class SpatialTransform_2(AbstractTransform):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size
    """

    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, deformation_scale=(0, 0.25),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="data",
                 label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis: float = 1,
                 p_independent_scale_per_axis: float = 1):
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.deformation_scale = deformation_scale
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_independent_scale_per_axis = p_independent_scale_per_axis
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_axis = p_rot_per_axis

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if self.patch_size is None:
            if data.ndim == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif data.ndim == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial_2(data, seg, patch_size=patch_size,
                                    patch_center_dist_from_border=self.patch_center_dist_from_border,
                                    do_elastic_deform=self.do_elastic_deform, deformation_scale=self.deformation_scale,
                                    do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                    angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                    border_mode_data=self.border_mode_data,
                                    border_cval_data=self.border_cval_data, order_data=self.order_data,
                                    border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                    order_seg=self.order_seg, random_crop=self.random_crop,
                                    p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                    p_rot_per_sample=self.p_rot_per_sample,
                                    independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                    p_rot_per_axis=self.p_rot_per_axis,
                                    p_independent_scale_per_axis=self.p_independent_scale_per_axis)

        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]

        return data_dict


class TransposeAxesTransform(AbstractTransform):
    def __init__(self, transpose_any_of_these=(0, 1, 2), data_key="data", label_key="seg", p_per_sample=1):
        '''
        This transform will randomly shuffle the axes of transpose_any_of_these.
        Requires your patch size to have the same dimension in all axes specified in transpose_any_of_these. So if
        transpose_any_of_these=(0, 1, 2) the shape must be (128x128x128) and cannotbe, for example (128x128x96)
        (transpose_any_of_these=(0, 1) would be the correct one here)!
        :param transpose_any_of_these: spatial dimensions to transpose, 0=x, 1=y, 2=z. Must be a tuple/list of len>=2
        :param data_key:
        :param label_key:
        '''
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.transpose_any_of_these = transpose_any_of_these
        if max(transpose_any_of_these) > 2:
            raise ValueError("TransposeAxesTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")
        assert isinstance(transpose_any_of_these, (list, tuple)), "transpose_any_of_these must be either list or tuple"
        assert len(
            transpose_any_of_these) >= 2, "len(transpose_any_of_these) must be >=2 -> we need at least 2 axes we " \
                                          "can transpose"

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        for b in range(len(data)):
            if np.random.uniform() < self.p_per_sample:
                if seg is not None:
                    s = seg[b]
                else:
                    s = None
                ret_val = augment_transpose_axes(data[b], s, self.transpose_any_of_these)
                data[b] = ret_val[0]
                if seg is not None:
                    seg[b] = ret_val[1]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg
        return data_dict


class AnatomyInformedTransform(AbstractTransform):
    """
    The data augmentation is presented at MICCAI 2023 in the proceedings of 'Anatomy-informed Data Augmentation for enhanced Prostate Cancer Detection'.
    It simulates the distension or evacuation of the bladder or/and rectal space to mimic typical physiological soft tissue deformations of the prostate
    and generates unique lesion shapes without altering their label.
    You can find more information here: https://github.com/MIC-DKFZ/anatomy_informed_DA
    If you use this augmentation please cite it.

    Args:
        `dil_ranges`: dilation range per organs
        `modalities`: on which input channels should the transformation be applied
        `directions_of_trans`: to which directions should the organs be dilated per organs
        `p_per_sample`: probability of the transformation per organs
        `spacing_ratio`: ratio of the transversal plane spacing and the slice thickness, in our case it was 0.3125/3
        `blur`: Gaussian kernel parameter, we used the value 32 for 0.3125mm transversal plane spacing
        `anisotropy_safety`: it provides a certain protection against transformation artifacts in 2 slices from the image border
        `max_annotation_value`: the value that should be still relevant for the main task
        `replace_value`: segmentation values larger than the `max_annotation_value` will be replaced with
    """

    def __init__(self, dil_ranges, modalities, directions_of_trans, p_per_sample,
                 spacing_ratio=0.3125 / 3.0, blur=32, anisotropy_safety=True,
                 max_annotation_value=1, replace_value=0):
        self.dil_ranges = dil_ranges
        self.modalities = modalities

        self.directions_of_trans = directions_of_trans
        self.p_per_sample = p_per_sample
        self.spacing_ratio = spacing_ratio
        self.blur = blur
        self.anisotropy_safety = anisotropy_safety

        self.max_annotation_value = max_annotation_value
        self.replace_value = replace_value

        self.dim = 3

    def __call__(self, **data_dict):

        data_shape = data_dict['data'].shape
        if len(data_shape) == 5:
            self.dim = 3

        active_organs = []
        for prob in self.p_per_sample:
            if np.random.uniform() < prob:
                active_organs.append(1)
            else:
                active_organs.append(0)

        for b in range(data_shape[0]):
            data_dict['data'][b, :, :, :, :], data_dict['seg'][b, 0, :, :, :] = augment_anatomy_informed(
                data=data_dict['data'][b, :, :, :, :],
                seg=data_dict['seg'][b, 0, :, :, :],
                active_organs=active_organs,
                dilation_ranges=self.dil_ranges,
                directions_of_trans=self.directions_of_trans,
                modalities=self.modalities,
                spacing_ratio=self.spacing_ratio,
                blur=self.blur,
                anisotropy_safety=self.anisotropy_safety,
                max_annotation_value=self.max_annotation_value,
                replace_value=self.replace_value)
        return data_dict


class MisalignTransform(AbstractTransform):
    """
    The misalignment data augmentation is introduced in Nature Scientific reports 2023.
    It simulates additional misalignments/registration errors between multi-channel (multi-modal, longitudinal)
    data to make neural networks robust for misalignments.
    Currently the following transformations are supported, but they can be extended easily:
    - squeezing/scaling (good approximation for misalignments between the T2w and DWI MRI sequences)
    - rotation
    - channel shift/translation
    You can find more information here: https://github.com/MIC-DKFZ/misalignmnet_DA
    If you use this augmentation please cite it: https://www.nature.com/articles/s41598-023-46747-z
    Always double check whether the directions/axes are correct!

    Additional Misalignment-related Args to the Spatial Transforms:
        `im_channels_2_misalign`: on which image channels should the transformation be applied
        `label_channels_2_misalign`: on which segmentation channels should the transformation be applied
        `do_squeeze`: whether misalignment resulted from squeezing is necessary
        `sq_x, sq_y`, `sq_z`: squeezing/scaling ranges per directions, randomly sampled from interval.
        `p_sq_per_sample`: probability of the transformation per sample
        `p_sq_per_dir`: probability of the transformation per direction
        `do_rotation`: whether misalignment resulted from rotation is necessary
        `angle_x`, `angle_y`, `angle_z`: rotation angels per axes, randomly sampled from interval.
        `p_rot_per_sample`: probability of the transformation per sample
        `p_rot_per_axis`: probability of the transformation per axes
        `do_transl`: whether misalignment resulted from rotation is necessary
        `tr_x`, `tr_y`, `tr_z`: shift/translation per directions, randomly sampled from interval.
        `p_transl_per_sample`: probability of the transformation per sample
        `p_transl_per_dir`: probability of the transformation per direction
    """

    def __init__(self, data_key="data", label_key="seg",
                 im_channels_2_misalign=[0, ], label_channels_2_misalign=[0, ],
                 do_squeeze=True, sq_x=[1.0, 1.0], sq_y=[0.9, 1.1], sq_z=[1.0, 1.0],
                 p_sq_per_sample=0.1, p_sq_per_dir=1.0,
                 do_rotation=True,
                 angle_x=(-0 / 360. * 2 * np.pi, 0 / 360. * 2 * np.pi),
                 angle_y=(-0 / 360. * 2 * np.pi, 0 / 360. * 2 * np.pi),
                 angle_z=(-15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                 p_rot_per_sample=0.1, p_rot_per_axis=1.0,
                 do_transl=True, tr_x=[-32, 32], tr_y=[-32, 32], tr_z=[-2, 2],
                 p_transl_per_sample=0.1, p_transl_per_dir=1.0,
                 border_mode_data='constant', border_cval_data=0,
                 border_mode_seg='constant', border_cval_seg=0,
                 order_data=3, order_seg=0):

        self.data_key = data_key
        self.label_key = label_key

        self.im_channels_2_misalign = im_channels_2_misalign
        self.label_channels_2_misalign = label_channels_2_misalign

        self.do_squeeze = do_squeeze
        self.sq_x = sq_x
        self.sq_y = sq_y
        self.sq_z = sq_z
        self.p_sq_per_sample = p_sq_per_sample
        self.p_sq_per_dir = p_sq_per_dir

        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.p_rot_per_sample = p_rot_per_sample
        self.p_rot_per_axis = p_rot_per_axis

        self.do_transl = do_transl
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.tr_z = tr_z
        self.p_transl_per_sample = p_transl_per_sample
        self.p_transl_per_dir = p_transl_per_dir

        self.order_data = order_data
        self.order_seg = order_seg
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if data.shape[1] < 2:
            raise ValueError("only support multi-modal images")
        else:
            if len(data.shape) == 4:
                data_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                data_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")

        ret_val = augment_misalign(data, seg, data_size=data_size,
                                   im_channels_2_misalign=self.im_channels_2_misalign,
                                   label_channels_2_misalign=self.label_channels_2_misalign,
                                   do_squeeze=self.do_squeeze,
                                   sq_x=self.sq_x,
                                   sq_y=self.sq_y,
                                   sq_z=self.sq_z,
                                   p_sq_per_sample=self.p_sq_per_sample,
                                   p_sq_per_dir=self.p_sq_per_dir,
                                   do_rotation=self.do_rotation,
                                   angle_x=self.angle_x,
                                   angle_y=self.angle_y,
                                   angle_z=self.angle_z,
                                   p_rot_per_sample=self.p_rot_per_sample,
                                   p_rot_per_axis=self.p_rot_per_axis,
                                   do_transl=self.do_transl,
                                   tr_x=self.tr_x,
                                   tr_y=self.tr_y,
                                   tr_z=self.tr_z,
                                   p_transl_per_sample=self.p_transl_per_sample,
                                   p_transl_per_dir=self.p_transl_per_dir,
                                   order_data=self.order_data,
                                   border_mode_data=self.border_mode_data,
                                   border_cval_data=self.border_cval_data,
                                   order_seg=self.order_seg,
                                   border_mode_seg=self.border_mode_seg,
                                   border_cval_seg=self.border_cval_seg)

        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]

        return data_dict
