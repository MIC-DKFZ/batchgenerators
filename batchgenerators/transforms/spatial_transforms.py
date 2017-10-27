from abstract_transform import AbstractTransform
from batchgenerators.augmentations.spatial_transformations import augment_spatial, augment_channel_translation, \
    augment_mirroring
import numpy as np


class Mirror(AbstractTransform):
    def __init__(self, axes=(2, 3, 4)):
        self.axes = axes

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        seg = data_dict.get("seg")
        ret_val = augment_mirroring(data=data, seg=seg, axes=self.axes)

        data_dict["data"] = ret_val[0]
        if seg is not None:
            data_dict["seg"] = ret_val[1]

        return data_dict


class ChannelTranslation(AbstractTransform):
    def __init__(self, const_channel=0, max_shifts=None):
        self.max_shift = max_shifts
        self.const_channel = const_channel

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        ret_val = augment_channel_translation(data=data, const_channel=self.const_channel, max_shifts=self.max_shift)

        data_dict["data"] = ret_val[0]

        return data_dict


class SpatialTransform(AbstractTransform):
    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=1,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True):
        self.patch_size = patch_size
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

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        seg = data_dict.get("seg")

        ret_val = augment_spatial(data, seg, patch_size=self.patch_size,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop)

        data_dict["data"] = ret_val[0]
        if seg is not None:
            data_dict["seg"] = ret_val[1]

        return data_dict
