from batchgenerators.transforms.abstract_transform import AbstractTransform
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop, center_crop_seg, random_crop, pad


class CenterCropTransform(AbstractTransform):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        seg = data_dict.get("seg")
        data, seg = center_crop(data, self.output_size, seg)

        data_dict["data"] = data
        if seg is not None:
            data_dict["seg"] = seg

        return data_dict


class CenterCropSegTransform(AbstractTransform):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, **data_dict):
        seg = data_dict.get("seg")

        if seg is not None:
            data_dict["seg"] = center_crop_seg(seg, self.output_size)
        else:
            from warnings import warn
            warn("You shall not pass data_dict without seg: Used CenterCropSegTransform, but there is no seg", Warning)
        return data_dict




class RandomCropTranform(AbstractTransform):
    def __init__(self, crop_size=128, margins =(0, 0, 0)):
        self.margins = margins
        self.crop_size = crop_size

    def __call__(self, **data_dict):

        data = data_dict.get("data")
        seg = data_dict.get("seg")

        data, seg = random_crop(data, seg, self.crop_size, self.margins)

        data_dict["data"] = data
        if seg is not None:
            data_dict["seg"] = seg

        return data_dict



class PadTransform(AbstractTransform):
    def __init__(self, new_size, pad_value_data =None, pad_value_seg =None):
        self.pad_value_seg = pad_value_seg
        self.pad_value_data = pad_value_data
        self.new_size = new_size

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        seg = data_dict.get("seg")

        data, seg = pad(data, self.new_size, seg, self.pad_value_data, self.pad_value_seg)

        if seg is not None:
            data_dict["seg"] = seg

        return data_dict