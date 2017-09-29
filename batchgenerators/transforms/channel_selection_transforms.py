from warnings import warn
from abstract_transform import AbstractTransform


class DataChannelSelectionTransform(AbstractTransform):
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, **data_dict):
        data_dict["data"] = data_dict["data"][:, self.channels]
        return data_dict


class SegChannelSelectionTransform(AbstractTransform):
    def __init__(self, channels, keep_discarded_seg=False):
        self.channels = channels
        self.keep_discarded = keep_discarded_seg

    def __call__(self, **data_dict):
        seg = data_dict.get("seg")

        if seg is None:
            warn("You used SegChannelSelectionTransform but there is no 'seg' key in your data_dict, returning data_dict unmodified", Warning)
        else:
            if self.keep_discarded:
                discarded_seg_idx = [i for i in range(len(seg[0])) if i not in self.channels]
                data_dict['discarded_seg'] = seg[:, discarded_seg_idx]
            data_dict["seg"] = seg[:, self.channels]
        return data_dict
