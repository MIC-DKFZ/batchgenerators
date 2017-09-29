import torch
from abstract_transform import AbstractTransform


class BGtoTensor(AbstractTransform):

    def __call__(self, **data_dict):

        data = data_dict.get("data")
        seg = data_dict.get("seg")

        data_dict["data"] = torch.from_numpy(data)
        if seg is not None:
            data_dict["seg"] = torch.from_numpy(seg)

        return data_dict
