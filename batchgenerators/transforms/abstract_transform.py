import abc
import numpy as np


class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str


class RndTransform(AbstractTransform):

    def __init__(self, transform, prob=0.5, alternative_transform=None):
        self.alternative_transform = alternative_transform
        self.transform = transform
        self.prob = prob

    def __call__(self, **data_dict):
        rnd_val = np.random.uniform()

        if rnd_val < self.prob:
            return self.transform(**data_dict)
        else:
            if self.alternative_transform is not None:
                return self.alternative_transform(**data_dict)
            else:
                return data_dict


class Compose(AbstractTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data_dict):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"
