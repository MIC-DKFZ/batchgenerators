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
    """Applies a transformation with a specified probability

    Args:
        transform: The transformation (or composed transformation)

        prob: The probability with which to apply it

        alternative_transform: Will be applied if transform is not called. If transform alters for example the
        spatial dimension of the data, you need to compensate that with calling a dummy transformation that alters the
        spatial dimension in a similar way. We included this functionality because of SpatialTransform which has the
        ability to do cropping. If we want to not apply the spatial transformation we will still need to crop and
        therefore set the alternative_transform to an instance of RandomCropTransform of CenterCropTransform
    """
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
