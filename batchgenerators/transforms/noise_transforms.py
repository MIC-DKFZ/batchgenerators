from abstract_transform import AbstractTransform
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_noise


# class RicianNoiseTransform(AbstractTransform):
#     def __init__(self, noise_variance=(0, 0.1)):
#         self.noise_variance = noise_variance
#
#     def __call__(self, **data_dict):
#         data_dict["data"] = augment_rician_noise(data_dict['data'], noise_variance=self.noise_variance)
#         return data_dict


class GaussianNoiseTransform(AbstractTransform):
    """Adds additive Gaussian Noise

    Args:
        noise_variance (tuple of float): samples variance of Gaussian distribution from this interval

    CAREFUL: This transform will modify the value range of your data!
    """
    def __init__(self, noise_variance=(0, 0.1)):
        self.noise_variance = noise_variance

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        data = augment_gaussian_noise(data, self.noise_variance)
        data_dict["data"] = data
        return data_dict


