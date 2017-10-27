from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_brightness_additive, \
    augment_brightness_multiplicative, augment_gamma, augment_illumination, augment_PCA_shift
from batchgenerators.transforms.abstract_transform import AbstractTransform


class ContrastAugmentationTransform(AbstractTransform):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        data_dict['data'] = augment_contrast(data_dict['data'], contrast_range=self.contrast_range,
                                             preserve_range=self.preserve_range, per_channel=self.per_channel)

        return data_dict


class BrightnessTransform(AbstractTransform):
    def __init__(self, mu, sigma, per_channel=True):
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        data_dict['data'] = augment_brightness_additive(data_dict['data'], self.mu, self.sigma, self.per_channel)
        return data_dict


class BrightnessMultiplicativeTransform(AbstractTransform):
    def __init__(self, multiplier_range=(0.5, 2), per_channel=True):
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        data_dict['data'] = augment_brightness_multiplicative(data_dict['data'], self.multiplier_range, self.per_channel)
        return data_dict


class GammaTransform(AbstractTransform):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False):
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        data_dict['data'] = augment_gamma(data_dict['data'], self.gamma_range, self.invert_image)
        return data_dict


class IlluminationTransform(AbstractTransform):
    def __init__(self, white_rgb):
        self.white_rgb = white_rgb

    def __call__(self, **data_dict):
        data_dict['data'] = augment_illumination(data_dict['data'], self.white_rgb)
        return data_dict


class FancyColorTransform(AbstractTransform):
    def __init__(self, U, s, sigma=0.2):
        self.s = s
        self.U = U
        self.sigma = sigma

    def __call__(self, **data_dict):
        data_dict['data'] = augment_PCA_shift(data_dict['data'], self.U, self.s, self.sigma)
        return data_dict


