from .abstract_transforms import AbstractTransform, Compose, RndTransform
from .channel_selection_transforms import DataChannelSelectionTransform, SegChannelSelectionTransform
from .color_transforms import BrightnessMultiplicativeTransform, BrightnessTransform, ContrastAugmentationTransform, \
    FancyColorTransform, GammaTransform, IlluminationTransform
from .crop_and_pad_transforms import CenterCropSegTransform, CenterCropTransform, PadTransform, RandomCropTransform
from .noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
from .sample_normalization_transforms import CutOffOutliersTransform, RangeTransform, ZeroMeanUnitVarianceTransform

from .utility_transforms import ConvertSegToOnehotTransform, ListToTensor, NumpyToTensor, RenameTransform
from .spatial_transforms import ChannelTranslation, Mirror, SpatialTransform, Zoom, TransposeAxesTransform

transform_list = [
    AbstractTransform, Compose, RndTransform, DataChannelSelectionTransform,
    SegChannelSelectionTransform, BrightnessMultiplicativeTransform, BrightnessTransform,
    ContrastAugmentationTransform, FancyColorTransform, GammaTransform, IlluminationTransform,
    CenterCropSegTransform, CenterCropTransform, PadTransform, RandomCropTransform,
    GaussianNoiseTransform, GaussianBlurTransform, CutOffOutliersTransform, RangeTransform,
    ZeroMeanUnitVarianceTransform, ChannelTranslation, Mirror, SpatialTransform, Zoom,
    ConvertSegToOnehotTransform, ListToTensor, NumpyToTensor
]

