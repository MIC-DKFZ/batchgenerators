"""Public one-hot/argmax round trips must preserve overlapping label values."""

import numpy as np
import pytest

from batchgenerators.transforms.utility_transforms import (
    ConvertMultiSegToArgmaxTransform,
    ConvertMultiSegToOnehotTransform,
    ConvertSegToArgmaxTransform,
    ConvertSegToOnehotTransform,
)

LABELS = [(-1, 0, 1), (-2, 0, 1), (0, 2, 5), (-3, -2, -1), None]


def make_segmentation(labels, channels=1):
    indices = np.array([[[0, 1, 2], [2, 1, 0]], [[1, 0, 2], [0, 2, 1]]])
    seg = np.asarray(labels, dtype=np.int16)[indices[:, None]]
    if channels == 2:
        seg = np.concatenate([seg, seg[..., ::-1]], axis=1)
    return seg


@pytest.mark.parametrize("labels", LABELS)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [np.int16, np.float32])
def test_single_segmentation_label_round_trip(labels, keepdim, dtype):
    classes = (0, 1, 2) if labels is None else labels
    seg = make_segmentation(classes)
    encoded = ConvertSegToOnehotTransform(classes)(seg=seg)["seg"].astype(dtype)
    original = encoded.copy()
    data = np.zeros_like(seg)

    result = ConvertSegToArgmaxTransform(labels, keepdim)(data=data, seg=encoded)

    expected = seg if keepdim else seg[:, 0]
    np.testing.assert_array_equal(result["seg"], expected)
    assert result["seg"].dtype == np.dtype(np.intp)
    np.testing.assert_array_equal(encoded, original)
    assert result["data"] is data


@pytest.mark.parametrize("labels", LABELS)
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("dtype", [np.int16, np.float32])
def test_multiple_segmentations_label_round_trip(labels, channels, dtype):
    classes = (0, 1, 2) if labels is None else labels
    seg = make_segmentation(classes, channels)
    encoded = ConvertMultiSegToOnehotTransform(classes)(seg=seg)["seg"].astype(dtype)
    original = encoded.copy()
    data = np.zeros_like(seg)

    result = ConvertMultiSegToArgmaxTransform(channels, labels)(data=data, seg=encoded)

    np.testing.assert_array_equal(result["seg"], seg)
    assert result["seg"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(encoded, original)
    assert result["data"] is data
