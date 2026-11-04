"""Public one-hot/argmax round trips must preserve overlapping label values."""

import itertools
import unittest

import numpy as np

from batchgenerators.transforms.utility_transforms import (
    ConvertMultiSegToArgmaxTransform,
    ConvertMultiSegToOnehotTransform,
    ConvertSegToArgmaxTransform,
    ConvertSegToOnehotTransform,
)

LABELS = [(-1, 0, 1), (-2, 0, 1), (0, 2, 5), (-3, -2, -1), None]
DTYPES = [np.int16, np.float32]


def make_segmentation(labels, channels=1):
    indices = np.array([[[0, 1, 2], [2, 1, 0]], [[1, 0, 2], [0, 2, 1]]])
    seg = np.asarray(labels, dtype=np.int16)[indices[:, None]]
    if channels == 2:
        seg = np.concatenate([seg, seg[..., ::-1]], axis=1)
    return seg


class TestArgmaxLabelMapping(unittest.TestCase):
    def test_single_segmentation_label_round_trip(self):
        for labels, keepdim, dtype in itertools.product(LABELS, [True, False], DTYPES):
            with self.subTest(labels=labels, keepdim=keepdim, dtype=dtype):
                classes = (0, 1, 2) if labels is None else labels
                seg = make_segmentation(classes)
                encoded = ConvertSegToOnehotTransform(classes)(seg=seg)["seg"].astype(dtype)
                original = encoded.copy()
                data = np.zeros_like(seg)

                result = ConvertSegToArgmaxTransform(labels, keepdim)(data=data, seg=encoded)

                expected = seg if keepdim else seg[:, 0]
                np.testing.assert_array_equal(result["seg"], expected)
                self.assertEqual(result["seg"].dtype, np.dtype(np.intp))
                np.testing.assert_array_equal(encoded, original)
                self.assertIs(result["data"], data)

    def test_multiple_segmentations_label_round_trip(self):
        for labels, channels, dtype in itertools.product(LABELS, [1, 2], DTYPES):
            with self.subTest(labels=labels, channels=channels, dtype=dtype):
                classes = (0, 1, 2) if labels is None else labels
                seg = make_segmentation(classes, channels)
                encoded = ConvertMultiSegToOnehotTransform(classes)(seg=seg)["seg"].astype(dtype)
                original = encoded.copy()
                data = np.zeros_like(seg)

                result = ConvertMultiSegToArgmaxTransform(channels, labels)(data=data, seg=encoded)

                np.testing.assert_array_equal(result["seg"], seg)
                self.assertEqual(result["seg"].dtype, np.dtype(dtype))
                np.testing.assert_array_equal(encoded, original)
                self.assertIs(result["data"], data)


if __name__ == "__main__":
    unittest.main()
