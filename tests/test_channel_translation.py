import unittest

import numpy as np

from batchgenerators.augmentations.spatial_transformations import augment_channel_translation
from batchgenerators.transforms.spatial_transforms import ChannelTranslation


class TestChannelTranslation(unittest.TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())

    def _check_translation(self, shape, data_key, const_channel):
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        seg = np.zeros((shape[0], 1, *shape[2:]), dtype=np.uint8)
        labels = ["first", "second"]
        max_shifts = {"x": 1, "y": 2, "z": 1}
        np.random.seed(7)
        expected = augment_channel_translation(
            data.copy(), const_channel=const_channel, max_shifts=max_shifts
        )
        np.random.seed(7)
        result = ChannelTranslation(
            const_channel=const_channel, max_shifts=max_shifts, data_key=data_key
        )(**{data_key: data.copy(), "seg": seg, "labels": labels})

        self.assertEqual(result[data_key].shape, data.shape)
        self.assertEqual(result[data_key].dtype, data.dtype)
        np.testing.assert_array_equal(result[data_key], expected)
        self.assertIs(result["seg"], seg)
        self.assertIs(result["labels"], labels)

    def test_preserves_2d_batch(self):
        self._check_translation((2, 3, 5, 5), "data", 0)

    def test_preserves_3d_batch_with_custom_key(self):
        self._check_translation((2, 3, 5, 5, 5), "images", 1)


if __name__ == "__main__":
    unittest.main()
