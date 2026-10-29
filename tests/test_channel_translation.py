import unittest
from unittest.mock import patch

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

    def _translate_by_extreme_shift(self, data, const_channel, pick):
        # pick chooses among the shifts offered for each axis
        with patch("numpy.random.choice", side_effect=pick):
            return augment_channel_translation(data.copy(), const_channel=const_channel,
                                               max_shifts={"z": 2, "y": 2, "x": 2})

    def test_const_channel_keeps_its_position(self):
        data = np.random.rand(2, 3, 6, 6)
        result = self._translate_by_extreme_shift(data, 1, lambda shifts: shifts[-1])
        np.testing.assert_array_equal(result[:, 1], data[:, 1])
        self.assertFalse(np.array_equal(result[:, [0, 2]], data[:, [0, 2]]))

    def test_shifts_reach_max_shift_in_both_directions(self):
        data = np.random.rand(2, 2, 6, 6, 6)
        largest = self._translate_by_extreme_shift(data, 0, lambda shifts: shifts[-1])
        smallest = self._translate_by_extreme_shift(data, 0, lambda shifts: shifts[0])
        expected_largest = np.zeros_like(data[:, 1])
        expected_largest[:, 2:, 2:, 2:] = data[:, 1, :-2, :-2, :-2]
        expected_smallest = np.zeros_like(data[:, 1])
        expected_smallest[:, :-2, :-2, :-2] = data[:, 1, 2:, 2:, 2:]
        np.testing.assert_array_equal(largest[:, 1], expected_largest)
        np.testing.assert_array_equal(smallest[:, 1], expected_smallest)
        np.testing.assert_array_equal(largest[:, 0], data[:, 0])

    def test_zero_max_shift_leaves_data_unchanged(self):
        for shape in ((2, 3, 5, 5), (2, 3, 5, 5, 5)):
            with self.subTest(shape=shape):
                data = np.random.rand(*shape)
                result = augment_channel_translation(data.copy(), const_channel=2,
                                                     max_shifts={"z": 0, "y": 0, "x": 0})
                np.testing.assert_array_equal(result, data)


if __name__ == "__main__":
    unittest.main()
