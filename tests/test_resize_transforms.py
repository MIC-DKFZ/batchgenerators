# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from itertools import product

import numpy as np

from batchgenerators.augmentations.spatial_transformations import augment_resize, augment_zoom
from batchgenerators.transforms.spatial_transforms import ResizeTransform, ZoomTransform

# transform, the function it applies to each sample, and the size argument for a batch
OPERATIONS = {
    "resize": (ResizeTransform, augment_resize, lambda data: (2,) * (data[0].ndim - 1)),
    "zoom": (ZoomTransform, augment_zoom, lambda data: 0.5),
}


class TestResizeTransforms(unittest.TestCase):
    def _check_transform(self, operation, data, seg, concatenate=False):
        transform_class, augment, size_for = OPERATIONS[operation]
        size = size_for(data)
        options = {"concatenate_list": True} if concatenate else {}
        transform = transform_class(size, order=0, order_seg=0, **options)
        metadata = object()
        inputs = {"data": data, "metadata": metadata}
        if seg is not None:
            inputs["seg"] = seg
        result = transform(**inputs)
        self.assertIs(result["metadata"], metadata)

        for key, original in (("data", data), ("seg", seg)):
            if original is None:
                self.assertNotIn(key, result)
                continue
            expected_type = np.ndarray if concatenate or isinstance(original, np.ndarray) else type(original)
            self.assertIsInstance(result[key], expected_type)
            self.assertEqual(len(result[key]), len(original))

        for index, sample in enumerate(data):
            sample_seg = None if seg is None else seg[index]
            expected_data, expected_seg = augment(sample, sample_seg, size, 0, 0)
            np.testing.assert_array_equal(result["data"][index], expected_data)
            if seg is not None:
                np.testing.assert_array_equal(result["seg"][index], expected_seg)

    def test_fixed_size_batches(self):
        for ndim, operation, data_type, seg_type, concatenate in product(
                (2, 3), OPERATIONS, (list, tuple, np.stack),
                (None, list, tuple, np.stack), (False, True)):
            with self.subTest(ndim=ndim, operation=operation, data_type=data_type,
                              seg_type=seg_type, concatenate=concatenate):
                shape = (2,) + (4,) * ndim
                sample = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                data = data_type([sample, sample + 100])
                seg = None if seg_type is None else seg_type([
                    (sample[:1] > 5).astype(np.uint8),
                    (sample[:1] > 10).astype(np.uint8),
                ])
                self._check_transform(operation, data, seg, concatenate)

    def test_variable_size_sequences(self):
        for ndim, operation, data_type, seg_type in product(
                (2, 3), OPERATIONS, (list, tuple), (list, tuple)):
            with self.subTest(ndim=ndim, operation=operation,
                              data_type=data_type, seg_type=seg_type):
                samples = [
                    np.arange(2 * side ** ndim, dtype=np.float32).reshape((2,) + (side,) * ndim)
                    for side in (4, 6)
                ]
                seg = seg_type([(sample[:1] > 5).astype(np.uint8) for sample in samples])
                self._check_transform(operation, data_type(samples), seg)


if __name__ == "__main__":
    unittest.main()
