import unittest

import numpy as np

from batchgenerators.augmentations.spatial_transformations import augment_misalign
from batchgenerators.transforms.spatial_transforms import MisalignTransform


class TestMisalign(unittest.TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())

    def _make_sample_2d(self):
        data = np.zeros((1, 2, 16, 16), dtype=np.float32)
        data[0, 0, 4:12, 4:12] = 1.0
        data[0, 1, 4:12, 4:12] = 2.0
        seg = np.zeros((1, 2, 16, 16), dtype=np.float32)
        seg[0, 0, 4:12, 4:12] = 1.0
        seg[0, 1, 4:12, 4:12] = 1.0
        return data, seg

    def _make_sample_3d(self):
        data = np.zeros((1, 2, 8, 12, 12), dtype=np.float32)
        data[0, 0, 2:6, 3:9, 3:9] = 1.0
        data[0, 1, 2:6, 3:9, 3:9] = 2.0
        seg = np.zeros((1, 1, 8, 12, 12), dtype=np.float32)
        seg[0, 0, 2:6, 3:9, 3:9] = 1.0
        return data, seg

    def test_squeeze_respects_label_channels_2_misalign(self):
        data, seg = self._make_sample_2d()
        out_data, out_seg = augment_misalign(
            data.copy(),
            seg.copy(),
            (16, 16),
            im_channels_2_misalign=[0],
            label_channels_2_misalign=[1],
            do_squeeze=True,
            sq_x=[0.5, 0.5],
            sq_y=[0.5, 0.5],
            p_sq_per_sample=1.0,
            p_sq_per_dir=1.0,
        )
        self.assertFalse(np.array_equal(out_data[0, 0], data[0, 0]))
        np.testing.assert_array_equal(out_data[0, 1], data[0, 1])
        np.testing.assert_array_equal(out_seg[0, 0], seg[0, 0])
        self.assertFalse(np.array_equal(out_seg[0, 1], seg[0, 1]))

    def test_rotation_respects_label_channels_2_misalign(self):
        data, seg = self._make_sample_2d()
        out_data, out_seg = augment_misalign(
            data.copy(),
            seg.copy(),
            (16, 16),
            im_channels_2_misalign=[0],
            label_channels_2_misalign=[1],
            do_rotation=True,
            angle_z=(np.pi / 4, np.pi / 4),
            p_rot_per_sample=1.0,
            p_rot_per_axis=1.0,
        )
        self.assertFalse(np.array_equal(out_data[0, 0], data[0, 0]))
        np.testing.assert_array_equal(out_data[0, 1], data[0, 1])
        np.testing.assert_array_equal(out_seg[0, 0], seg[0, 0])
        self.assertFalse(np.array_equal(out_seg[0, 1], seg[0, 1]))

    def test_empty_label_channels_leaves_seg_unchanged(self):
        data, seg = self._make_sample_2d()
        transform = MisalignTransform(
            im_channels_2_misalign=[0],
            label_channels_2_misalign=[],
            do_squeeze=True,
            sq_x=[0.5, 0.5],
            sq_y=[0.5, 0.5],
            p_sq_per_sample=1.0,
            p_sq_per_dir=1.0,
            do_rotation=True,
            angle_z=(np.pi / 4, np.pi / 4),
            p_rot_per_sample=1.0,
            p_rot_per_axis=1.0,
            do_transl=True,
            tr_x=[2, 2],
            tr_y=[2, 2],
            p_transl_per_sample=1.0,
            p_transl_per_dir=1.0,
        )
        result = transform(data=data.copy(), seg=seg.copy())
        self.assertFalse(np.array_equal(result["data"][0, 0], data[0, 0]))
        np.testing.assert_array_equal(result["data"][0, 1], data[0, 1])
        np.testing.assert_array_equal(result["seg"], seg)

    def test_single_label_channel_follows_nonzero_image_channel_3d(self):
        data, seg = self._make_sample_3d()
        out_data, out_seg = augment_misalign(
            data.copy(),
            seg.copy(),
            (8, 12, 12),
            im_channels_2_misalign=[1],
            label_channels_2_misalign=[0],
            do_squeeze=True,
            sq_x=[0.5, 0.5],
            sq_y=[0.5, 0.5],
            sq_z=[1.0, 1.0],
            p_sq_per_sample=1.0,
            p_sq_per_dir=1.0,
            do_rotation=True,
            angle_x=(0.0, 0.0),
            angle_y=(0.0, 0.0),
            angle_z=(np.pi / 4, np.pi / 4),
            p_rot_per_sample=1.0,
            p_rot_per_axis=1.0,
        )
        np.testing.assert_array_equal(out_data[0, 0], data[0, 0])
        self.assertFalse(np.array_equal(out_data[0, 1], data[0, 1]))
        self.assertFalse(np.array_equal(out_seg[0, 0], seg[0, 0]))


if __name__ == "__main__":
    unittest.main()
