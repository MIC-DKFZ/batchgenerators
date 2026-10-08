import unittest

import numpy as np
from skimage.transform import resize

from batchgenerators.augmentations.utils import (interpolate_img, resize_multichannel_image,
                                                resize_segmentation)


def nearest(seg, new_shape):
    return resize(seg.astype(float), new_shape, 0, mode="edge", clip=True, anti_aliasing=False).astype(seg.dtype)


def label_scores(seg, new_shape, order):
    labels = np.sort(np.unique(seg))
    return labels, np.stack([resize((seg == c).astype(float), new_shape, order, mode="edge", clip=True,
                                    anti_aliasing=False) for c in labels])


class TestResizeSegmentation(unittest.TestCase):
    """
    resize_segmentation and interpolate_img resize a segmentation by interpolating each label's indicator and
    picking a winner per voxel. They used to assign with `interpolated >= 0.5` into a zero-initialized result,
    which could leave a voxel unwritten (it then kept a 0 that need not be a label of the input) and, where
    several labels passed 0.5, handed the voxel to whichever label was written last, i.e. the largest.
    """

    def test_no_label_is_invented(self):
        """Four labels meeting at an output centre score 0.25 each, so the old rule wrote nothing there."""
        seg = np.array([[1, 2, 1, 1],
                        [3, 4, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]], dtype=np.uint8)
        labels, scores = label_scores(seg, (2, 2), 1)
        self.assertLess(scores[:, 0, 0].max(), 0.5, 'this test is only meaningful if no label reaches 0.5')
        for tiebreak in ('nearest', 'lowest', 'highest'):
            out = resize_segmentation(seg, (2, 2), order=1, seg_tiebreak=tiebreak)
            self.assertTrue(set(np.unique(out).tolist()) <= set(np.unique(seg).tolist()),
                            f'{np.unique(out)} contains a label that is not in the input ({tiebreak})')

    def test_no_label_is_invented_randomized(self):
        """At order 3 skimage's clip=True breaks sum-to-one outright, so the old rule missed whole voxels."""
        rng = np.random.RandomState(3)
        missed_by_the_old_rule = 0
        for _ in range(60):
            shape = tuple(rng.randint(6, 14) for _ in range(2))
            seg = rng.randint(1, 5, shape).astype(np.uint8)  # labels 1..4, the input never contains 0
            new_shape = tuple(max(int(round(s / f)), 2)
                              for s, f in zip(shape, rng.choice([1.5, 2., 2.5, 3.], 2)))
            labels, scores = label_scores(seg, new_shape, 3)
            missed_by_the_old_rule += int(((scores >= 0.5).sum(0) == 0).sum())
            for tiebreak in ('nearest', 'lowest', 'highest'):
                out = resize_segmentation(seg, new_shape, order=3, seg_tiebreak=tiebreak)
                self.assertTrue(np.isin(out, labels).all(),
                                f'{np.unique(out)} vs input labels {labels} ({tiebreak})')
        self.assertGreater(missed_by_the_old_rule, 0, 'the old rule should have left voxels unwritten here')

    def test_nearest_tiebreak_is_symmetric_in_the_label_values(self):
        """
        An even integer factor puts an output centre exactly between two labels at every boundary voxel, and
        an exact tie carries no information about which label belongs there. 'nearest' answers it from the
        geometry, so relabelling moves the same decision with the labels; 'lowest' and 'highest' answer it
        from the label values, so they do not.
        """
        for lo, hi in ((1, 2), (2, 7), (3, 4)):
            forward = np.full((8, 2), lo, dtype=np.uint8)
            forward[5:] = hi                                  # boundary lands on the output centre x = 4.5
            swapped = np.full((8, 2), hi, dtype=np.uint8)
            swapped[5:] = lo
            remap = {lo: hi, hi: lo}

            a = resize_segmentation(forward, (4, 2), order=1, seg_tiebreak='nearest')
            b = resize_segmentation(swapped, (4, 2), order=1, seg_tiebreak='nearest')
            self.assertTrue(np.array_equal(np.vectorize(remap.get)(a), b),
                            f"'nearest' is not symmetric for ({lo}, {hi}): {a[:, 0]} vs {b[:, 0]}")

            # the tie is at output index 2; 'lowest' and 'highest' take it by label value
            self.assertEqual(resize_segmentation(forward, (4, 2), 1, seg_tiebreak='lowest')[2, 0], lo)
            self.assertEqual(resize_segmentation(forward, (4, 2), 1, seg_tiebreak='highest')[2, 0], hi)

    def test_nearest_tiebreak_matches_a_nearest_neighbour_resize(self):
        seg = np.full((8, 2), 1, dtype=np.uint8)
        seg[5:] = 2
        self.assertTrue(np.array_equal(resize_segmentation(seg, (4, 2), order=1, seg_tiebreak='nearest'),
                                       nearest(seg, (4, 2))))

    def test_nearest_tiebreak_only_picks_among_the_tied_labels(self):
        """
        Where three or more labels meet, the nearest neighbour can be a label that lost: halving this 2x2x2
        block scores 1 and 2 at 0.375 each and 3 at 0.25, and the nearest neighbour of the centre is a 3.
        """
        block = np.array([3, 1, 1, 1, 2, 2, 2, 3], dtype=np.int16).reshape(2, 2, 2)
        seg = np.tile(block, (4, 4, 4))
        labels, scores = label_scores(seg, (4, 4, 4), 1)
        self.assertTrue(np.allclose(scores[:, 0, 0, 0], [0.375, 0.375, 0.25]))
        self.assertEqual(int(nearest(seg, (4, 4, 4))[0, 0, 0]), 3, 'the nearest neighbour has to be the loser')
        for tiebreak, expected in (('nearest', 1), ('lowest', 1), ('highest', 2)):
            out = resize_segmentation(seg, (4, 4, 4), order=1, seg_tiebreak=tiebreak)
            self.assertTrue((out == expected).all(), f'{np.unique(out)} ({tiebreak})')

    def test_winner_has_the_top_score_randomized(self):
        """whatever settles a tie, the label a voxel gets must be one that attains the maximum score there"""
        rng = np.random.RandomState(4)
        for _ in range(40):
            shape = tuple(rng.randint(4, 9) for _ in range(3))
            seg = rng.randint(0, rng.randint(2, 6), shape).astype(np.int16)
            new_shape = tuple(max(s // 2, 1) for s in shape)
            labels, scores = label_scores(seg, new_shape, 1)
            top = scores.max(0)
            for tiebreak in ('nearest', 'lowest', 'highest'):
                out = resize_segmentation(seg, new_shape, order=1, seg_tiebreak=tiebreak)
                own = np.take_along_axis(scores, np.searchsorted(labels, out)[None], 0)[0]
                self.assertTrue((own == top).all(), f'{int((own != top).sum())} voxels ({tiebreak})')

    def test_block_aligned_roundtrip_is_exact(self):
        """00001111 -> 0011 -> 00001111: down and up share the block grid, so nothing may move."""
        seg = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)[:, None].repeat(2, axis=1)
        for tiebreak in ('nearest', 'lowest', 'highest'):
            down = resize_segmentation(seg, (4, 2), order=1, seg_tiebreak=tiebreak)
            back = resize_segmentation(down, (8, 2), order=1, seg_tiebreak=tiebreak)
            self.assertTrue(np.array_equal(back, seg), f'tiebreak={tiebreak}')

    def test_single_label_is_passed_through(self):
        seg = np.full((8, 6), 3, dtype=np.uint8)
        out = resize_segmentation(seg, (4, 3), order=3)
        self.assertTrue(np.array_equal(out, np.full((4, 3), 3, dtype=np.uint8)))

    def test_order_zero_is_untouched(self):
        rng = np.random.RandomState(0)
        seg = rng.randint(0, 5, (9, 7)).astype(np.uint8)
        self.assertTrue(np.array_equal(resize_segmentation(seg, (5, 4), order=0), nearest(seg, (5, 4))))

    def test_dtype_is_preserved(self):
        for dtype in (np.uint8, np.int16, np.int32):
            seg = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=dtype)[:, None].repeat(2, axis=1)
            self.assertEqual(resize_segmentation(seg, (4, 2), order=1).dtype, dtype)

    def test_unknown_tiebreak_is_rejected(self):
        seg = np.zeros((8, 8), dtype=np.uint8)
        seg[4:] = 1
        with self.assertRaises(ValueError):
            resize_segmentation(seg, (4, 4), order=1, seg_tiebreak='bogus')


class TestInterpolateImg(unittest.TestCase):
    """interpolate_img is the same rule on the augment_spatial path, and had the same two problems."""

    @staticmethod
    def _coords(xs, ys):
        return np.stack(np.meshgrid(np.array(xs, dtype=float), np.array(ys, dtype=float), indexing='ij'))

    def test_no_label_is_invented(self):
        img = np.array([[1, 2, 1, 1],
                        [3, 4, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]], dtype=np.uint8)
        coords = self._coords([0.5, 2.5], [0.5, 2.5])
        for tiebreak in ('nearest', 'lowest', 'highest'):
            out = interpolate_img(img, coords, order=1, is_seg=True, seg_tiebreak=tiebreak)
            self.assertTrue(set(np.unique(out).tolist()) <= set(np.unique(img).tolist()),
                            f'{np.unique(out)} contains a label that is not in the input ({tiebreak})')

    def test_nearest_tiebreak_is_symmetric_in_the_label_values(self):
        coords = self._coords([4.5], [0.0])
        for lo, hi in ((1, 2), (2, 7)):
            forward = np.full((8, 2), lo, dtype=np.uint8)
            forward[5:] = hi
            swapped = np.full((8, 2), hi, dtype=np.uint8)
            swapped[5:] = lo
            a = interpolate_img(forward, coords, order=1, is_seg=True, seg_tiebreak='nearest')
            b = interpolate_img(swapped, coords, order=1, is_seg=True, seg_tiebreak='nearest')
            self.assertEqual({lo: hi, hi: lo}[int(a.ravel()[0])], int(b.ravel()[0]))
            self.assertEqual(int(interpolate_img(forward, coords, 1, is_seg=True, seg_tiebreak='lowest').ravel()[0]), lo)
            self.assertEqual(int(interpolate_img(forward, coords, 1, is_seg=True, seg_tiebreak='highest').ravel()[0]), hi)

    def test_constant_padding_still_uses_cval_outside_the_image(self):
        """
        The old code zero-initialized the result, so voxels sampled outside the image happened to come back
        as 0 - right whenever cval was 0, and only then. Now that the result carries a real label from the
        start, cval has to be written there deliberately.
        """
        img = np.array([[1, 1, 2, 2]] * 4, dtype=np.uint8)  # the input does not contain 0
        coords = self._coords([-5., 1.5, 9.], [2.5])        # outside, inside (clear of the 1/2 tie), outside
        for cval in (0.0, 7.0):
            for tiebreak in ('nearest', 'lowest', 'highest'):
                out = interpolate_img(img, coords, order=1, mode='constant', cval=cval, is_seg=True,
                                      seg_tiebreak=tiebreak).ravel()
                self.assertEqual(int(out[0]), int(cval), tiebreak)
                self.assertEqual(int(out[2]), int(cval), tiebreak)
                self.assertEqual(int(out[1]), 2, tiebreak)
        # a cval the dtype cannot hold comes out as order 0 converts it, instead of raising
        for order in (0, 1, 3):
            out = interpolate_img(img, coords, order=order, mode='constant', cval=-1, is_seg=True).ravel()
            self.assertEqual(int(out[0]), int(np.array(-1.).astype(np.uint8)), f'order {order}')
        # padding modes with no 'outside' must be left alone
        self.assertTrue((interpolate_img(img, coords, order=1, mode='nearest', is_seg=True) == 2).all())

    def test_order_zero_and_non_seg_are_untouched(self):
        img = np.arange(16, dtype=np.uint8).reshape(4, 4)
        coords = self._coords([0.5, 2.5], [0.5, 2.5])
        self.assertTrue(np.array_equal(interpolate_img(img, coords, order=0, is_seg=True),
                                       interpolate_img(img, coords, order=0, is_seg=False)))


class TestResizeMultichannelImage(unittest.TestCase):
    """
    The image counterpart in the same module. It has no per-label reduction and therefore none of the
    tie-break problem, but it used to allocate its output with the *input* dtype and assign the
    interpolated float into it, which truncates towards zero for integer input.
    """

    def test_integer_input_is_rounded_not_truncated(self):
        img = (np.arange(8, dtype=np.uint8) * 30)[None, :, None].repeat(2, axis=2)
        exact = resize(img[0].astype(float), (4, 2), 3, clip=True, anti_aliasing=False)
        got = resize_multichannel_image(img, (4, 2), 3)
        self.assertTrue(np.array_equal(got[0], np.rint(exact).astype(np.uint8)))
        self.assertFalse(np.array_equal(got[0], np.trunc(exact).astype(np.uint8)),
                         'this test needs a case where rounding and truncation disagree')

    def test_integer_input_has_no_downward_bias(self):
        """truncation costs about half an intensity step per voxel, everywhere, in one direction"""
        rng = np.random.RandomState(0)
        bias = []
        for _ in range(50):
            a = rng.randint(0, 256, (1, 16, 16)).astype(np.uint8)
            exact = resize(a[0].astype(float), (9, 9), 3, clip=True, anti_aliasing=False)
            bias.append((resize_multichannel_image(a, (9, 9), 3)[0].astype(float) - exact).mean())
        self.assertLess(abs(float(np.mean(bias))), 0.02)

    def test_float_input_is_unchanged(self):
        """the only path our pipelines use, and it must stay bit-identical"""
        rng = np.random.RandomState(0)
        for dtype in (np.float32, np.float64):
            a = rng.rand(2, 16, 16).astype(dtype)
            expected = np.stack([resize(a[i].astype(float), (9, 9), 3, clip=True,
                                        anti_aliasing=False).astype(dtype) for i in range(2)])
            self.assertTrue(np.array_equal(resize_multichannel_image(a, (9, 9), 3), expected))

    def test_dtype_is_preserved(self):
        rng = np.random.RandomState(0)
        for dtype in (np.uint8, np.int16, np.int32, np.float32, np.float64):
            a = (rng.rand(1, 8, 8) * 100).astype(dtype)
            self.assertEqual(resize_multichannel_image(a, (4, 4), 3).dtype, dtype)


if __name__ == '__main__':
    unittest.main()
