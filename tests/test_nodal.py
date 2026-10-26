"""Nodal (Alt -> camera) solver on synthetic rigs. Synthetic only; not a hardware result."""
import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rema import nodal as N
from rema.pose import inv, rt

K = np.array([[1174.3, 0, 678.1], [0, 1178.4, 390.9], [0, 0, 1.0]])   # values from the user's lens solve
DIST = np.array([-0.2979, 0.1255, 0.0001, 0.0002, 0.0])
BOARD = cv2.aruco.CharucoBoard((10, 7), 0.025, 0.0185, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
OBJ = BOARD.getChessboardCorners().astype(np.float64)
# camera <- tracker: Alt faces up (its +Z = camera -Y), mounted ~12 cm above and 5 cm behind the lens
X_TRUE = inv(rt(cv2.Rodrigues(np.array([-math.pi / 2, 0, 0]))[0], [0.01, -0.12, -0.05]))
Z_TRUE = rt(cv2.Rodrigues(np.array([0.05, -0.1, 0.02]))[0], [-0.12, -0.05, 1.6])   # world <- board (on a wall)


def make_samples(rotations, seed=0, noise_px=0.3, Z_board=None, travel=0.3):
    """noise_px = corner noise of the observation used for PnP (0.3 single frame; ~0.05 for a 32-frame hold average)."""
    rng = np.random.default_rng(seed)
    out = []
    for rvec in rotations:
        Twc = rt(cv2.Rodrigues(np.asarray(rvec, float))[0], rng.uniform(-travel, travel, 3) * [1, 0.5, 1])
        A = Twc @ X_TRUE                            # world <- tracker = (world <- camera) @ (camera <- tracker)
        B = X_TRUE @ inv(A) @ (Z_TRUE if Z_board is None else Z_board)   # camera <- board
        assert ((OBJ @ B[:3, :3].T + B[:3, 3])[:, 2] > 0.3).all(), 'synthetic board must be in front of the camera'
        rv, _ = cv2.Rodrigues(B[:3, :3])
        img, _ = cv2.projectPoints(OBJ, rv, B[:3, 3], K, DIST)
        img = img.reshape(-1, 2) + rng.normal(0, noise_px, (len(OBJ), 2))
        if not ((img[:, 0] > 0) & (img[:, 0] < 1360) & (img[:, 1] > 0) & (img[:, 1] < 768)).all():
            continue
        Bm, info = N.board_pose(OBJ, img, K, DIST, noise_px=noise_px)
        if Bm is not None:
            out.append({'A': A, 'B': Bm, 'obj': OBJ, 'img': img, 'ambiguous': info['ambiguous'], 'rot_std_deg': info['rot_std_deg']})
    return out


def diverse(n=16, seed=1):
    rng = np.random.default_rng(seed)
    return [np.radians([rng.uniform(-15, 15), rng.uniform(-20, 20), rng.uniform(-12, 12)]) for _ in range(n)]


class Nodal(unittest.TestCase):
    def test_recovers_offset_with_diverse_rotations(self):
        s = make_samples(diverse())
        self.assertGreaterEqual(len(s), 12)
        r = N.solve_nodal(s, K, DIST)
        T = np.array(r['T_tracker_cam'])
        T_true = inv(X_TRUE)
        self.assertLess(np.linalg.norm(T[:3, 3] - T_true[:3, 3]) * 1000, 3.0, r)
        self.assertLess(N.rotation_angle_deg(T[:3, :3].T @ T_true[:3, :3]), 0.2)
        self.assertLess(r['heldout_rms_px_median_max'][0], 1.0)
        self.assertTrue(r['diversity']['ok'])
        self.assertAlmostEqual(r['camera_axis_vs_tracker_deg'], 90.0, delta=0.5)

    def test_pan_only_is_rejected_with_advice(self):
        pans = [np.radians([0, a, 0]) for a in np.linspace(-20, 20, 14)]
        s = make_samples(pans)
        with self.assertRaises(ValueError) as cm:
            N.solve_nodal(s, K, DIST)
        self.assertIn('ONE axis', str(cm.exception))

    def test_corrupted_sample_is_dropped(self):
        s = make_samples(diverse(18, seed=3))
        bad = 2                                      # a fitting sample (index 3 would be held out)
        s[bad]['B'] = s[bad]['B'] @ rt(cv2.Rodrigues(np.radians([25, 0, 0]))[0], [0, 0, 0])
        s[bad]['img'] = s[bad]['img'] + 25.0         # inconsistent observation
        r = N.solve_nodal(s, K, DIST)
        self.assertIn(bad, [d['sample'] for d in r['dropped_outliers']])
        self.assertLess(np.linalg.norm(np.array(r['T_tracker_cam'])[:3, 3] - inv(X_TRUE)[:3, 3]) * 1000, 3.0)

    def test_ambiguous_board_flagged_not_lost(self):
        # far, frontal board: IPPE's two solutions become indistinguishable under noise
        rng = np.random.default_rng(5)
        img, _ = cv2.projectPoints(OBJ, np.zeros(3), np.array([0, 0, 6.0]), K, DIST)
        T, info = N.board_pose(OBJ, img.reshape(-1, 2) + rng.normal(0, 0.5, (len(OBJ), 2)), K, DIST)
        self.assertIsNotNone(T)
        self.assertTrue(info['ambiguous'])

    def test_bad_board_reprojection_rejected(self):
        img, _ = cv2.projectPoints(OBJ, np.zeros(3), np.array([0, 0, 1.0]), K, DIST)
        img = img.reshape(-1, 2)
        img[::3] += 8.0
        T, info = N.board_pose(OBJ, img, K, DIST)
        self.assertIsNone(T)
        self.assertEqual(info['reason'], 'board_reprojection_high')

    def test_close_board_uses_rotation_first(self):
        # board ~0.45 m away (like the real recording): precise orientations -> rotation-first is chosen
        Zc = rt(cv2.Rodrigues(np.array([0.05, -0.1, 0.02]))[0], [-0.12, -0.08, 0.45])
        rng = np.random.default_rng(11)
        rots = [np.radians([rng.uniform(-12, 12), rng.uniform(-15, 15), rng.uniform(-8, 8)]) for _ in range(18)]
        s = make_samples(rots, seed=11, Z_board=Zc, travel=0.08, noise_px=0.05)
        self.assertGreaterEqual(len(s), 10)
        r = N.solve_nodal(s, K, DIST)
        self.assertEqual(r['method'], 'rotation_first')
        T = np.array(r['T_tracker_cam'])
        self.assertLess(N.rotation_angle_deg(T[:3, :3].T @ inv(X_TRUE)[:3, :3]), 0.1)
        self.assertLess(r['methods']['rotation_first']['heldout_mm_deg_median'][1], 0.1)

    def test_far_board_keeps_pixel_method(self):
        r = N.solve_nodal(make_samples(diverse()), K, DIST)
        self.assertEqual(r['method'], 'pixel')

    def test_diversity_metric(self):
        single = [cv2.Rodrigues(np.radians([0, a, 0]))[0] for a in range(0, 40, 5)]
        self.assertLess(N.rotation_diversity(single)['axis_spread'], 0.05)
        mixed = [cv2.Rodrigues(r)[0] for r in diverse(10)]
        self.assertGreater(N.rotation_diversity(mixed)['axis_spread'], 0.3)


if __name__ == '__main__':
    unittest.main()
