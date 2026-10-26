"""RED + Antilatency rig: pose conventions, history timing, rendering, SDI capture helpers.
All synthetic; none of this is hardware validation."""
import ctypes
import importlib.util
import json
import math
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rema import pose as P
from rema import sdi_capture as SC
from rema import virtual_scene as VS

RNG = np.random.default_rng(7)


def random_quat():
    q = RNG.normal(size=4)
    return q / np.linalg.norm(q)


def alt2freed_packet(position, rotation, cam_id=1):
    """Python port of alt2freed main.cpp quatToYawPitchRoll + buildFreeD (default signs/offsets)."""
    x, y, z, w = rotation
    m02, m22 = 2 * (x * z + w * y), 1 - 2 * (x * x + y * y)
    m12, m10, m11 = 2 * (y * z - w * x), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z)
    pitch = math.degrees(math.asin(max(-1, min(1, -m12))))
    yaw = math.degrees(math.atan2(m02, m22))
    roll = math.degrees(math.atan2(m10, m11))
    pan, tilt, rl = yaw, -pitch, roll
    fx, fy, fz = position[0], position[2], position[1]

    def put24(v):
        v = max(-0x800000, min(0x7FFFFF, v))
        return (v & 0xFFFFFF).to_bytes(3, 'big')
    body = bytes([0xD1, cam_id]) + put24(round(pan * 32768)) + put24(round(tilt * 32768)) + put24(round(rl * 32768)) \
        + put24(round(fx * 64000)) + put24(round(fy * 64000)) + put24(round(fz * 64000)) + put24(0) + put24(0) + b'\0\0'
    return body + bytes([(0x40 - sum(body)) & 0xFF])


def angle_deg(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))


class Conventions(unittest.TestCase):
    def test_unity_euler_round_trip(self):
        for _ in range(200):
            R = P.quat_to_mat(random_quat())
            # Compare matrices directly; acos-based angles bottom out near 1e-6 deg in float64.
            self.assertLess(np.abs(R - P.unity_euler_to_mat(*P.mat_to_unity_euler(R))).max(), 1e-9)

    def test_freed_matches_alt2freed_encoder(self):
        for _ in range(200):
            q = random_quat()
            if abs(P.mat_to_unity_euler(P.quat_to_mat(q))[0]) > math.radians(85):
                continue                      # gimbal lock: Euler transport is ill-conditioned
            p = RNG.uniform(-5, 5, 3)
            s = P.freed_to_sample(P.decode_freed_d1(alt2freed_packet(p, q)), 0)
            self.assertLess(np.abs(np.array(s.position) - p).max(), 1e-4)
            self.assertLess(angle_deg(P.quat_to_mat(q), P.quat_to_mat(s.rotation)), 0.01)

    def test_freed_checksum_rejected(self):
        pkt = bytearray(alt2freed_packet((0, 1, 2), (0, 0, 0, 1)))
        pkt[5] ^= 1
        with self.assertRaises(ValueError):
            P.decode_freed_d1(bytes(pkt))

    def test_tracker_rh_matches_vpcamtracking(self):
        calib = ROOT / 'VPcamtracking/VPcamtracking/alt2freed/ui/calib.py'
        if not calib.exists():
            self.skipTest('VPcamtracking copy not present')
        src = calib.read_text()
        ns = {}
        # Execute only its pure math helpers (the module itself starts threads/creates folders on import).
        start, end = src.index('def quat_to_mat'), src.index('def run_bin')
        exec('import math\nimport numpy as np\nS = np.diag([1.0, -1.0, 1.0])\n' + src[start:end], ns)
        for _ in range(50):
            p, q = RNG.uniform(-3, 3, 3), random_quat()
            np.testing.assert_allclose(P.tracker_rh(p, q), ns['tracker_rh'](p, q), atol=1e-12)

    def test_world_frame_is_proper_rotation(self):
        self.assertAlmostEqual(np.linalg.det(P.W_FROM_S), 1.0)

    def test_physical_meaning(self):
        """Tracker 1.5 m above the floor looking along Antilatency +Z: in W it sits at z=1.5 looking +Y.
        A point straight ahead projects to the principal point; a lower point projects lower."""
        rig = P.provisional_rig(1360, 768)
        s = P.PoseSample(0, (1.0, 1.5, 2.0), (0, 0, 0, 1), 'Tracking6Dof')
        Twc = rig.world_from_camera(s)
        np.testing.assert_allclose(Twc[:3, 3], [1.0, 2.0, 1.5], atol=1e-12)
        np.testing.assert_allclose(Twc[:3, 2], [0, 1, 0], atol=1e-12)        # optical axis = W forward
        Tcw = rig.camera_from_world(s)

        def project(pw):
            c = Tcw[:3, :3] @ pw + Tcw[:3, 3]
            return (rig.K @ (c / c[2]))[:2]
        np.testing.assert_allclose(project(np.array([1.0, 6.0, 1.5])), [680, 384], atol=1e-9)
        self.assertGreater(project(np.array([1.0, 6.0, 0.5]))[1], 384)       # below horizon -> lower in image
        self.assertGreater(project(np.array([2.0, 6.0, 1.5]))[0], 680)       # W +X -> image right

    def test_yaw_left_handed_sense(self):
        """Antilatency yaw +90 deg (about +Y, left-handed) turns Z-forward towards +X (right)."""
        q = P.mat_to_quat(P.unity_euler_to_mat(0, math.radians(90), 0))
        rig = P.provisional_rig(640, 480)
        Twc = rig.world_from_camera(P.PoseSample(0, (0, 1, 0), tuple(q), 'Tracking6Dof'))
        np.testing.assert_allclose(Twc[:3, 2], [1, 0, 0], atol=1e-9)

    def test_floor_offset(self):
        rig = P.provisional_rig(640, 480)
        rig.floor_z_m = 0.25
        Twc = rig.world_from_camera(P.PoseSample(0, (0, 1.5, 0), (0, 0, 0, 1), 'Tracking6Dof'))
        self.assertAlmostEqual(Twc[2, 3], 1.25)

    def test_vpcamtracking_profile(self):
        prof = {'name': 'T', 'lens': {'K': [[1000, 0, 680], [0, 1001, 384], [0, 0, 1]], 'dist': [0.1, 0, 0, 0, 0],
                                      'size': [1360, 768], 'rms': 0.3, 'samples': 20},
                'imageCenter': {'dx': 2.0, 'dy': -1.0},
                'nodal': {'T_tracker_cam': P.rt(np.eye(3), [0.0, 0.1, -0.2]).tolist(), 'reprojRmsPx': 1.2, 'samples': 12}}
        rig = P.rig_from_vpcamtracking(prof)
        self.assertEqual(rig.K[0, 2], 682.0)
        self.assertEqual(rig.K[1, 2], 383.0)
        self.assertFalse(rig.qualified)
        np.testing.assert_allclose(rig.T_tracker_cam[:3, 3], [0, 0.1, -0.2])
        np.testing.assert_allclose(rig.scaled_K(680, 384)[0, 0], 500)
        with self.assertRaises(ValueError):
            P.rig_from_vpcamtracking({'name': 'empty', 'lens': None})


class History(unittest.TestCase):
    def make(self, stages=None):
        h = P.PoseHistory(max_gap_ms=60, max_extrapolate_ms=4)
        for i in range(10):
            st = (stages or {}).get(i, 'Tracking6Dof')
            yaw = math.radians(i * 10)
            h.add(P.PoseSample(i * 10_000_000, (i * 0.01, 1.5, 0), tuple(P.mat_to_quat(P.unity_euler_to_mat(0, yaw, 0))), st))
        return h

    def test_interpolates_position_and_rotation(self):
        q = self.make().at(25_000_000)
        self.assertTrue(q.valid)
        self.assertAlmostEqual(q.sample.position[0], 0.025)
        yaw = P.mat_to_unity_euler(P.quat_to_mat(q.sample.rotation))[1]
        self.assertAlmostEqual(math.degrees(yaw), 25.0, places=6)

    def test_rejects_invalid_stage_bracket(self):
        q = self.make({3: 'TrackingBlind6Dof'}).at(25_000_000)
        self.assertFalse(q.valid)
        self.assertEqual(q.reason, 'tracking_stage:TrackingBlind6Dof')

    def test_future_and_expired(self):
        h = self.make()
        self.assertTrue(h.at(92_000_000).valid)                    # within 4 ms of newest
        self.assertEqual(h.at(100_000_000).reason, 'pose_not_yet_available')
        self.assertEqual(h.at(-1).reason, 'pose_expired')

    def test_gap_and_monotonic(self):
        h = P.PoseHistory(max_gap_ms=60)
        h.add(P.PoseSample(0, (0, 0, 0), (0, 0, 0, 1), 'Tracking6Dof'))
        h.add(P.PoseSample(100_000_000, (0, 0, 0), (0, 0, 0, 1), 'Tracking6Dof'))
        self.assertEqual(h.at(50_000_000).reason, 'pose_gap')
        self.assertFalse(h.add(P.PoseSample(100_000_000, (0, 0, 0), (0, 0, 0, 1), 'Tracking6Dof')))
        self.assertEqual(h.rejected_non_monotonic, 1)

    def test_bounded(self):
        h = P.PoseHistory(seconds=0.05)
        for i in range(100):
            h.add(P.PoseSample(i * 1_000_000, (0, 0, 0), (0, 0, 0, 1), 'Tracking6Dof'))
        self.assertLessEqual(len(h.samples), 52)

    def test_parsers(self):
        line = json.dumps({'stage': 'Tracking6Dof', 'q': 0.9, 'tp': [1, 2, 3], 'tq': [0, 0, 0, 1], 'x': 9})
        s = P.parse_alt2freed_json(line, 123)
        self.assertEqual((s.t_ns, s.position, s.clock), (123, (1, 2, 3), 'host_arrival_pipe'))
        self.assertIsNone(P.parse_alt2freed_json('{"event":"tracking"}', 1))
        self.assertIsNone(P.parse_alt2freed_json('not json', 1))
        b = P.parse_bridge_json(json.dumps({'t_ns': 5, 'p': [0, 1, 0], 'q': [0, 0, 0, 1], 'stage': 'Tracking6Dof', 'quality': 1}))
        self.assertEqual((b.t_ns, b.clock), (5, 'sdk_query_monotonic'))


class Quality(unittest.TestCase):
    def stream(self, n=250, noise_m=0.0005, stage='Tracking6Dof', seed=0):
        rng = np.random.default_rng(seed)
        h = P.PoseHistory(seconds=2.0)
        for i in range(n):
            h.add(P.PoseSample(i * 4_000_000, tuple(np.array([0.01 * i * 0.004, 1.2, 0]) + rng.normal(0, noise_m, 3)), (0, 0, 0, 1), stage, 0.027))
        return h, (n - 1) * 4_000_000

    def test_good_tracking_is_full(self):
        h, now = self.stream()
        q, why = P.tracking_quality(h, now + 5_000_000, baseline_value=0.027)
        self.assertGreater(q, 0.95)
        self.assertEqual(why, 'good')

    def test_dropout_stage_and_jitter_lower_the_bar(self):
        h, now = self.stream()
        self.assertEqual(P.tracking_quality(h, now + 300_000_000)[0], 0.0)            # samples stopped (radio dropout)
        h3, now3 = self.stream(stage='Tracking3Dof')
        self.assertEqual(P.tracking_quality(h3, now3)[0], 0.0)
        hn, nown = self.stream(noise_m=0.006)
        q, why = P.tracking_quality(hn, nown)
        self.assertLess(q, 0.2)
        self.assertIn('jitter', why)
        self.assertEqual(P.tracking_quality(P.PoseHistory(), 0)[0], 0.0)


class Render(unittest.TestCase):
    def test_floor_point_lands_where_projected(self):
        rig = P.provisional_rig(640, 360)
        s = P.PoseSample(0, (0.0, 1.6, -3.0), tuple(P.mat_to_quat(P.unity_euler_to_mat(math.radians(15), 0, 0))), 'Tracking6Dof')
        Tcw = rig.camera_from_world(s)
        scene = VS.Scene(blocks=[])
        img = VS.render(Tcw, rig.K, rig.dist, (640, 360), scene)
        self.assertEqual(img.shape, (360, 640, 3))
        # The green Y axis line (x = 0) must cross the projected point of W (0, 2, 0).
        c = Tcw[:3, :3] @ np.array([0, 2.0, 0]) + Tcw[:3, 3]
        u, v = (rig.K @ (c / c[2]))[:2]
        patch = img[int(v) - 2:int(v) + 3, int(u) - 2:int(u) + 3].reshape(-1, 3).astype(int)
        self.assertTrue(((patch[:, 1] - patch[:, 2]) > 40).any(), 'Y axis line missing at projected location')

    def test_looking_down_has_no_sky_and_blocks_render(self):
        rig = P.provisional_rig(320, 240)
        down = P.mat_to_quat(P.unity_euler_to_mat(math.radians(89), 0, 0))
        img = VS.render(rig.camera_from_world(P.PoseSample(0, (1.0, 2.0, 1.7), tuple(down), 'Tracking6Dof')),
                        rig.K, rig.dist, (320, 240))
        self.assertFalse((img == np.array(VS.Scene().sky_top_bgr, np.uint8)).all(axis=2).any())
        self.assertGreater(len(np.unique(img.reshape(-1, 3), axis=0)), 10)

    def test_camera_inside_grid_near_plane_safe(self):
        rig = P.provisional_rig(320, 240)
        img = VS.render(rig.camera_from_world(P.PoseSample(0, (0, 0.01, 0), (0, 0, 0, 1), 'Tracking6Dof')),
                        rig.K, np.array([-0.3, 0.1, 0, 0, 0]), (320, 240))
        self.assertEqual(img.dtype, np.uint8)

    def test_horizon_is_exact_line(self):
        # Regression: a coarse-grid sky/floor split once produced an 8 px horizon staircase.
        import cv2 as _cv
        K = np.array([[1173.6, 0, 679.2], [0, 1178.1, 387.1], [0, 0, 1.0]])
        w, h = 1360, 768
        R0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
        for pitch, roll in ((0, 0), (5, 7), (-5, 15), (2, 40)):
            R = R0 @ _cv.Rodrigues(np.array([math.radians(pitch), 0, 0]))[0] @ _cv.Rodrigues(np.array([0, 0, math.radians(roll)]))[0]
            Tcw = P.inv(P.rt(R, [0, 0, 1.2]))
            img = VS.render(Tcw, K, np.zeros(5), (w, h), VS.Scene(blocks=[], grid_half_extent_m=0.01))
            up = Tcw[:3, :3] @ np.array([0, 0, 1.0])
            ys = np.arange(h)
            for x in range(0, w, 40):
                e = np.stack([(x - K[0, 2]) / K[0, 0] * np.ones(h), (ys - K[1, 2]) / K[1, 1], np.ones(h)], 1) @ up
                exact = np.where(np.diff(np.sign(e)) != 0)[0]
                floor = np.all(img[:, x] == VS.Scene().floor_bgr, axis=1)
                drawn = np.where(np.diff(floor.astype(int)) != 0)[0]
                if len(exact):
                    self.assertTrue(len(drawn) and abs(int(exact[0]) - int(drawn[0])) <= 1, (pitch, roll, x))

    def test_composite_preserves_actor_pixels(self):
        video = RNG.integers(0, 255, (20, 30, 3), dtype=np.uint8)
        bg = np.zeros_like(video)
        alpha = np.zeros((20, 30), np.float32)
        alpha[5:15, 10:20] = 1
        out = VS.composite(video, alpha, bg)
        np.testing.assert_array_equal(out[5:15, 10:20], video[5:15, 10:20])
        self.assertEqual(int(out[0, 0].max()), 0)


class Capture(unittest.TestCase):
    def test_v4l2_abi_sizes(self):
        for name, size in {'v4l2_buffer': 88, 'v4l2_format': 208, 'v4l2_streamparm': 204, 'v4l2_requestbuffers': 20,
                           'v4l2_capability': 104, 'v4l2_fmtdesc': 64}.items():
            self.assertEqual(ctypes.sizeof(getattr(SC, name)), size, name)
        self.assertEqual(SC.VIDIOC_DQBUF, 0xC0585611)
        self.assertEqual(SC.fourcc_str(SC.fourcc('YUYV')), 'YUYV')

    def test_bt709_limited_range(self):
        def pix(y, cb, cr):
            packed = np.array([[[y, cb], [y, cr]]], np.uint8)
            return SC.yuv422_to_bgr(packed)[0, 0].astype(int)
        np.testing.assert_allclose(pix(235, 128, 128), [255, 255, 255], atol=1)
        np.testing.assert_allclose(pix(16, 128, 128), [0, 0, 0], atol=1)
        # BT.709 75% red: Y=51, Cb=109, Cr=212 (8-bit legal)
        np.testing.assert_allclose(pix(51, 109, 212), [0, 0, 191], atol=3)

    def test_colour_bars_vs_scene(self):
        bars = np.zeros((768, 1360, 3), np.uint8)
        cols = [(16, 180, 180), (180, 180, 16), (16, 180, 16), (180, 16, 180), (16, 16, 180), (180, 16, 16), (180, 180, 180), (16, 16, 16)]
        for i, c in enumerate(cols):
            bars[:, i * 170:(i + 1) * 170] = c
        self.assertTrue(SC.colour_bars_score(bars)[0])
        scene = (RNG.random((768, 1360, 3)) * 60).astype(np.uint8)
        scene[300:600, 200:500] = 200
        self.assertFalse(SC.colour_bars_score(scene)[0])
        flat_grey = np.full((768, 1360, 3), 90, np.uint8)
        self.assertFalse(SC.colour_bars_score(flat_grey)[0])

    def test_picture_inset_detects_overlay_feed(self):
        rng = np.random.default_rng(3)
        clean = (rng.random((768, 1360, 3)) * 200 + 20).astype(np.uint8)
        self.assertIsNone(SC.picture_inset(clean))
        overlay = np.zeros_like(clean)
        overlay[37:708, 88:1278] = clean[37:708, 88:1278]                 # picture shrunk into a window
        overlay[5:30, 200:1100] = 230                                      # status text bars
        overlay[715:760, 300:1200] = 200
        x0, y0, x1, y1 = SC.picture_inset(overlay)
        self.assertEqual((x0, x1), (88, 1278))
        dark_left = clean.copy()
        dark_left[:, :120] = rng.integers(0, 6, (768, 120, 3))             # dark scenery on one side only
        self.assertIsNone(SC.picture_inset(dark_left))

    def test_signal_monitor_frozen(self):
        m = SC.SignalMonitor(frozen_after=3)
        img = (RNG.random((90, 160, 3)) * 255).astype(np.uint8)
        states = [m.update(img) for _ in range(5)]
        self.assertEqual(states[0], 'live')
        self.assertEqual(states[-1], 'frozen')


if __name__ == '__main__':
    unittest.main()
