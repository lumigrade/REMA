"""rig_live engine policy on tracking loss (fake camera + fake tracker; no hardware)."""
import argparse
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import rig_live as RL
from rema import pose as P
from rema.sdi_capture import SdiFrame


class FakeCapture:
    def __init__(self):
        self.seq, self.error, self.desc = 0, None, {'card': 'Osprey VB-USL Video Bridge (fake)'}
        rng = np.random.default_rng(0)
        self.img = (rng.random((96, 160, 3)) * 255).astype(np.uint8)   # textured, so the signal monitor says 'live'

    def next_frame(self, after_seq, timeout=1.0):
        self.seq += 1
        return SdiFrame(self.img, self.seq, time.monotonic_ns(), time.monotonic_ns(), 'monotonic_soe', False, 160, 96, 'YUYV')


class FakePose:
    def __init__(self):
        self.valid = True
        self.lock = __import__('threading').Lock()
        self.count = 1
        self.status = 'fake'

    def query(self, t_ns):
        s = P.PoseSample(t_ns, (0.0, 1.2, -2.0), (0, 0, 0, 1), 'Tracking6Dof', 0.027)
        return P.PoseQuery(True, 'interpolated', s, 4.0) if self.valid else P.PoseQuery(False, 'pose_not_yet_available', s)

    def stop(self):
        pass


def make_engine(policy):
    a = argparse.Namespace(device='Osprey', pose='none', alt2freed='', alt_serial='', freed_port=0, freed_sink_port=0, trust_freed=False,
                           profile='', hfov=60.0, floor_z=0.0, video_delay_ms=0.0, pose_wait_ms=0.0, max_pose_gap_ms=60.0, matte='none',
                           matte_ratio=0.5, view='composite', overlay_opacity=0.5, background='grid', scene='', scene_align='',
                           tracking_loss=policy)
    e = RL.Engine(a)
    e.capture = FakeCapture()
    e.pose = FakePose()
    e.side_by_side = False
    return e


class TrackingLoss(unittest.TestCase):
    def run_sequence(self, policy):
        e = make_engine(policy)
        reasons = []
        for i in range(12):
            e.pose.valid = not (4 <= i < 9)          # tracking lost for frames 4..8
            out, tele = e.step()
            reasons.append(tele['pose_reason'])
        return e, reasons

    def test_hold_keeps_compositing_and_flags(self):
        e, reasons = self.run_sequence('hold')
        self.assertEqual(reasons[:4], ['ok'] * 4)
        self.assertTrue(all(r.startswith('held:') for r in reasons[4:9]), reasons)
        self.assertEqual(reasons[9:], ['ok'] * 3)
        self.assertEqual(e.stats['held'], 5)
        self.assertEqual(e.stats['rendered'], 12)
        lines, _ = e.info_lines(reasons[5], {'pose_source': None, 'pose_clock': None, 'matte_ms': 0}, e.rig)
        self.assertIn('TRACKING LOST', lines[0])

    def test_raw_policy_falls_back(self):
        e, reasons = self.run_sequence('raw')
        self.assertEqual(reasons[4:9], ['pose_not_yet_available'] * 5)
        self.assertEqual(e.stats['rendered'], 7)
        self.assertEqual(e.stats['fallback'].get('pose_not_yet_available'), 5)

    def test_no_pose_yet_cannot_hold(self):
        e = make_engine('hold')
        e.pose.valid = False
        out, tele = e.step()
        self.assertEqual(tele['pose_reason'], 'pose_not_yet_available')   # nothing to hold before the first valid pose


if __name__ == '__main__':
    unittest.main()
