#!/usr/bin/env python3
"""Guided nodal (Alt -> camera) calibration on the live RED feed + Antilatency bridge.

  python tools/rig_nodal.py --profile ~/Desktop/VPcamtracking/alt2freed/calib/profiles/<name>.json

Lens (K, distortion) and board come from the VPcamtracking profile. Fix the board in the room,
then move the camera between positions and hold still: a sample is taken automatically when the
board is detected, the image and tracker are both still, and the position is new.

Keys: [Space] capture now (still checks apply)  [S] solve  [D] delete last  [Q] quit.
Every sample (tracker pose, corners, frame) is saved, so a session can be re-solved later with
--resume DIR --solve-only. After a solve, a magenta outline predicted from TRACKING ONLY is drawn
on the board: it should stay locked to the printed board as the camera moves.
Output: $REMA_DATA/calibration/rig/nodal_<stamp>/ (samples + result.json) and a REMA rig profile
$REMA_DATA/calibration/rig/<profile>_rema.json usable with tools/rig_live.py --profile.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from rema import nodal as N
from rema import pose as P
from rema.video_out import put_lines

STILL_WINDOW_S = 0.6
STILL_POS_MM = 5.0   # stillness detector; ALT6 still noise ~1 mm std, 2.4 mm peak (2026-10-24)
STILL_ROT_DEG = 0.15
IMAGE_STILL_PX = 0.5
NOVEL_ROT_DEG = 5.0
NOVEL_POS_M = 0.08
MIN_CORNERS = 20


def make_board(cfg):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg['dictionary']))
    sq = float(cfg['squareLength'])
    return cv2.aruco.CharucoBoard((int(cfg['squaresX']), int(cfg['squaresY'])), sq, sq * float(cfg['markerRatio']), d)


class Session:
    def __init__(self, profile_path, out_dir):
        self.profile_path = Path(profile_path).expanduser()
        self.profile = json.loads(self.profile_path.read_text())
        self.rig = P.rig_from_vpcamtracking(self.profile)
        self.board = make_board(self.profile['board'])
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.corners_all = self.board.getChessboardCorners().astype(np.float64)
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.samples = []
        self.result = None
        self.prev_corners = {}
        self.still_frames = 0
        self.message = 'Fix the board in the room. Move the camera, then hold still.'

    # ---------------------------------------------------------------- persistence
    def load(self):
        f = self.out / 'samples.jsonl'
        if f.exists():
            for line in f.read_text().splitlines():
                d = json.loads(line)
                self.samples.append({'A': np.array(d['A']), 'B': np.array(d['B']), 'obj': np.array(d['obj']),
                                     'img': np.array(d['img']), 'ambiguous': d['ambiguous'], 'meta': d['meta']})
        rf = self.out / 'result.json'
        if rf.exists():
            self.result = json.loads(rf.read_text())

    def save_sample(self, s, frame):
        i = len(self.samples)
        cv2.imwrite(str(self.out / f'frame_{i:03d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with (self.out / 'samples.jsonl').open('a') as f:
            f.write(json.dumps({'A': s['A'].tolist(), 'B': s['B'].tolist(), 'obj': s['obj'].tolist(),
                                'img': s['img'].tolist(), 'ambiguous': bool(s['ambiguous']), 'meta': s['meta']}) + '\n')

    def rewrite_samples(self):
        with (self.out / 'samples.jsonl').open('w') as f:
            for s in self.samples:
                f.write(json.dumps({'A': s['A'].tolist(), 'B': s['B'].tolist(), 'obj': s['obj'].tolist(),
                                    'img': s['img'].tolist(), 'ambiguous': bool(s['ambiguous']), 'meta': s['meta']}) + '\n')

    # ---------------------------------------------------------------- per frame
    def detect(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cc, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < MIN_CORNERS:
            return None
        ids = ids.ravel()
        img = cc.reshape(-1, 2).astype(np.float64)
        return {'ids': ids, 'img': img, 'obj': self.corners_all[ids]}

    def image_still(self, det):
        if det is None:
            self.prev_corners, self.still_frames = {}, 0
            return False
        cur = dict(zip(det['ids'].tolist(), det['img']))
        common = [k for k in cur if k in self.prev_corners]
        moved = np.mean([np.linalg.norm(cur[k] - self.prev_corners[k]) for k in common]) if len(common) >= 10 else 99
        self.prev_corners = cur
        self.still_frames = self.still_frames + 1 if moved < IMAGE_STILL_PX else 0
        return self.still_frames >= 3

    @staticmethod
    def tracker_still(history, t_ns):
        win = [s for s in history.samples if t_ns - STILL_WINDOW_S * 1e9 <= s.t_ns <= t_ns + 5e6]
        if len(win) < 10:
            return None, 'waiting for tracker samples'
        if any(s.stage != 'Tracking6Dof' for s in win):
            return None, 'tracker not in Tracking6Dof'
        Pp = np.array([s.position for s in win])
        spread = np.linalg.norm(Pp - Pp.mean(0), axis=1).max() * 1000
        R0 = P.quat_to_mat(win[0].rotation)
        rot = max(N.rotation_angle_deg(R0.T @ P.quat_to_mat(s.rotation)) for s in win)
        if spread > STILL_POS_MM or rot > STILL_ROT_DEG:
            return None, f'camera moving ({spread:.1f} mm, {rot:.2f} deg)'
        q = np.array([s.rotation for s in win])
        q[(q @ q[0]) < 0] *= -1
        qm = q.mean(0)
        qm /= np.linalg.norm(qm)
        return P.tracker_rh(Pp.mean(0), qm), 'still'

    def novel(self, A):
        for s in self.samples:
            if N.rotation_angle_deg(s['A'][:3, :3].T @ A[:3, :3]) < NOVEL_ROT_DEG and \
                    np.linalg.norm(s['A'][:3, 3] - A[:3, 3]) < NOVEL_POS_M:
                return False
        return True

    def try_capture(self, frame, det, history, force=False):
        if det is None:
            return 'board not detected (need %d+ corners)' % MIN_CORNERS
        if not self.image_still(det):
            return 'hold the camera still'
        A, why = self.tracker_still(history, frame.kernel_ns)
        if A is None:
            return why
        B, info = N.board_pose(det['obj'], det['img'], self.rig.K, self.rig.dist)
        if B is None:
            return 'board pose rejected: ' + info['reason']
        if not force and not self.novel(A):
            return 'already have this position: move/rotate to a new one'
        s = {'A': A, 'B': B, 'obj': det['obj'], 'img': det['img'], 'ambiguous': info['ambiguous'],
             'meta': {'sequence': frame.sequence, 'kernel_ns': frame.kernel_ns, 'corners': int(len(det['ids'])),
                      'pnp_rms_px': info['rms_px'], 'ambiguity_ratio': info['ambiguity_ratio'],
                      'time': datetime.datetime.now().isoformat(timespec='seconds')}}
        self.save_sample(s, frame.image_bgr)
        self.samples.append(s)
        self.prev_corners, self.still_frames = {}, 0
        return f'CAPTURED sample {len(self.samples)}' + (' (ambiguous board pose: used for refinement only)' if info['ambiguous'] else '')

    def diversity(self):
        if len(self.samples) < 2:
            return None
        return N.rotation_diversity([s['A'][:3, :3] for s in self.samples])

    def solve(self):
        r = N.solve_nodal([dict(s) for s in self.samples], self.rig.K, self.rig.dist)
        r['created'] = datetime.datetime.now().astimezone().isoformat()
        r['profile_source'] = str(self.profile_path)
        r['lens'] = {'K': self.rig.K.tolist(), 'dist': self.rig.dist.tolist(), 'size': list(self.rig.size)}
        r['board'] = self.profile['board']
        r['scope'] = 'Tracking-predicted board reprojection on still samples. Not a registration/latency qualification.'
        (self.out / 'result.json').write_text(json.dumps(r, indent=2))
        rig_profile = {'name': self.profile.get('name', 'rig') + '_rema', 'info': self.profile.get('info'),
                       'board': self.profile['board'], 'lens': self.profile['lens'],
                       'imageCenter': {'dx': 0.0, 'dy': 0.0},   # already folded into rig.K
                       'nodal': {'T_tracker_cam': r['T_tracker_cam'], 'T_world_board': r['T_world_board'],
                                 'reprojRmsPx': r['heldout_rms_px_median_max'][0] if r['heldout_rms_px_median_max'] else r['fit_rms_px_median_max'][0],
                                 'samples': len(r['fit_samples']), 'session': str(self.out), 'solver': 'rema.nodal'}}
        rig_profile['lens'] = dict(rig_profile['lens'], K=self.rig.K.tolist())
        safe = ''.join(c for c in self.profile.get('name', 'rig') if c.isalnum() or c in '-_.') or 'rig'
        dest = _P.DATA / 'calibration' / 'rig' / (safe + '_rema.json')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rig_profile, indent=2))
        r['rig_profile'] = str(dest)
        self.result = r
        return r

    # ---------------------------------------------------------------- drawing
    def draw(self, frame, det, status, history):
        img = frame.image_bgr.copy()
        if det is not None:
            for x, y in det['img']:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
        if self.result:
            q = history.at(frame.kernel_ns)
            if q.valid:
                X = P.inv(np.array(self.result['T_tracker_cam']))
                Z = np.array(self.result['T_world_board'])
                T = X @ P.inv(P.tracker_rh(q.sample.position, q.sample.rotation)) @ Z
                c = self.corners_all
                outline = np.array([c[:, :2].min(0), [c[:, 0].max(), c[:, 1].min()], c[:, :2].max(0), [c[:, 0].min(), c[:, 1].max()]])
                pts3 = np.c_[outline, np.zeros(4)]
                if ((pts3 @ T[:3, :3].T + T[:3, 3])[:, 2] > 0.05).all():
                    rv, _ = cv2.Rodrigues(T[:3, :3])
                    uv, _ = cv2.projectPoints(pts3, rv, T[:3, 3], self.rig.K, self.rig.dist)
                    cv2.polylines(img, [np.rint(uv.reshape(-1, 2)).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
        div = self.diversity()
        lines = [f'REMA nodal calibration  samples {len(self.samples)}  (need ~12-20)',
                 'status: ' + status,
                 ('rotation variety: ' + (f"max {div['max_relative_deg']:.0f} deg, axis spread {div['axis_spread']:.2f} - " + div['advice'])) if div else 'rotation variety: -',
                 '[Space] capture  [S] solve  [D] delete last  [Q] quit']
        colours = [None, (80, 220, 80) if status.startswith('CAPTURED') else (0, 200, 255), (80, 220, 80) if div and div['ok'] else (0, 200, 255), None]
        if self.result:
            r = self.result
            ho = r['heldout_rms_px_median_max']
            lines.append(f"SOLVED: held-out reprojection median {ho[0]:.2f} px (max {ho[1]:.2f})" if ho else 'SOLVED (no held-out samples)')
            lines.append(f"offset {r['offset_m'] * 100:.1f} cm, camera vs Alt {r['camera_axis_vs_tracker_deg']:.1f} deg, dropped {len(r['dropped_outliers'])}. Magenta = board predicted from tracking")
            colours += [(255, 0, 255), (255, 0, 255)]
        put_lines(img, lines, scale=0.6, colours=colours)
        return img


def run(a):
    from rema.qt_env import ensure_xcb_cursor
    ensure_xcb_cursor()
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QApplication, QLabel
    sys.path.insert(0, str(_P.ROOT / 'tools'))
    from rig_live import CaptureThread, SubprocessPose
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(a.resume).expanduser() if a.resume else _P.DATA / 'calibration' / 'rig' / f'nodal_{stamp}'
    ses = Session(a.profile, out)
    ses.load()
    if a.solve_only:
        print(json.dumps(ses.solve(), indent=2))
        return
    history = P.PoseHistory(seconds=3.0, max_gap_ms=60)
    pose = SubprocessPose(history, [str(_P.WORK / 'alt_pose_bridge'), '--rate', '250', '--sdk-lib', str(_P.WORK / 'lib'),
                                    '--serial', a.alt_serial], P.parse_bridge_json)
    cap = CaptureThread(a.device)
    pose.start()
    cap.start()
    app = QApplication(sys.argv)
    label = QLabel()
    label.setWindowTitle('REMA nodal calibration  [Space] capture  [S] solve  [D] delete last  [Q] quit')
    label.setAlignment(Qt.AlignCenter)
    label.resize(1400, 800)
    label.show()
    st = {'seq': None, 'status': ses.message, 'force': False, 'deadline': time.monotonic() + a.seconds}

    def tick():
        if time.monotonic() > st['deadline']:
            app.quit()
            return
        if cap.error:
            st['status'] = 'capture error: ' + cap.error
        f = cap.latest
        if f is None or f.sequence == st['seq']:
            return
        st['seq'] = f.sequence
        det = ses.detect(f.image_bgr)
        with pose.lock:
            msg = ses.try_capture(f, det, history, force=st['force']) if (a.auto or st['force']) else \
                ('ready - press Space' if det is not None else 'board not detected')
        st['force'] = False
        if msg.startswith('CAPTURED') or not st['status'].startswith('CAPTURED') or time.monotonic() - st.get('cap_t', 0) > 2:
            if msg.startswith('CAPTURED'):
                st['cap_t'] = time.monotonic()
                div = ses.diversity()
                print(time.strftime('%H:%M:%S'), msg, '| corners', len(det['ids']),
                      '| variety:', (f"{div['max_relative_deg']:.0f} deg, spread {div['axis_spread']:.2f}, {div['advice']}" if div else '-'), flush=True)
            st['status'] = msg
        if time.monotonic() - st.get('log_t', 0) > 5:
            st['log_t'] = time.monotonic()
            print(time.strftime('%H:%M:%S'), 'status:', st['status'], '| tracker:', pose.status, '| samples', len(ses.samples), flush=True)
        with pose.lock:
            img = ses.draw(f, det, st['status'] + ('' if pose.count else '  [tracker: ' + pose.status + ']'), history)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def key(ev):
        k = ev.text().lower()
        if k == 'q':
            app.quit()
        elif k == ' ':
            st['force'] = True
        elif k == 'd' and ses.samples:
            ses.samples.pop()
            ses.rewrite_samples()
            st['status'] = f'deleted last sample ({len(ses.samples)} left)'
        elif k == 's':
            try:
                r = ses.solve()
                st['status'] = 'SOLVED - saved ' + r['rig_profile']
                print(json.dumps({k: r[k] for k in ('fit_rms_px_median_max', 'heldout_rms_px_median_max', 'offset_m',
                                                    'camera_axis_vs_tracker_deg', 'dropped_outliers', 'rig_profile')}, indent=2), flush=True)
            except ValueError as e:
                st['status'] = 'cannot solve: ' + str(e)
                print('cannot solve:', e, flush=True)
    label.keyPressEvent = key
    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(10)
    try:
        app.exec()
    finally:
        cap.stop_event.set()
        pose.stop()
    print('session', out, 'samples', len(ses.samples))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--device', default='Osprey', help='capture device: name fragment (default Osprey) or /dev/videoN')
    ap.add_argument('--alt-serial', default='4C0046001951383039383438')
    ap.add_argument('--resume', default='')
    ap.add_argument('--solve-only', action='store_true')
    ap.add_argument('--manual', dest='auto', action='store_false', help='capture only on Space')
    ap.add_argument('--seconds', type=float, default=3600)
    a = ap.parse_args()
    if not _P.disk_reserve_ok():
        sys.exit('25 GiB free-space reserve failed')
    (_P.DATA / 'calibration' / 'rig').mkdir(parents=True, exist_ok=True)
    run(a)


if __name__ == '__main__':
    main()
