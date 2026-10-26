#!/usr/bin/env python3
"""Moving-camera registration test: does a world-anchored object stay on its real counterpart?

  python tools/rig_registration.py                    # record (countdown + motion prompts) then analyse
  python tools/rig_registration.py --analyse DIR      # re-analyse a recording

The ChArUco board stays FIXED where it was for the nodal recording. Every frame, its corners are
predicted from ALT6 tracking + the rig profile (lens, Alt->camera, board-in-world) and compared with
the corners detected in the RED image. Reported per frame, split by camera angular speed:
  - registration error (px, and degrees = px / fx)
  - best video-vs-tracking delay (the delay minimising error while moving)
  - whether the board moved since calibration (board-in-world re-estimated from still frames)
Outputs in $REMA_DATA/recordings/rig_reg_<stamp>/: red/*.jpg, red.jsonl, alt6.jsonl, plan.json,
registration.json, registration.mp4 (RED | overlay: green detected, magenta predicted, grid).
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import argparse
import datetime
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from rema import nodal as N
from rema import pose as P
from rema import virtual_scene as VS
from rema.video_out import TimelineH264, put_lines

DEFAULT_PROFILE = _P.DATA / 'calibration/rig/RED_VRaptor_Optimo2876_ALT6_rema.json'
SEGMENTS = [   # (seconds, instruction)
    (10, 'SLOW PAN left and right (keep board in frame)'),
    (8, 'SLOW TILT up and down'),
    (10, 'MOVE the tripod / dolly in and out, re-aim at board'),
    (8, 'FASTER pans and tilts'),
    (5, 'HOLD STILL'),
]


def load_profile(path):
    prof = json.loads(Path(path).expanduser().read_text())
    lens, nod = prof['lens'], prof['nodal']
    bc = prof['board']
    board = cv2.aruco.CharucoBoard((int(bc['squaresX']), int(bc['squaresY'])), bc['squareLength'], bc['squareLength'] * bc['markerRatio'],
                                   cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, bc['dictionary'])))
    return {'K': np.array(lens['K'], float), 'dist': np.array(lens['dist'], float), 'X': P.inv(np.array(nod['T_tracker_cam'])),
            'Z': np.array(nod['T_world_board']), 'board': board, 'corners': board.getChessboardCorners().astype(np.float64),
            'rig': P.rig_from_vpcamtracking(prof), 'name': prof.get('name')}


def predict(pr, sample, Z=None):
    T = pr['X'] @ P.inv(P.tracker_rh(sample.position, sample.rotation)) @ (pr['Z'] if Z is None else Z)
    pts = pr['corners'] @ T[:3, :3].T + T[:3, 3]
    if (pts[:, 2] <= 0.05).any():
        return None
    rv, _ = cv2.Rodrigues(T[:3, :3])
    uv, _ = cv2.projectPoints(pr['corners'], rv, T[:3, 3], pr['K'], pr['dist'])
    return uv.reshape(-1, 2)


def detect_board(det, gray, min_corners=12):
    """Plain ChArUco detection, then a fallback for small/dim boards: CLAHE on a 2x upscaled image,
    corners mapped back and sub-pixel refined on the ORIGINAL pixels (no geometry is invented)."""
    cc, ids, _, _ = det.detectBoard(gray)
    if ids is not None and len(ids) >= min_corners:
        return ids.ravel(), cc.reshape(-1, 2).astype(np.float64), 'plain'
    up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    up = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(up)
    cc, ids, _, _ = det.detectBoard(up)
    if ids is None or len(ids) < min_corners:
        return None, None, 'none'
    pts = (cc.reshape(-1, 1, 2) / 2.0).astype(np.float32)
    cv2.cornerSubPix(gray, pts, (3, 3), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return ids.ravel(), pts.reshape(-1, 2).astype(np.float64), 'upscaled_clahe'


# ---------------------------------------------------------------------------- record

def record(a):
    from rema.qt_env import ensure_xcb_cursor
    ensure_xcb_cursor()
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QApplication, QLabel
    sys.path.insert(0, str(_P.ROOT / 'tools'))
    from rig_live import CaptureThread
    from rig_record import FrameWriter

    pr = load_profile(a.profile)
    det = cv2.aruco.CharucoDetector(pr['board'])
    out = _P.DATA / 'recordings' / ('rig_reg_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    (out / 'red').mkdir(parents=True)
    hist = P.PoseHistory(seconds=600, max_gap_ms=60)
    lock = threading.Lock()
    alt_log = (out / 'alt6.jsonl').open('w')
    bridge = subprocess.Popen([str(_P.WORK / 'alt_pose_bridge'), '--rate', '250', '--sdk-lib', str(_P.WORK / 'lib'), '--serial', a.alt_serial],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def read_bridge():
        for line in bridge.stdout:
            alt_log.write(line)
            s = P.parse_bridge_json(line)
            if s:
                with lock:
                    hist.add(s)
    threading.Thread(target=read_bridge, daemon=True).start()
    cap = CaptureThread(a.device)
    cap.start()
    writer = FrameWriter(out)
    writer.start()

    t0 = time.monotonic_ns() + int(1.5e9)
    plan, t = [{'phase': 'intro', 'start_ns': t0, 'end_ns': t0 + int(a.intro * 1e9), 'instruction': 'GET READY - board fixed, as in calibration'}], t0 + int(a.intro * 1e9)
    for secs, text in SEGMENTS:
        plan.append({'phase': 'motion', 'start_ns': t, 'end_ns': t + int(secs * 1e9), 'instruction': text})
        t += int(secs * 1e9)
    end_ns = t
    (out / 'plan.json').write_text(json.dumps({'phases': plan, 'profile': str(a.profile), 'alt_serial': a.alt_serial,
                                                'video_delay_ms_live': a.video_delay_ms}, indent=2))
    app = QApplication(sys.argv)
    label = QLabel()
    label.setWindowTitle('REMA registration test  [Esc] abort')
    label.setAlignment(Qt.AlignCenter)
    label.resize(1500, 860)
    label.show()
    st = {'seq': None, 'abort': False}

    def tick():
        now = time.monotonic_ns()
        if now >= end_ns or st['abort']:
            app.quit()
            return
        f = cap.latest
        if f is None or f.sequence == st['seq']:
            return
        st['seq'] = f.sequence
        cur = next((p for p in plan if p['start_ns'] <= now < p['end_ns']), {'phase': 'lead', 'instruction': 'starting', 'end_ns': t0})
        writer.put((f, 'hold' if a.lossless else 'move', 0))
        img = f.image_bgr.copy()
        with lock:
            q = hist.at(f.kernel_ns - int(a.video_delay_ms * 1e6))
        if q.valid:
            uv = predict(pr, q.sample)
            if uv is not None:
                c = uv.reshape(pr['board'].getChessboardSize()[1] - 1, pr['board'].getChessboardSize()[0] - 1, 2)
                outline = np.array([c[0, 0], c[0, -1], c[-1, -1], c[-1, 0]])
                cv2.polylines(img, [np.rint(outline).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
        remain = max(0.0, (cur['end_ns'] - now) / 1e9)
        text = f"{cur['instruction']}   {remain:0.0f}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 70), (20, 20, 20), -1)
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 220, 255), 3, cv2.LINE_AA)
        put_lines(img, ['magenta = board predicted from ALT6 tracking (live preview)', 'tracking: ' + (q.reason if not q.valid else 'ok')],
                  origin=(16, 110), scale=0.8, colours=[(255, 0, 255), (80, 220, 80) if q.valid else (60, 60, 255)])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qi).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def key(ev):
        if ev.key() == Qt.Key_Escape:
            st['abort'] = True
    label.keyPressEvent = key
    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(10)
    print(f'recording {(end_ns - t0) / 1e9 + 1.5:.0f} s to {out}', flush=True)
    try:
        app.exec()
    finally:
        label.close()
        app.processEvents()
        print('recording finished - analysing (about 1-2 minutes)', flush=True)
        cap.stop_event.set()
        writer.put(None)
        writer.join(timeout=60)
        try:
            bridge.stdin.close()
            bridge.wait(timeout=3)
        except Exception:
            bridge.terminate()
        alt_log.close()
    (out / 'meta.json').write_text(json.dumps({'aborted': st['abort'], 'writer_dropped_frames': writer.dropped,
                                               'driver_dropped_frames': cap.cap.dropped_by_driver if cap.cap else None}, indent=2))
    if not st['abort']:
        analyse(out, a.profile)


# ---------------------------------------------------------------------------- analyse

def analyse(rec, profile=None):
    rec = Path(rec)
    plan = json.loads((rec / 'plan.json').read_text())
    pr = load_profile(profile or plan['profile'])
    det = cv2.aruco.CharucoDetector(pr['board'])
    hist = P.PoseHistory(seconds=1e6, max_gap_ms=60)
    for line in (rec / 'alt6.jsonl').read_text().splitlines():
        s = P.parse_bridge_json(line)
        if s:
            hist.add(s)
    frames = [json.loads(l) for l in (rec / 'red.jsonl').read_text().splitlines()]
    obs = []
    for f in frames:
        ids, img_pts, how = detect_board(det, cv2.imread(str(rec / f['file']), cv2.IMREAD_GRAYSCALE))
        if ids is not None:
            obs.append({'f': f, 'ids': ids, 'img': img_pts, 'how': how})
    if not obs:
        raise SystemExit('board never detected')
    times = np.array([s.t_ns for s in hist.samples])
    Rs = [P.quat_to_mat(s.rotation) for s in hist.samples]

    def ang_speed(t_ns, win=0.04e9):
        i, j = np.searchsorted(times, [t_ns - win, t_ns + win])
        j = min(j, len(times) - 1)
        if j <= i or times[j] == times[i]:
            return float('nan')
        return N.rotation_angle_deg(Rs[i].T @ Rs[j]) / ((times[j] - times[i]) / 1e9)

    for o in obs:
        o['speed'] = ang_speed(o['f']['kernel_ns'])

    def errors(delay_ms, Z=None, subset=None):
        e = []
        for o in (subset or obs):
            q = hist.at(o['f']['kernel_ns'] - int(delay_ms * 1e6))
            if not q.valid:
                e.append(np.nan)
                continue
            uv = predict(pr, q.sample, Z)
            e.append(np.nan if uv is None else float(np.sqrt(np.mean(np.sum((uv[o['ids']] - o['img']) ** 2, 1)))))
        return np.array(e)

    # 1) did the board move since calibration? re-estimate board-in-world from still frames
    still = [o for o in obs if o['speed'] < 2.0]
    Zs = []
    for o in still[::3]:
        B, info = N.board_pose(pr['corners'][o['ids']], o['img'], pr['K'], pr['dist'])
        q = hist.at(o['f']['kernel_ns'])
        if B is not None and q.valid and not info['ambiguous']:
            Zs.append(P.tracker_rh(q.sample.position, q.sample.rotation) @ P.inv(pr['X']) @ B)
    board_shift = None
    Z_now = None
    if Zs:
        Z_now = P.rt(N._chordal_mean(np.array([z[:3, :3] for z in Zs])), np.median([z[:3, 3] for z in Zs], axis=0))
        board_shift = {'mm': float(np.linalg.norm(Z_now[:3, 3] - pr['Z'][:3, 3]) * 1000),
                       'deg': N.rotation_angle_deg(Z_now[:3, :3].T @ pr['Z'][:3, :3])}
    moved = board_shift is not None and (board_shift['mm'] > 20 or board_shift['deg'] > 1.0)
    Zuse = Z_now if moved else None

    # 2) video delay: minimise median error over MOVING frames
    moving = [o for o in obs if o['speed'] >= 10.0]
    delays = list(range(-40, 201, 10))
    curve = {d: float(np.nanmedian(errors(d, Zuse, moving))) if moving else float('nan') for d in delays}
    best = min(curve, key=lambda d: curve[d] if np.isfinite(curve[d]) else 1e9) if moving else 0
    fine = {d: float(np.nanmedian(errors(d, Zuse, moving))) for d in range(best - 10, best + 11, 2)} if moving else {}
    if fine:
        best = min(fine, key=fine.get)
    e = errors(best, Zuse)
    sp = np.array([o['speed'] for o in obs])
    fx = pr['K'][0, 0]

    def stats(mask):
        v = e[mask & np.isfinite(e)]
        return None if not len(v) else {'frames': int(len(v)), 'px_p50': round(float(np.median(v)), 2), 'px_p95': round(float(np.percentile(v, 95)), 2),
                                         'deg_p50': round(float(np.degrees(np.median(v) / fx)), 3)}
    report = {
        'recording': str(rec), 'profile': str(profile or plan['profile']), 'frames': len(frames), 'board_frames': len(obs),
        'board_frames_by_detector': {h: sum(o['how'] == h for o in obs) for h in ('plain', 'upscaled_clahe')},
        'board_moved_since_calibration': board_shift, 'board_pose_used': 're-estimated (board moved)' if moved else 'from nodal calibration',
        'video_delay_ms_best': best, 'delay_curve_moving_px_p50': {str(k): round(v, 2) for k, v in curve.items()},
        'still_lt2dps': stats(sp < 2), 'slow_2_15dps': stats((sp >= 2) & (sp < 15)), 'medium_15_40dps': stats((sp >= 15) & (sp < 40)),
        'fast_ge40dps': stats(sp >= 40), 'all': stats(np.ones_like(sp, bool)),
        'error_at_zero_delay_moving_px_p50': round(curve.get(0, float('nan')), 2),
        'scope': 'Board-corner registration on a fixed board used in the nodal fit (new camera poses, same board). Software timing; not glass-to-glass latency.',
    }
    (rec / 'registration.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # 3) review video
    vid = None
    byseq = {o['f']['sequence']: (o, err) for o, err in zip(obs, e)}
    for f in frames:
        img = cv2.imread(str(rec / f['file']))
        if img is None:
            continue
        right = img.copy()
        q = hist.at(f['kernel_ns'] - int(best * 1e6))
        if q.valid:
            Tcw = pr['rig'].camera_from_world(q.sample)
            right = VS.overlay(right, VS.render(Tcw, pr['K'], pr['dist'], (img.shape[1], img.shape[0])), 0.35)
            uv = predict(pr, q.sample, Zuse)
            if uv is not None:
                for x, y in uv:
                    cv2.circle(right, (int(x), int(y)), 4, (255, 0, 255), 1, cv2.LINE_AA)
        o = byseq.get(f['sequence'])
        if o:
            for x, y in o[0]['img']:
                cv2.circle(right, (int(x), int(y)), 2, (0, 255, 0), -1)
        speed = ang_speed(f['kernel_ns'])
        put_lines(img, ['RED via Osprey (recorded)', f"seq {f['sequence']}"], scale=0.7)
        put_lines(right, ['green: detected corners   magenta: predicted from ALT6 tracking',
                          f"registration error {o[1]:.1f} px ({np.degrees(o[1] / fx):.2f} deg)" if o and np.isfinite(o[1]) else 'board not detected',
                          f'camera speed {speed:.0f} deg/s   video delay {best} ms', 'calibration NOT QUALIFIED'], scale=0.7,
                  colours=[None, (255, 0, 255), None, (0, 200, 255)])
        fr = np.hstack([img, right])
        if vid is None:
            vid = TimelineH264(rec / 'registration.mp4', fr.shape[1], fr.shape[0])
        vid.write(fr, f['kernel_ns'])
    if vid:
        vid.close()
    print('video:', rec / 'registration.mp4')
    from rema.video_out import notify
    st_, mv = report.get('still_lt2dps') or {}, report.get('slow_2_15dps') or {}
    notify('REMA registration test finished',
           f"still {st_.get('px_p50', '-')} px, slow motion {mv.get('px_p50', '-')} px, delay {report['video_delay_ms_best']} ms. Details in the terminal; video registration.mp4")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', default=str(DEFAULT_PROFILE))
    ap.add_argument('--analyse', default='')
    ap.add_argument('--device', default='Osprey', help='capture device: name fragment (default Osprey) or /dev/videoN')
    ap.add_argument('--alt-serial', default='4C0046001951383039383438')
    ap.add_argument('--intro', type=float, default=10.0)
    ap.add_argument('--video-delay-ms', type=float, default=80.0, help='live preview only (measured 80 ms on 2026-10-24); analysis re-measures it')
    ap.add_argument('--jpeg', dest='lossless', action='store_false', help='JPEG q85 frames instead of lossless PNG')
    a = ap.parse_args()
    if not _P.disk_reserve_ok():
        sys.exit('25 GiB free-space reserve failed')
    if a.analyse:
        analyse(Path(a.analyse).expanduser(), a.profile)
    else:
        record(a)


if __name__ == '__main__':
    main()
