#!/usr/bin/env python3
"""Solve the nodal (Alt -> camera) offset from a tools/rig_record.py recording.

  python tools/rig_nodal_offline.py ~/rema-data/recordings/rig_rec_<stamp> [--lens lens.json] [--no-video]

For each HOLD phase: skip the first 0.8 s (settling), require the tracker still (<1.5 mm, <0.15 deg,
all Tracking6Dof) and the board corners still (<0.5 px) across the remaining lossless frames, then
average corners and tracker pose into one sample. Every rejected hold is reported with its reason.
Writes nodal_result.json, a REMA rig profile ($REMA_DATA/calibration/rig/<name>_rema.json) and a
labeled H.264 review: RED frame | detected corners + board outline predicted from TRACKING ONLY.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import argparse
import datetime
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from rema import nodal as N
from rema import pose as P
from rema.video_out import TimelineH264, put_lines

# Stillness detector (not an accuracy gate): ALT6 still noise measured 2026-10-24 ~1 mm std, 2.4 mm peak,
# 0.05 deg, so 1.5 mm misread a still camera as moving. Averaging ~400 samples per hold removes the noise.
SETTLE_S, STILL_POS_MM, STILL_ROT_DEG, STILL_PX, MIN_CORNERS = 0.8, 5.0, 0.15, 0.5, 20
MIN_TAIL_FRAMES = 5   # >= ~0.33 s at the 14.8 fps the Osprey delivers


def load_alt(path):
    h = P.PoseHistory(seconds=1e6, max_gap_ms=60)
    for line in Path(path).read_text().splitlines():
        s = P.parse_bridge_json(line)
        if s:
            h.add(s)
    return h


def make_board(bc):
    return cv2.aruco.CharucoBoard((int(bc['squaresX']), int(bc['squaresY'])), bc['squareLength'], bc['squareLength'] * bc['markerRatio'],
                                  cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, bc['dictionary'])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recording')
    ap.add_argument('--lens', default='')
    ap.add_argument('--no-video', action='store_true')
    a = ap.parse_args()
    rec = Path(a.recording).expanduser()
    plan = json.loads((rec / 'plan.json').read_text())
    prof = json.loads(Path(plan['profile']).expanduser().read_text())
    lens_path = Path(a.lens or plan.get('lens') or '')
    lens = json.loads(lens_path.read_text()) if lens_path.is_file() else prof['lens']
    K, dist = np.array(lens['K'], float), np.array(lens['dist'], float)
    board = make_board(prof['board'])
    det = cv2.aruco.CharucoDetector(board)
    corners_all = board.getChessboardCorners().astype(np.float64)
    hist = load_alt(rec / 'alt6.jsonl')
    frames = [json.loads(l) for l in (rec / 'red.jsonl').read_text().splitlines()]

    samples, report = [], []
    for ph in [p for p in plan['phases'] if p['phase'] == 'hold']:
        t0, t1 = ph['start_ns'] + int(SETTLE_S * 1e9), ph['end_ns']
        fs = [f for f in frames if t0 <= f['kernel_ns'] < t1 and f['file'].endswith('.png')]
        row = {'position': ph['position'] + 1, 'frames': len(fs)}
        report.append(row)
        if len(fs) < 3:
            row['result'] = 'too few frames in hold'
            continue
        win = [s for s in hist.samples if t0 <= s.t_ns <= t1]
        if len(win) < 20 or any(s.stage != 'Tracking6Dof' for s in win):
            row['result'] = 'tracker not in Tracking6Dof for the whole hold'
            continue
        Pp = np.array([s.position for s in win])
        spread = float(np.linalg.norm(Pp - Pp.mean(0), axis=1).max() * 1000)
        R0 = P.quat_to_mat(win[0].rotation)
        rot = max(N.rotation_angle_deg(R0.T @ P.quat_to_mat(s.rotation)) for s in win)
        row['tracker_spread_mm'], row['tracker_rot_deg'] = round(spread, 2), round(rot, 3)
        if spread > STILL_POS_MM or rot > STILL_ROT_DEG:
            row['result'] = 'camera moved during hold'
            continue
        dets = []
        for f in fs:
            cc, ids, _, _ = det.detectBoard(cv2.imread(str(rec / f['file']), cv2.IMREAD_GRAYSCALE))
            if ids is not None and len(ids) >= MIN_CORNERS:
                dets.append(dict(zip(ids.ravel().tolist(), cc.reshape(-1, 2).astype(np.float64))))
        row['board_frames'] = len(dets)
        if len(dets) < 3:
            row['result'] = 'board not detected (need %d+ corners)' % MIN_CORNERS
            continue
        # Use the longest SETTLED TAIL of the hold: the image may still drift early in a hold (e.g. lens
        # stabilisation re-centring) while the tracker is already still. Stillness limits are unchanged.
        common = stack = None
        for start in range(0, len(dets) - MIN_TAIL_FRAMES + 1):
            tail = dets[start:]
            cm = sorted(set.intersection(*[set(d) for d in tail]))
            if len(cm) < MIN_CORNERS:
                continue
            st_ = np.array([[d[k] for k in cm] for d in tail])
            if float(np.linalg.norm(st_ - st_.mean(0), axis=2).max()) <= STILL_PX * 2:
                common, stack = cm, st_
                row['settled_tail_frames'] = len(tail)
                row['drift_before_tail_frames'] = start
                break
        if stack is None:
            first = sorted(set.intersection(*[set(d) for d in dets]))
            full = np.array([[d[k] for k in first] for d in dets]) if len(first) >= MIN_CORNERS else None
            row['board_motion_px'] = round(float(np.linalg.norm(full - full.mean(0), axis=2).max()), 2) if full is not None else None
            row['result'] = 'image never settled during hold (camera/board/stabiliser moving)'
            continue
        row['board_motion_px'] = round(float(np.linalg.norm(stack - stack.mean(0), axis=2).max()), 2)
        img = stack.mean(0)
        obj = corners_all[common]
        # Noise of the AVERAGED corners, measured from this hold's frame-to-frame scatter.
        frame_noise = float(np.median(np.linalg.norm(stack - img, axis=2)) / np.sqrt(2)) if len(stack) > 1 else 0.3
        mean_noise = max(frame_noise / np.sqrt(len(stack)), 0.01)
        row['corner_noise_px_frame_mean'] = [round(frame_noise, 3), round(mean_noise, 4)]
        B, info = N.board_pose(obj, img, K, dist, noise_px=mean_noise)
        if B is None:
            row['result'] = 'board pose rejected: ' + info['reason']
            continue
        q = np.array([s.rotation for s in win])
        q[(q @ q[0]) < 0] *= -1
        qm = q.mean(0) / np.linalg.norm(q.mean(0))
        samples.append({'A': P.tracker_rh(Pp.mean(0), qm), 'B': B, 'obj': obj, 'img': img, 'ambiguous': info['ambiguous'],
                        'rot_std_deg': info['rot_std_deg']})
        row.update(result='sample %d' % len(samples), corners=len(common), pnp_rms_px=round(info['rms_px'], 3),
                   ambiguous=info['ambiguous'], board_distance_m=round(float(B[2, 3]), 2))
    print('position  result')
    for r in report:
        print(f"{r['position']:>8}  {r['result']}  " + ' '.join(f'{k}={v}' for k, v in r.items() if k not in ('position', 'result')))

    out = {'created': datetime.datetime.now().astimezone().isoformat(), 'recording': str(rec), 'holds': report,
           'lens_source': str(lens_path) if lens_path.is_file() else 'profile lens', 'samples': len(samples)}
    try:
        r = N.solve_nodal(samples, K, dist)
        out['solve'] = r
        X, Z = P.inv(np.array(r['T_tracker_cam'])), np.array(r['T_world_board'])
        name = ''.join(c for c in prof.get('name', 'rig') if c.isalnum() or c in '-_.')
        rig_profile = {'name': name + '_rema', 'info': prof.get('info'), 'board': prof['board'],
                       'lens': {**lens, 'K': K.tolist(), 'dist': dist.tolist(), 'size': lens.get('size', [1360, 768])},
                       'imageCenter': {'dx': 0.0, 'dy': 0.0},
                       'nodal': {'T_tracker_cam': r['T_tracker_cam'], 'T_world_board': r['T_world_board'],
                                 'reprojRmsPx': (r['heldout_rms_px_median_max'] or r['fit_rms_px_median_max'])[0],
                                 'samples': len(r['fit_samples']), 'recording': str(rec), 'solver': 'rema.nodal',
                                 'method': r['method'], 'heldout_board_error_mm_deg': r['methods'][r['method']]['heldout_mm_deg_median'],
                                 'heldout_rot_deg_max': r['methods'][r['method']]['heldout_deg_max'],
                                 'qualified': False, 'status': 'Solved; not yet validated by a live registration test.'}}
        dest = _P.DATA / 'calibration' / 'rig' / (name + '_rema.json')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rig_profile, indent=2))
        out['rig_profile'] = str(dest)
        ho = r['heldout_rms_px_median_max']
        print(f"\nSOLVED from {len(samples)} samples (fit {len(r['fit_samples'])}, held out {len(r['heldout_samples'])}, dropped {len(r['dropped_outliers'])})")
        print(f"  fit reprojection median/max {r['fit_rms_px_median_max'][0]:.2f}/{r['fit_rms_px_median_max'][1]:.2f} px")
        print(f"  HELD-OUT reprojection median/max {ho[0]:.2f}/{ho[1]:.2f} px" if ho else '  no held-out samples')
        print(f"  camera offset from Alt {r['offset_m'] * 100:.1f} cm, rotation {r['camera_axis_vs_tracker_deg']:.1f} deg")
        print(f"  method used: {r['method']} (board orientation precision median {r['board_rot_std_deg_median']:.3f} deg; rotation-first if < {N.ROTATION_FIRST_MAX_STD_DEG})")
        for name, m in r['methods'].items():
            ho = m['heldout_mm_deg_median']
            print(f"    {name:15s} held-out board error {ho[0]:.1f} mm / {ho[1]:.3f} deg (max {m['heldout_deg_max']:.3f}), held-out px {m['heldout_px_median']:.1f}, offset {m['offset_m'] * 100:.1f} cm" if ho else f"    {name}: no held-out")
        print(f"  rotation variety: {r['diversity']['advice']} (max {r['diversity']['max_relative_deg']:.0f} deg, spread {r['diversity']['axis_spread']:.2f})")
        print('  rig profile:', dest)
    except ValueError as e:
        out['error'] = str(e)
        X = Z = None
        print('\nCANNOT SOLVE:', e)
    (rec / 'nodal_result.json').write_text(json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))

    if not a.no_video:
        vid = None
        c = corners_all
        outline = np.c_[np.array([c[:, :2].min(0), [c[:, 0].max(), c[:, 1].min()], c[:, :2].max(0), [c[:, 0].min(), c[:, 1].max()]]), np.zeros(4)]
        for f in frames[::2]:
            img = cv2.imread(str(rec / f['file']))
            if img is None:
                continue
            right = img.copy()
            cc, ids, _, _ = det.detectBoard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if ids is not None:
                for x, y in cc.reshape(-1, 2):
                    cv2.circle(right, (int(x), int(y)), 3, (0, 255, 0), -1)
            qy = hist.at(f['kernel_ns'])
            if X is not None and qy.valid:
                T = X @ P.inv(P.tracker_rh(qy.sample.position, qy.sample.rotation)) @ Z
                if ((outline @ T[:3, :3].T + T[:3, 3])[:, 2] > 0.05).all():
                    rv, _ = cv2.Rodrigues(T[:3, :3])
                    uv, _ = cv2.projectPoints(outline, rv, T[:3, 3], K, dist)
                    cv2.polylines(right, [np.rint(uv.reshape(-1, 2)).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
            put_lines(img, ['RED via Osprey (recorded)', f"seq {f['sequence']}  phase {f['phase']}  position {f['position'] + 1}"], scale=0.7)
            put_lines(right, ['green: detected corners   magenta: board predicted from ALT6 tracking only',
                              'nodal ' + ('solved - outline should sit on the board' if X is not None else 'NOT solved'),
                              'calibration not yet independently qualified'], scale=0.7,
                      colours=[None, (255, 0, 255) if X is not None else (60, 60, 255), (0, 200, 255)])
            frame = np.hstack([img, right])
            if vid is None:
                vid = TimelineH264(rec / 'nodal_review.mp4', frame.shape[1], frame.shape[0])
            vid.write(frame, f['kernel_ns'])
        if vid:
            vid.close()
            print('review video:', rec / 'nodal_review.mp4')
    from rema.video_out import notify
    notify('REMA nodal solve finished', ('solved: ' + out['solve']['method'] if out.get('solve') else 'NOT solved: ' + out.get('error', '')) + ' - details in the terminal')


if __name__ == '__main__':
    main()
