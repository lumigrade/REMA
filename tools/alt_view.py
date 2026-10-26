#!/usr/bin/env python3
"""Record and visualise what an Antilatency Alt sees (LED rays), against the environment model.

  python tools/alt_view.py [--serial SN] [--rays-seconds 10]      # record + render
  python tools/alt_view.py --render DIR                            # re-render an existing recording

The Alt runs one task at a time: first ~4 s of normal tracking (rig must be still) give its pose;
then the tracking-data task streams each optical frame's detected LED rays (no pose during that).
The video is a plot, not a camera image: green = LEDs the Alt detected (size ~ brightness),
magenta rings = where the environment says each LED should appear from the measured pose.
Close other Alt users first (AntilatencyService, alt2freed / VPcamtracking UI stream, REMA bridge).
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import argparse
import datetime
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from rema import pose as P
from rema.video_out import TimelineH264, put_lines

SIZE = 900
FOV_DEG = 70.0          # radius of the plot in degrees from the optics axis


def plot_xy(dirs):
    """Azimuthal-equidistant projection around the optics +Z axis -> pixel coordinates."""
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    theta = np.degrees(np.arccos(np.clip(d[:, 2], -1, 1)))
    phi = np.arctan2(d[:, 1], d[:, 0])
    r = theta / FOV_DEG * (SIZE / 2 - 30)
    return np.c_[SIZE / 2 + r * np.cos(phi), SIZE / 2 + r * np.sin(phi)]


def render(rec):
    L = [json.loads(l) for l in (rec / 'rays.jsonl').read_text().splitlines() if l.startswith('{')]
    ev = {l['event']: l for l in L if 'event' in l}
    F = [l for l in L if 't_ns' in l]
    if 'pose' not in ev or not ev['pose'].get('samples_6dof'):
        raise SystemExit('no 6DoF pose recorded: ' + json.dumps(ev.get('pose')))
    M = np.array(ev['environment']['markers'])
    p = np.array(ev['pose']['p'])
    R = P.quat_to_mat(ev['pose']['q']) @ P.quat_to_mat(ev['optics_to_body']['q'])   # optics -> world (Antilatency frame)
    expected = ((M - p) / np.linalg.norm(M - p, axis=1, keepdims=True)) @ R          # rows: R^T d
    exp_xy = plot_xy(expected)
    stats = {'frames': len(F), 'rays_per_frame': [], 'residual_deg': []}
    base = np.full((SIZE, SIZE, 3), 18, np.uint8)
    for deg in range(10, int(FOV_DEG) + 1, 10):
        cv2.circle(base, (SIZE // 2, SIZE // 2), int(deg / FOV_DEG * (SIZE / 2 - 30)), (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(base, f'{deg}', (SIZE // 2 + 4, SIZE // 2 - int(deg / FOV_DEG * (SIZE / 2 - 30)) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    cv2.line(base, (SIZE // 2, 30), (SIZE // 2, SIZE - 30), (45, 45, 45), 1)
    cv2.line(base, (30, SIZE // 2), (SIZE - 30, SIZE // 2), (45, 45, 45), 1)
    vid = TimelineH264(rec / 'alt_view.mp4', SIZE, SIZE, fps=25, bitrate_kbps=4000)
    for f in F:
        img = base.copy()
        for i, (x, y) in enumerate(exp_xy):
            cv2.circle(img, (int(x), int(y)), 14, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(i), (int(x) + 16, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 200), 1)
        rays = np.array(f['rays']) if f['rays'] else np.zeros((0, 6))
        stats['rays_per_frame'].append(len(rays))
        if len(rays):
            d = rays[:, 2:5] / np.linalg.norm(rays[:, 2:5], axis=1, keepdims=True)
            res = np.degrees(np.arccos(np.clip(d @ expected.T, -1, 1))).min(1)
            stats['residual_deg'].extend(res.tolist())
            for (x, y), pw in zip(plot_xy(d), rays[:, 5]):
                cv2.circle(img, (int(x), int(y)), int(3 + 2.5 * pw), (80, 230, 80), -1, cv2.LINE_AA)
        put_lines(img, ['ALT6 optical view (plot of detected LED rays, not a camera image)',
                        f'LEDs detected {len(rays)}/{len(M)}   env {ev["environment"]["name"]}   rings every 10 deg from optics axis',
                        'green = detected LED (size ~ brightness)   magenta = expected from pose + environment',
                        f'pose quality value {ev["pose"].get("quality", float("nan")):.3f}   height {p[1]:.2f} m'], scale=0.55,
                  colours=[None, None, None, (0, 200, 255)])
        vid.write(img, f['t_ns'])
    vid.close()
    n = np.array(stats['rays_per_frame'])
    r = np.array(stats['residual_deg'])
    summary = {'frames': len(F), 'duration_s': (F[-1]['t_ns'] - F[0]['t_ns']) / 1e9 if F else 0,
               'optical_fps': (len(F) - 1) / ((F[-1]['t_ns'] - F[0]['t_ns']) / 1e9) if len(F) > 1 else None,
               'markers_in_environment': len(M), 'rays_per_frame_min_p50_max': [int(n.min()), float(np.median(n)), int(n.max())],
               'frames_without_rays': int((n == 0).sum()),
               'ray_vs_expected_deg_p50_p95': [float(np.median(r)), float(np.percentile(r, 95))] if len(r) else None,
               'alt_height_m': float(p[1]), 'distance_to_markers_m': np.linalg.norm(M - p, axis=1).round(2).tolist(),
               'scope': 'Single static pose; expected directions use the Alt pose estimated from the same markers, so agreement checks consistency, not absolute accuracy.'}
    (rec / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('video:', rec / 'alt_view.mp4')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--serial', default='4C0046001951383039383438')
    ap.add_argument('--rays-seconds', type=float, default=10)
    ap.add_argument('--render', default='')
    a = ap.parse_args()
    if a.render:
        render(Path(a.render).expanduser())
        return
    rec = _P.DATA / 'recordings' / ('alt_view_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    rec.mkdir(parents=True)
    with (rec / 'rays.jsonl').open('w') as out:
        p = subprocess.Popen([str(_P.WORK / 'alt_rays_bridge'), '--serial', a.serial, '--track-seconds', '4',
                              '--rays-seconds', str(a.rays_seconds), '--sdk-lib', str(_P.WORK / 'lib')],
                             stdin=subprocess.PIPE, stdout=out, stderr=subprocess.STDOUT)
        p.wait(timeout=a.rays_seconds + 40)
    render(rec)


if __name__ == '__main__':
    main()
