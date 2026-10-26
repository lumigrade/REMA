#!/usr/bin/env python3
"""Probe the SDI bridge: advertised modes, negotiated mode, real frame rate, driver drops,
timestamp source, signal state (live / colour bars / frozen), plus a labeled H.264 preview.

  python tools/probe_sdi.py --seconds 10 [--device Osprey|/dev/videoN]

Writes $REMA_DATA/recordings/sdi_probe_<stamp>/{report.json,frames.jsonl,preview.mp4}.
Measures capture timing only; it says nothing about camera exposure or glass-to-glass latency.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import argparse
import datetime
import json
import subprocess
import time

import cv2
import numpy as np

from rema.sdi_capture import SignalMonitor, V4L2Capture, list_modes


OUT_FPS = 25


def h264_writer(path, width, height, fps):
    """GStreamer x264 encoder fed raw BGR on stdin (ffmpeg is not installed on this host)."""
    cmd = ['gst-launch-1.0', '-q', 'fdsrc', 'fd=0', '!',
           f'rawvideoparse', 'format=bgr', f'width={width}', f'height={height}', f'framerate={int(round(fps))}/1', '!',
           'videoconvert', '!', 'x264enc', 'speed-preset=veryfast', 'tune=zerolatency', 'bitrate=8000', '!',
           'video/x-h264,profile=high', '!', 'mp4mux', '!', 'filesink', f'location={path}']
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def label(img, lines):
    out = img.copy()
    y = 34
    for text in lines:
        cv2.putText(out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        y += 32
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='Osprey', help='capture device: name fragment (default Osprey) or /dev/videoN')
    ap.add_argument('--seconds', type=float, default=10)
    ap.add_argument('--no-video', action='store_true')
    a = ap.parse_args()
    if not 1 <= a.seconds <= 120:
        ap.error('seconds must be 1-120')
    if not _P.disk_reserve_ok():
        sys.exit('25 GiB free-space reserve failed')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = _P.DATA / 'recordings' / f'sdi_probe_{stamp}'
    out.mkdir(parents=True)
    modes = list_modes(a.device)
    rows, states = [], {}
    writer = None
    with V4L2Capture(a.device) as cap, (out / 'frames.jsonl').open('w') as log:
        desc = cap.describe()
        mon = SignalMonitor()
        t0 = time.monotonic()
        while time.monotonic() - t0 < a.seconds:
            f = cap.read(latest_only=False)
            if f is None:
                continue
            state = mon.update(f.image_bgr)
            states[state] = states.get(state, 0) + 1
            row = {'sequence': f.sequence, 'kernel_ns': f.kernel_ns, 'arrival_ns': f.arrival_ns,
                   'timestamp_source': f.timestamp_source, 'error_flag': f.error_flag, 'signal': state,
                   'luma_mean': float(f.image_bgr.mean())}
            rows.append(row)
            log.write(json.dumps(row) + '\n')
            if not a.no_video:
                if writer is None:
                    writer = h264_writer(out / 'preview.mp4', f.width, f.height, OUT_FPS)
                    first_ns, written = f.kernel_ns, 0
                # Place frames on the real kernel timeline so playback speed is truthful even when
                # the bridge delivers fewer frames than its advertised rate (repeats fill the gaps).
                slots = int((f.kernel_ns - first_ns) / 1e9 * OUT_FPS) + 1 - written
                frame_bytes = label(f.image_bgr, [
                    f'REMA SDI probe  {desc["card"]}  {f.width}x{f.height} {f.pixel_format}',
                    f'seq {f.sequence}  signal {state}  kernel->app {(f.arrival_ns - f.kernel_ns) / 1e6:.1f} ms',
                    'Capture-path check only - not latency or colour qualification']).tobytes()
                for _ in range(max(slots, 0)):
                    writer.stdin.write(frame_bytes)
                written += max(slots, 0)
        drops = cap.dropped_by_driver
    if writer:
        writer.stdin.close()
        writer.wait()
    k = np.array([r['kernel_ns'] for r in rows], float)
    dk = np.diff(k) / 1e6
    lat = np.array([(r['arrival_ns'] - r['kernel_ns']) / 1e6 for r in rows])
    report = {
        'created': datetime.datetime.now().astimezone().isoformat(), 'device': desc, 'advertised_modes': modes,
        'frames': len(rows), 'driver_dropped_frames': drops,
        'measured_fps_kernel_clock': round((len(k) - 1) / ((k[-1] - k[0]) / 1e9), 3) if len(k) > 1 else None,
        'interval_ms_p5_p50_p95_max': [round(float(v), 2) for v in (*np.percentile(dk, [5, 50, 95]), dk.max())] if len(dk) else None,
        'kernel_to_app_ms_p50_p95': [round(float(v), 2) for v in np.percentile(lat, [50, 95])] if len(lat) else None,
        'timestamp_source': rows[-1]['timestamp_source'] if rows else None,
        'signal_states': states,
        'scope': 'Capture path only. Kernel timestamps are host start-of-frame, not RED exposure. '
                 'No glass-to-glass latency, colour pipeline or calibration qualification.',
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('output', out)


if __name__ == '__main__':
    sys = _sys
    main()
