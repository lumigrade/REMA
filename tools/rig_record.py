#!/usr/bin/env python3
"""Countdown-guided rig recording: RED (Osprey) + ALT6 pose + Astra RGB-D, all on CLOCK_MONOTONIC.

  python tools/rig_record.py --positions 14            # nodal: board FIXED, move the camera
  python tools/rig_record.py --positions 14 --no-solve # record only

On screen: a big countdown. MOVE phase: move the camera (tripod head pan/tilt, tripod position or
height) to the suggested new view with the whole board in frame. HOLD phase: hands off, everything
still. Nothing is decided live: every frame and tracker sample is recorded, and the offline step
(tools/rig_nodal_offline.py, run automatically at the end) picks the still part of each hold.

Output $REMA_DATA/recordings/rig_rec_<stamp>/:
  red/<seq>.png (hold, lossless) or .jpg (move, preview), red.jsonl   (sequence, kernel_ns, arrival_ns, phase, position)
  alt6.jsonl  raw bridge lines (t_ns = SDK-query CLOCK_MONOTONIC)
  orbbec/     astra_capture dataset (depth .bin + RGB, host steady_clock = CLOCK_MONOTONIC)
  plan.json   phase timeline (monotonic ns)
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import argparse
import datetime
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from rema.sdi_capture import find_video_device
from rema.video_out import put_lines

MOVES = [
    'start: aim at the board, board fully in frame',
    'pan LEFT ~15 deg (keep board in frame)',
    'tilt UP ~10 deg',
    'pan RIGHT ~30 deg',
    'tilt DOWN ~20 deg',
    'move tripod ~50 cm LEFT, re-aim at board',
    'raise tripod ~20 cm, tilt down at board',
    'pan + tilt diagonally (up-right)',
    'move tripod ~40 cm CLOSER, re-aim',
    'lower tripod ~30 cm, tilt up at board',
    'move tripod ~60 cm RIGHT, re-aim',
    'tilt DOWN + pan LEFT',
    'move tripod BACK ~50 cm, re-aim',
    'roll the camera a little if the head allows, else tilt UP',
    'any new angle: pan/tilt somewhere not used yet',
    'any new angle: change tripod height again',
]


class FrameWriter(threading.Thread):
    def __init__(self, out):
        super().__init__(daemon=True)
        self.q, self.out = queue.Queue(maxsize=64), out
        self.log = (out / 'red.jsonl').open('w')
        self.dropped = 0

    def put(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1            # recorded in meta; never silently

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            f, phase, pos = item
            lossless = phase == 'hold'
            name = f'red/{f.sequence:06d}.' + ('png' if lossless else 'jpg')
            cv2.imwrite(str(self.out / name), f.image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1] if lossless else [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.log.write(json.dumps({'sequence': f.sequence, 'kernel_ns': f.kernel_ns, 'arrival_ns': f.arrival_ns,
                                       'timestamp_source': f.timestamp_source, 'phase': phase, 'position': pos,
                                       'file': name}) + '\n')
        self.log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--positions', type=int, default=14)
    ap.add_argument('--move', type=float, default=5.0, help='seconds to move between positions')
    ap.add_argument('--hold', type=float, default=3.0, help='seconds to hold still')
    ap.add_argument('--intro', type=float, default=10.0)
    ap.add_argument('--device', default='Osprey', help='capture device: name fragment (default Osprey) or /dev/videoN')
    ap.add_argument('--alt-serial', default='4C0046001951383039383438')
    ap.add_argument('--no-orbbec', action='store_true')
    ap.add_argument('--no-solve', action='store_true')
    ap.add_argument('--profile', default=str(Path.home() / 'Desktop/VPcamtracking/alt2freed/calib/profiles/RED_VRaptor_Optimo2876_ALT6.json'))
    ap.add_argument('--lens', default=str(_P.DATA / 'calibration/rig/lens_34view.json'))
    a = ap.parse_args()
    if not 4 <= a.positions <= len(MOVES):
        ap.error(f'positions must be 4..{len(MOVES)}')
    if not _P.disk_reserve_ok():
        sys.exit('25 GiB free-space reserve failed')
    total = a.intro + a.positions * (a.move + a.hold) + 1.0
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = _P.DATA / 'recordings' / f'rig_rec_{stamp}'
    (out / 'red').mkdir(parents=True)

    from rema.qt_env import ensure_xcb_cursor
    ensure_xcb_cursor()
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QApplication, QLabel
    sys.path.insert(0, str(_P.ROOT / 'tools'))
    from rig_live import CaptureThread

    # board detector for live feedback only (corner count); offline step does the real work
    prof = json.loads(Path(a.profile).expanduser().read_text())
    bc = prof['board']
    board = cv2.aruco.CharucoBoard((int(bc['squaresX']), int(bc['squaresY'])), bc['squareLength'], bc['squareLength'] * bc['markerRatio'],
                                   cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, bc['dictionary'])))
    detector = cv2.aruco.CharucoDetector(board)

    # ---- start sources
    alt_log = (out / 'alt6.jsonl').open('w')
    alt_state = {'stage': 'starting', 'count': 0}
    bridge = subprocess.Popen([str(_P.WORK / 'alt_pose_bridge'), '--rate', '250', '--sdk-lib', str(_P.WORK / 'lib'), '--serial', a.alt_serial],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def read_bridge():
        for line in bridge.stdout:
            alt_log.write(line)
            if '"t_ns"' in line:
                alt_state['count'] += 1
                i = line.find('"stage":"')
                alt_state['stage'] = line[i + 9:line.find('"', i + 9)] if i >= 0 else '?'
    threading.Thread(target=read_bridge, daemon=True).start()

    orbbec = None
    if not a.no_orbbec:
        secs = int(total + 3)
        orbbec = subprocess.Popen(['python3', str(_P.ROOT / 'tools/launch.py'), '--seconds', str(secs + 30), '--', '/usr/bin/env',
                                   'LD_PRELOAD=' + str(_P.WORK / 'astra_usb_compat.so'), str(_P.WORK / 'astra_capture'),
                                   str(out / 'orbbec'), str(secs), find_video_device('Astra')],
                                  stdout=(out / 'orbbec_log.txt').open('w'), stderr=subprocess.STDOUT, cwd=str(_P.ROOT))
    cap = CaptureThread(a.device)
    cap.start()
    writer = FrameWriter(out)
    writer.start()

    # ---- timeline
    t0 = time.monotonic_ns() + int(1.5e9)          # small lead so sources settle
    plan, t = [], t0 + int(a.intro * 1e9)
    plan.append({'phase': 'intro', 'position': -1, 'start_ns': t0, 'end_ns': t})
    for i in range(a.positions):
        plan.append({'phase': 'move', 'position': i, 'start_ns': t, 'end_ns': t + int(a.move * 1e9), 'instruction': MOVES[i]})
        t += int(a.move * 1e9)
        plan.append({'phase': 'hold', 'position': i, 'start_ns': t, 'end_ns': t + int(a.hold * 1e9)})
        t += int(a.hold * 1e9)
    end_ns = t
    (out / 'plan.json').write_text(json.dumps({'positions': a.positions, 'move_s': a.move, 'hold_s': a.hold, 'phases': plan,
                                                'mode': 'nodal: board fixed in room, camera moves', 'alt_serial': a.alt_serial,
                                                'profile': a.profile, 'lens': a.lens}, indent=2))

    app = QApplication(sys.argv)
    label = QLabel()
    label.setWindowTitle('REMA rig recording  [Esc] abort')
    label.setAlignment(Qt.AlignCenter)
    label.resize(1500, 860)
    label.show()
    st = {'seq': None, 'aborted': False}

    def phase_at(now):
        for p in plan:
            if p['start_ns'] <= now < p['end_ns']:
                return p
        return None

    def tick():
        now = time.monotonic_ns()
        if now >= end_ns or st['aborted']:
            app.quit()
            return
        f = cap.latest
        if f is None or f.sequence == st['seq']:
            return
        st['seq'] = f.sequence
        p = phase_at(f.kernel_ns) or {'phase': 'lead', 'position': -1}
        writer.put((f, p['phase'], p['position']))
        cur = phase_at(now) or {'phase': 'lead', 'position': -1, 'end_ns': t0}
        remain = max(0.0, (cur['end_ns'] - now) / 1e9)
        small = cv2.resize(f.image_bgr, (680, 384))
        cc, ids, _, _ = detector.detectBoard(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        n = 0 if ids is None else len(ids)
        img = f.image_bgr.copy()
        big = {'intro': ('GET READY', (0, 200, 255)), 'move': ('MOVE CAMERA', (0, 200, 255)),
               'hold': ('HOLD STILL', (60, 60, 255)), 'lead': ('STARTING', (200, 200, 200))}[cur['phase']]
        text = f'{big[0]}  {remain:0.0f}' if cur['phase'] != 'hold' else f'{big[0]}  {remain:0.1f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2.6, 6)
        cv2.rectangle(img, (0, 0), (img.shape[1], th + 50), (20, 20, 20), -1)
        cv2.putText(img, text, ((img.shape[1] - tw) // 2, th + 22), cv2.FONT_HERSHEY_DUPLEX, 2.6, big[1], 6, cv2.LINE_AA)
        lines = [f"position {cur['position'] + 1}/{a.positions}" if cur['position'] >= 0 else 'board FIXED in the room - camera moves',
                 'next: ' + MOVES[cur['position']] if cur['phase'] == 'move' else ('hands off the camera' if cur['phase'] == 'hold' else MOVES[0]),
                 f"board corners {n}/{len(board.getChessboardCorners())} (preview)   ALT6 {alt_state['stage']}"]
        colours = [None, (0, 255, 255), (80, 220, 80) if n >= 20 and alt_state['stage'] == 'Tracking6Dof' else (60, 60, 255)]
        if orbbec is None:
            lines.append('Astra off'); colours.append((200, 200, 200))
        elif orbbec.poll() is None:
            lines.append('Astra recording'); colours.append((80, 220, 80))
        else:
            lines.append('Astra FAILED (see orbbec_log.txt) - RED + ALT6 still recording'); colours.append((60, 60, 255))
        put_lines(img, lines, origin=(16, th + 95), scale=0.9, colours=colours)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def key(ev):
        if ev.key() == Qt.Key_Escape:
            st['aborted'] = True
    label.keyPressEvent = key
    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(10)
    print(f'recording {total:.0f} s to {out}', flush=True)
    try:
        app.exec()
    finally:
        label.close()                      # don't leave a frozen countdown on screen while solving
        app.processEvents()
        print('recording finished - saving files, then solving (about 1 minute; see this terminal)', flush=True)
        cap.stop_event.set()
        writer.put(None)
        writer.join(timeout=30)
        try:
            bridge.stdin.close()
            bridge.wait(timeout=3)
        except Exception:
            bridge.terminate()
        alt_log.close()
        if orbbec:
            try:
                orbbec.wait(timeout=60)
            except subprocess.TimeoutExpired:
                orbbec.terminate()
    meta = {'created': datetime.datetime.now().astimezone().isoformat(), 'aborted': st['aborted'], 'writer_dropped_frames': writer.dropped,
            'driver_dropped_frames': cap.cap.dropped_by_driver if cap.cap else None, 'capture': cap.desc,
            'alt_samples': alt_state['count'], 'orbbec_exit': orbbec.returncode if orbbec else None,
            'clock': 'all timestamps CLOCK_MONOTONIC ns (V4L2 SOF, bridge SDK query, astra steady_clock arrival)'}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str), flush=True)
    if not a.no_solve and not st['aborted']:
        subprocess.run([sys.executable, str(_P.ROOT / 'tools/rig_nodal_offline.py'), str(out)])


if __name__ == '__main__':
    main()
