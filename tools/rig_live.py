#!/usr/bin/env python3
"""REMA rig live: RED SDI (Osprey) + Antilatency pose -> virtual grid/blocks, optional RVM actor matte.

  # grid registration check with the real tracker (Alt must be visible to the SDK):
  python tools/rig_live.py --pose alt2freed --profile ~/Desktop/VPcamtracking/alt2freed/calib/profiles/Default.json
  # pipeline demo without a tracker (clearly labeled SYNTHETIC POSE):
  python tools/rig_live.py --pose synthetic --headless --record 12
  # actor over grid (GPU env):
  /home/pc/rema-data/envs/live-gpu/bin/python tools/gpu_session.py -- <that python> tools/rig_live.py --matte cuda ...

Views: split (RED | output), overlay (grid blended on RED, for registration), composite.
Fallback: any invalid/stale/missing pose shows the current RED frame with the reason; no stale pose.
Nothing here qualifies calibration, timing or latency; on-screen labels say what is provisional.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import argparse
import datetime
import json
import math
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from rema import pose as P
from rema import virtual_scene as VS
from rema.sdi_capture import SignalMonitor, V4L2Capture, picture_inset
from rema.video_out import TimelineH264, put_lines

cv2.setNumThreads(2)
DEFAULT_ALT2FREED = Path.home() / 'Desktop/VPcamtracking/alt2freed/alt2freed'
VERSION = 'rig_live_v1'


# ---------------------------------------------------------------------------- pose sources

class PoseSource(threading.Thread):
    def __init__(self, history):
        super().__init__(daemon=True)
        self.history, self.lock = history, threading.Lock()
        self.status, self.stop_event, self.count = 'starting', threading.Event(), 0

    def push(self, sample):
        if sample is None:
            return
        with self.lock:
            if self.history.add(sample):
                self.count += 1

    def query(self, t_ns):
        with self.lock:
            return self.history.at(t_ns)

    def stop(self):
        self.stop_event.set()


class SubprocessPose(PoseSource):
    """alt2freed `stream --json` (raw tracker pose, ~30 Hz, pipe-arrival time) or the REMA bridge."""

    def __init__(self, history, cmd, parser):
        super().__init__(history)
        self.cmd, self.parser, self.proc = cmd, parser, None
        self.events = []

    def run(self):
        try:
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as e:
            self.status = f'cannot start tracker: {e}'
            return
        self.status = 'waiting for tracker'
        for line in self.proc.stdout:
            now = time.monotonic_ns()
            if self.stop_event.is_set():
                break
            s = self.parser(line, now) if self.parser is P.parse_alt2freed_json else self.parser(line)
            if s is not None:
                self.push(s)
                self.status = f'{s.stage} q={s.quality:.2f}'
            elif line.strip():
                self.events.append(line.strip()[:200])
                if '"event":"tracking"' in line:
                    self.status = 'tracking started'
                elif 'Error' in line or 'disconnected' in line:
                    self.status = line.strip()[:80]
        self.status = f'tracker exited ({self.proc.poll()})'

    def stop(self):
        super().stop()
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.close()          # alt2freed --json exits when stdin closes
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.terminate()


class FreeDPose(PoseSource):
    def __init__(self, history, port):
        super().__init__(history)
        self.port = port

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', self.port))
        sock.settimeout(0.5)
        self.status = f'listening FreeD UDP {self.port}'
        while not self.stop_event.is_set():
            try:
                data, _ = sock.recvfrom(64)
            except socket.timeout:
                continue
            now = time.monotonic_ns()
            try:
                self.push(P.freed_to_sample(P.decode_freed_d1(data), now))
                self.status = 'FreeD (no stage/time in protocol)'
            except ValueError as e:
                self.status = str(e)


class SyntheticPose(PoseSource):
    """Smooth dolly/pan path at 200 Hz in the Antilatency frame. For pipeline demos ONLY."""

    def run(self):
        self.status = 'SYNTHETIC'
        t0 = time.monotonic_ns()
        while not self.stop_event.is_set():
            now = time.monotonic_ns()
            t = (now - t0) / 1e9
            pos = (0.6 * math.sin(0.35 * t), 1.55 + 0.15 * math.sin(0.5 * t), -2.0 + 0.8 * math.sin(0.23 * t))
            R = P.unity_euler_to_mat(math.radians(14 + 4 * math.sin(0.4 * t)), math.radians(12 * math.sin(0.3 * t)), 0.0)
            self.push(P.PoseSample(now, pos, tuple(P.mat_to_quat(R)), 'Tracking6Dof', 1.0, 'synthetic_monotonic', 'synthetic'))
            time.sleep(0.005)


# ---------------------------------------------------------------------------- capture

class CaptureThread(threading.Thread):
    def __init__(self, device):
        super().__init__(daemon=True)
        self.device, self.latest, self.cond = device, None, threading.Condition()
        self.error, self.stop_event, self.desc, self.cap = None, threading.Event(), None, None

    def run(self):
        try:
            with V4L2Capture(self.device) as cap:
                self.cap, self.desc = cap, cap.describe()
                while not self.stop_event.is_set():
                    f = cap.read(timeout=0.5)
                    if f is None:
                        continue
                    with self.cond:
                        self.latest = f
                        self.cond.notify_all()
        except Exception as e:      # report, never hide
            self.error = f'{type(e).__name__}: {e}'
            with self.cond:
                self.cond.notify_all()

    def next_frame(self, after_seq, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.error or (self.latest and self.latest.sequence != after_seq), timeout)
            return self.latest


# ---------------------------------------------------------------------------- engine

class Engine:
    def __init__(self, a):
        self.a = a
        self.capture = CaptureThread(a.device)
        self.history = P.PoseHistory(max_gap_ms=a.max_pose_gap_ms,
                                     valid_stages={'Tracking6Dof', 'FreeD_unknown'} if a.trust_freed else P.VALID_STAGES)
        self.pose = self._make_pose_source()
        self.rig = None
        self.matte = None
        self.matte_error = None
        if a.matte != 'none':
            os.environ['REMA_MATTE_BACKEND'] = a.matte
            try:
                from rema.matting import TemporalMatte
                self.matte = TemporalMatte(ratio=a.matte_ratio)
            except Exception as e:
                self.matte_error = f'{type(e).__name__}: {e}'
        self.monitor = SignalMonitor()
        self.view = a.view
        self.side_by_side = True
        self.last = None
        self.sdk_baseline = None
        self.held_sample = None
        self.held_since = None
        self.inset = None
        self.splat = None
        self.splat_error = None
        self.scene = None
        if a.background == 'splat':
            from rema import splat as SP
            self.scene = SP.load_scene(a.scene)
            self.scene_align = SP.load_alignment(a.scene_align) if a.scene_align else {}
        self.last_seq = None
        self.times = []
        self.stats = {'frames': 0, 'rendered': 0, 'fallback': {}}

    def _make_pose_source(self):
        a = self.a
        if a.pose == 'alt2freed':
            cmd = [str(a.alt2freed), 'stream', '--json', '--no-smooth', '--rate', '120', '--ip', '127.0.0.1', '--port', str(a.freed_sink_port)]
            if a.alt_serial:
                cmd += ['--serial', a.alt_serial]
            return SubprocessPose(self.history, cmd, P.parse_alt2freed_json)
        if a.pose == 'bridge':
            cmd = [str(_P.WORK / 'alt_pose_bridge'), '--rate', '250', '--sdk-lib', str(_P.WORK / 'lib')]
            if a.alt_serial:
                cmd += ['--serial', a.alt_serial]
            return SubprocessPose(self.history, cmd, P.parse_bridge_json)
        if a.pose == 'freed':
            return FreeDPose(self.history, a.freed_port)
        if a.pose == 'synthetic':
            return SyntheticPose(self.history)
        return None

    def start(self):
        self.capture.start()
        if self.pose:
            self.pose.start()

    def stop(self):
        self.capture.stop_event.set()
        if self.pose:
            self.pose.stop()

    def _rig_for(self, w, h):
        if self.rig is None:
            if self.a.profile:
                self.rig = P.rig_from_vpcamtracking(Path(self.a.profile).expanduser())
            else:
                self.rig = P.provisional_rig(w, h, self.a.hfov)
            self.rig.floor_z_m = self.a.floor_z
        return self.rig

    def _background(self, Tcw, K, dist, w, h):
        if self.scene is not None and self.splat_error is None:
            if self.splat is None:
                try:     # created lazily on THIS thread: an OpenGL context belongs to its creating thread
                    from rema import splat as SP
                    self.splat = SP.SplatRenderer(self.scene, self.scene_align, K, dist, (w, h))
                except Exception as e:
                    self.splat_error = f'{type(e).__name__}: {e}'
            if self.splat is not None:
                return self.splat.render(Tcw)
        return VS.render(Tcw, K, dist, (w, h))

    def step(self):
        """Process the next fresh frame. Returns (display_bgr, telemetry) or (None, None)."""
        f = self.capture.next_frame(self.last_seq)
        if self.capture.error:
            raise RuntimeError('capture failed: ' + self.capture.error)
        if f is None or f.sequence == self.last_seq:
            return None, None
        self.last_seq = f.sequence
        t_start = time.monotonic_ns()
        h, w = f.image_bgr.shape[:2]
        rig = self._rig_for(w, h)
        signal = self.monitor.update(f.image_bgr)
        if self.stats['frames'] % 15 == 0:          # camera monitoring overlays shrink the picture into a window
            self.inset = picture_inset(f.image_bgr)
        # Pose for this frame's estimated exposure: kernel start-of-frame minus the configured upstream delay.
        t_query = f.kernel_ns - int(self.a.video_delay_ms * 1e6)
        q = P.PoseQuery(False, 'no_pose_source')
        if self.pose:
            deadline = time.monotonic() + self.a.pose_wait_ms / 1e3
            q = self.pose.query(t_query)
            while q.reason == 'pose_not_yet_available' and time.monotonic() < deadline:
                time.sleep(0.001)
                q = self.pose.query(t_query)
        render_ms = matte_ms = 0.0
        alpha = None
        out = f.image_bgr
        # Tracking loss policy: 'raw' shows the current camera frame (old behaviour); 'hold' keeps compositing
        # over the background rendered from the LAST VALID pose, flagged in the UI, telemetry and take metadata.
        use = None
        if q.valid:
            use = q.sample
            self.held_sample, self.held_since = q.sample, None
        elif self.a.tracking_loss == 'hold' and self.held_sample is not None:
            use = self.held_sample
            if self.held_since is None:
                self.held_since = time.monotonic()
        if signal != 'live':
            reason = 'no_sdi_signal:' + signal
        elif use is None:
            reason = q.reason
        else:
            reason = 'ok' if q.valid else 'held:' + q.reason
            if not q.valid:
                self.stats['held'] = self.stats.get('held', 0) + 1
            r0 = time.perf_counter()
            Tcw = rig.camera_from_world(use)
            K = rig.scaled_K(w, h)
            bg = self._background(Tcw, K, rig.dist, w, h)
            render_ms = (time.perf_counter() - r0) * 1e3
            if self.matte is not None:
                m0 = time.perf_counter()
                alpha = self.matte.infer(f.image_bgr, f.kernel_ns / 1e9)
                matte_ms = (time.perf_counter() - m0) * 1e3
            if self.view == 'overlay' or alpha is None and self.view != 'composite':
                out = VS.overlay(f.image_bgr, bg, self.a.overlay_opacity)
            elif alpha is not None:
                out = VS.composite(f.image_bgr, alpha, bg)
            else:
                out = bg
            self.stats['rendered'] += 1
        if reason != 'ok' and not reason.startswith('held:'):
            self.stats['fallback'][reason.split(':')[0]] = self.stats['fallback'].get(reason.split(':')[0], 0) + 1
        self.stats['frames'] += 1
        done = time.monotonic_ns()
        self.times.append(done)
        self.times = [t for t in self.times if done - t < 2e9]
        fps = (len(self.times) - 1) / ((self.times[-1] - self.times[0]) / 1e9) if len(self.times) > 2 else 0.0
        tele = {'sequence': f.sequence, 'kernel_ns': f.kernel_ns, 'arrival_ns': f.arrival_ns, 'process_start_ns': t_start,
                'done_ns': done, 'signal': signal, 'pose_reason': reason, 'pose_valid': q.valid,
                'pose_clock': q.sample.clock if q.sample else None, 'pose_source': q.sample.source if q.sample else None,
                'pose_bracket_gap_ms': q.bracket_gap_ms, 'video_delay_ms_assumed': self.a.video_delay_ms,
                'render_ms': round(render_ms, 2), 'matte_ms': round(matte_ms, 2), 'calibration': rig.source,
                'calibration_qualified': rig.qualified, 'version': VERSION}
        if self.pose is not None:
            with self.pose.lock:
                last = self.history.latest()
                if last is not None and last.stage == 'Tracking6Dof' and np.isfinite(last.quality) and last.quality > 0:
                    self.sdk_baseline = last.quality if self.sdk_baseline is None else 0.99 * self.sdk_baseline + 0.01 * last.quality
                quality, why = P.tracking_quality(self.history, time.monotonic_ns(), baseline_value=self.sdk_baseline)
        else:
            quality, why = 0.0, 'no tracker'
        tele['picture_inset'] = list(self.inset) if self.inset else None
        tele['tracking_quality'] = round(quality, 3)
        tele['tracking_quality_reason'] = why
        self.last = {'frame': f, 'out': out, 'reason': reason, 'fps': fps, 'tele': tele, 'rig': rig,
                     'quality': quality, 'quality_reason': why}
        if not self.side_by_side:
            return out, tele
        return self._decorate(f, out, reason, fps, tele, rig), tele

    def info_lines(self, reason, tele, rig):
        warn = []
        if self.inset:
            x0, y0, x1, y1 = self.inset
            warn = [f'CAMERA SDI OVERLAYS ON: picture shrunk to {x1 - x0}x{y1 - y0} inside the frame - set the RED SDI output to Clean']
        if reason.startswith('held:'):
            secs = time.monotonic() - self.held_since if self.held_since else 0.0
            lines = [f'TRACKING LOST ({reason[5:]}) - holding last pose {secs:4.1f} s: background does not follow the camera',
                     'calibration: ' + rig.source + ('' if rig.qualified else '  (NOT QUALIFIED)')]
            return warn + lines, [(60, 60, 255)] * len(warn) + [(60, 60, 255), (0, 200, 255)]
        ok = reason == 'ok'
        head = 'REMA ' + {'split': 'output', 'overlay': 'grid overlay (registration check)', 'composite': 'composite'}[self.view]
        lines = [head + ('' if ok else '  -> FALLBACK: current RED frame'),
                 ('pose OK ' if ok else 'pose INVALID: ') + (reason if not ok else f'{tele["pose_source"]} / {tele["pose_clock"]}'),
                 'calibration: ' + rig.source + ('' if rig.qualified else '  (NOT QUALIFIED)')]
        colours = [None, (80, 220, 80) if ok else (60, 60, 255), (0, 200, 255)]
        if self.a.pose == 'synthetic':
            lines.append('SYNTHETIC POSE - NOT TRACKED, pipeline demo only')
            colours.append((0, 200, 255))
        if self.scene is not None:
            lines.append(('background: 3DGS ' + Path(self.a.scene).name + f' ({self.splat.n} splats, SH0)') if self.splat is not None
                         else 'background: grid (3DGS failed: ' + str(self.splat_error)[:50] + ')')
            colours.append(None)
        if self.matte is not None:
            lines.append(f'matte RVM {self.a.matte} {tele["matte_ms"]:.1f} ms (RGB only, no depth)')
            colours.append(None)
        elif self.matte_error:
            lines.append('matte unavailable: ' + self.matte_error[:60])
            colours.append((60, 60, 255))
        return warn + lines, [(60, 60, 255)] * len(warn) + colours

    def _decorate(self, f, out, reason, fps, tele, rig):
        left = f.image_bgr.copy()
        right = out.copy() if out is not f.image_bgr else f.image_bgr.copy()
        ok = reason == 'ok'
        synthetic = self.a.pose == 'synthetic'
        card = (self.capture.desc or {}).get('card', '?')
        src = 'RED SDI via ' + card if 'osprey' in card.lower() else 'SOURCE IS NOT THE OSPREY: ' + card
        put_lines(left, [src + ' (original)', f'{f.width}x{f.height}  seq {f.sequence}  {fps:4.1f} fps out',
                         f'kernel->app {(f.arrival_ns - f.kernel_ns) / 1e6:5.1f} ms   proc {(tele["done_ns"] - tele["process_start_ns"]) / 1e6:5.1f} ms'])
        lines, colours = self.info_lines(reason, tele, rig)
        put_lines(right, lines, colours=colours)
        return np.hstack([left, right])


# ---------------------------------------------------------------------------- front ends

def run_headless(engine, a, out_dir):
    writer, rows = None, []
    t_end = time.monotonic() + a.record
    with (out_dir / 'telemetry.jsonl').open('w') as log:
        while time.monotonic() < t_end:
            img, tele = engine.step()
            if img is None:
                continue
            if not a.no_video:
                if writer is None:
                    writer = TimelineH264(out_dir / 'rig_live.mp4', img.shape[1], img.shape[0])
                writer.write(img, tele['kernel_ns'])
            log.write(json.dumps(tele) + '\n')
            rows.append(tele)
    if writer:
        writer.close()
    return rows


def quality_colour(q):
    """Red (0) -> amber (0.5) -> green (1), as RGB."""
    q = float(np.clip(q, 0, 1))
    return (230, int(60 + 340 * q), 40) if q < 0.5 else (int(230 - 400 * (q - 0.5)), 210, 60)


def run_gui(engine, a, out_dir):
    """Operator view: clean composite fullscreen; top-right picture-in-pictures (camera input, Astra depth),
    each pip_scale of the screen size, always shown; tracking-quality bar; I = text/indicators on/off, R = record take, V = view,
    F = fullscreen, Esc = leave fullscreen, Q = quit."""
    from rema.qt_env import ensure_xcb_cursor
    ensure_xcb_cursor()
    from PySide6.QtCore import QTimer, Qt, QRectF
    from PySide6.QtGui import QImage, QPainter, QColor, QFont, QPen
    from PySide6.QtWidgets import QApplication, QWidget
    from rema.video_out import TakeRecorder
    from rema.astra_depth import DepthStream, colorize
    from PySide6.QtGui import QFontDatabase

    def ui_font(points):
        """Professional sans-serif UI font (Noto Sans, else Ubuntu / DejaVu Sans)."""
        fams = set(QFontDatabase.families())
        f = QFont(next((x for x in ('Noto Sans', 'Inter', 'Roboto', 'Ubuntu', 'DejaVu Sans') if x in fams), 'Sans Serif'))
        f.setPointSizeF(max(5.0, float(points)))
        f.setHintingPreference(QFont.PreferFullHinting)
        return f
    engine.side_by_side = False
    depth = None
    if not a.no_depth:
        depth = DepthStream()
        depth.start()
    app = QApplication(sys.argv)
    app.setApplicationName('REMA')            # X11 window class 'REMA' -> grouped under the Dash icon (StartupWMClass)
    app.setDesktopFileName('rema')

    class View(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('REMA live  [I] info  [R] record  [V] view  [F] fullscreen  [Q] quit')
            self.setStyleSheet('background: black;')
            self.canvas = None
            self.info = True
            self.quality = 0.0
            self.rec = None
            self.msg = ''
            self.msg_until = 0.0

        def paintEvent(self, ev):
            pnt = QPainter(self)
            if self.canvas is None:
                pnt.fillRect(self.rect(), QColor(0, 0, 0))
                pnt.setPen(QColor(200, 200, 200))
                pnt.drawText(self.rect(), Qt.AlignCenter, 'waiting for the first frame...')
                return
            h, w = self.canvas.shape[:2]
            img = QImage(self.canvas.data, w, h, self.canvas.strides[0], QImage.Format_BGR888)
            pnt.drawImage(0, 0, img)
            W, H = self.width(), self.height()
            if self.info:
                pnt.setRenderHint(QPainter.Antialiasing)
                last = engine.last or {}
                f = last.get('frame')
                lines, colours = engine.info_lines(last['reason'], last['tele'], last['rig']) if last else ([], [])
                if f is not None:
                    t = last['tele']
                    card = (engine.capture.desc or {}).get('card', '?')
                    lines = lines + [('source ' + card) if 'osprey' in card.lower() else 'SOURCE IS NOT THE OSPREY: ' + card,
                                     f"{f.width}x{f.height}  {last['fps']:4.1f} fps   processing {(t['done_ns'] - t['process_start_ns']) / 1e6:4.1f} ms"
                                     f"   frame->output {(t['done_ns'] - t['kernel_ns']) / 1e6:4.0f} ms",
                                     'depth: ' + ('off' if depth is None else (('Astra %d frames' % depth.frames) if depth.latest is not None else 'Astra ' + depth.status[:70]))]
                    colours = colours + [None if 'osprey' in card.lower() else (60, 60, 255), None, None if (depth is None or depth.latest is not None) else (0, 160, 255)]
                font = ui_font(max(10, H // 70) * a.text_scale)
                pnt.setFont(font)
                fm = pnt.fontMetrics()
                lh = fm.height() + 2
                pad = max(4, lh // 3)
                y0 = 12 + (lh + pad * 2 if self.rec else 0)
                tw_max = max([fm.horizontalAdvance(t) for t in lines] or [0])
                pnt.fillRect(QRectF(10, y0, tw_max + 2 * pad, lh * len(lines) + pad), QColor(0, 0, 0, 120))
                for i, (text, c) in enumerate(zip(lines, colours + [None] * len(lines))):
                    pnt.setPen(QColor(*(c[::-1] if c else (235, 235, 235))))
                    pnt.drawText(10 + pad, y0 + lh * (i + 1) - fm.descent(), text)
                # tracking quality bar (bottom centre)
                bw, bh = W * 0.32, max(6, H // 120)
                bx, by = (W - bw) / 2, H - bh - max(20, H // 40)
                pnt.fillRect(QRectF(bx - 2, by - 2, bw + 4, bh + 4), QColor(0, 0, 0, 160))
                pnt.fillRect(QRectF(bx, by, bw * self.quality, bh), QColor(*quality_colour(self.quality)))
                pnt.setPen(QPen(QColor(220, 220, 220), 1))
                pnt.drawRect(QRectF(bx, by, bw, bh))
                why = last.get('quality_reason', '') if last else ''
                label = f'TRACKING {self.quality * 100:3.0f}%' + ('' if why in ('good', '') else '  -  ' + why)
                tw = fm.horizontalAdvance(label) + 2 * pad
                pnt.fillRect(QRectF((W - tw) / 2, by - lh - 6, tw, lh + 2), QColor(0, 0, 0, 170))
                pnt.setPen(QColor(*quality_colour(self.quality)) if self.quality < 0.5 else QColor(235, 235, 235))
                pnt.drawText(QRectF((W - tw) / 2, by - lh - 6, tw, lh + 2), Qt.AlignCenter, label)
                if self.rec:
                    rec_text = f'REC   Take{self.rec.take}   {self.rec.seconds():5.1f} s'
                    dot = lh * 0.6
                    pnt.fillRect(QRectF(10, 10, fm.horizontalAdvance(rec_text) + dot + 3 * pad, lh + pad), QColor(0, 0, 0, 170))
                    pnt.setBrush(QColor(230, 30, 30))
                    pnt.setPen(Qt.NoPen)
                    pnt.drawEllipse(QRectF(10 + pad, 10 + (lh + pad - dot) / 2, dot, dot))
                    pnt.setPen(QColor(255, 90, 90))
                    pnt.drawText(QRectF(10 + 2 * pad + dot, 10, fm.horizontalAdvance(rec_text) + pad, lh + pad), Qt.AlignVCenter, rec_text)
                if time.monotonic() < self.msg_until:       # key feedback, hidden with the rest of the info
                    pnt.setFont(ui_font(max(12, H // 45) * a.text_scale))
                    mh = pnt.fontMetrics().height() * 2
                    pnt.fillRect(QRectF(0, H * 0.5 - mh / 2, W, mh), QColor(0, 0, 0, 150))
                    pnt.setPen(QColor(255, 255, 255))
                    pnt.drawText(QRectF(0, H * 0.5 - mh / 2, W, mh), Qt.AlignCenter, self.msg)

        def flash(self, text, seconds=3.0):
            self.msg, self.msg_until = text, time.monotonic() + seconds
            print(time.strftime('%H:%M:%S'), text, flush=True)

        def keyPressEvent(self, ev):
            k = ev.key()
            if k == Qt.Key_Q:
                app.quit()
            elif k == Qt.Key_I:
                self.info = not self.info
            elif k == Qt.Key_V:
                order = ['composite', 'overlay']
                engine.view = order[(order.index(engine.view) + 1) % 2] if engine.view in order else 'composite'
                self.flash('view: ' + engine.view, 1.5)
            elif k == Qt.Key_F:
                self.showNormal() if self.isFullScreen() else self.showFullScreen()
            elif k == Qt.Key_Escape and self.isFullScreen():
                self.showNormal()
            elif k == Qt.Key_R:
                if self.rec is None:
                    last = engine.last
                    if last is None:
                        return
                    h, w = last['out'].shape[:2]
                    try:
                        self.held_at_start = engine.stats.get('held', 0)
                        self.rec = TakeRecorder(w, h, folder=a.take_folder or None, meta={'args': vars(a), 'profile': a.profile, 'scene': a.scene if a.background == 'splat' else None,
                                                              'scene_align': a.scene_align if a.background == 'splat' else None, 'session': str(out_dir)})
                        self.flash(f'RECORDING Take{self.rec.take} -> {self.rec.path.name}')
                    except Exception as e:
                        self.flash('cannot record: ' + str(e))
                else:
                    rec, self.rec = self.rec, None
                    rec.meta['frames_with_held_pose'] = engine.stats.get('held', 0) - getattr(self, 'held_at_start', 0)
                    rc = rec.close()
                    self.flash(f'saved {rec.path.name}  ({rec.seconds():.1f} s, {rec.frames} frames, {rec.dropped} dropped)' if rc == 0
                               else f'Take{rec.take} encoder error {rc}')

    view = View()
    view.resize(1600, 900)
    view.showFullScreen() if not a.windowed else view.show()
    state = {'log': (out_dir / 'telemetry.jsonl').open('w'), 'deadline': time.monotonic() + a.seconds}

    def compose(out, f):
        W, H = max(view.width(), 64), max(view.height(), 64)
        canvas = np.zeros((H, W, 3), np.uint8)
        h, w = out.shape[:2]
        s_ = min(W / w, H / h)
        dw, dh = int(w * s_), int(h * s_)
        x0, y0 = (W - dw) // 2, (H - dh) // 2
        canvas[y0:y0 + dh, x0:x0 + dw] = cv2.resize(out, (dw, dh), interpolation=cv2.INTER_LINEAR)
        if True:                          # picture-in-pictures stay on; I hides text and indicators only
            pw = max(64, int(W * a.pip_scale))
            ph = int(pw * h / w)
            m = max(10, W // 120)
            x = W - m - pw
            pips = [(f.image_bgr, 'CAMERA')]
            if depth is not None:
                d = depth.latest
                if d:
                    pips.append((colorize(d[2]), 'DEPTH (Astra view)'))
                else:
                    blank = np.zeros((480, 640, 3), np.uint8)
                    cv2.putText(blank, 'NO DEPTH', (170, 230), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 160, 255), 4, cv2.LINE_AA)
                    cv2.putText(blank, 'reconnecting...', (190, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
                    pips.append((blank, 'DEPTH (Astra view)'))
            for img, label in pips:
                small = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
                cv2.rectangle(small, (0, 0), (pw - 1, ph - 1), (200, 200, 200), 1)
                cv2.putText(small, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(small, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                canvas[m:m + ph, x:x + pw] = small
                x -= pw + m
        return canvas

    def tick():
        if time.monotonic() > state['deadline'] or not _P.disk_reserve_ok():
            app.quit()
            return
        out, tele = engine.step()
        if out is None:
            return
        state['log'].write(json.dumps(tele) + '\n')
        last = engine.last
        view.quality = 0.7 * view.quality + 0.3 * last['quality']
        if view.rec is not None:
            view.rec.write(out, tele['kernel_ns'])
        view.canvas = compose(out, last['frame'])
        view.update()

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(5)
    # Test aids: --snapshot saves the operator view every 3 s; --script-keys 'R:2,I:9' presses keys at times (s).
    if a.snapshot:
        snap = QTimer()
        snap.timeout.connect(lambda: view.grab().save(str(Path(a.snapshot).expanduser().with_name(
            Path(a.snapshot).stem + time.strftime('_%H%M%S') + Path(a.snapshot).suffix))))
        snap.start(3000)
    if a.script_keys:
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent
        for item in a.script_keys.split(','):
            key, at = item.split(':')
            code = getattr(Qt, 'Key_' + key.upper())
            QTimer.singleShot(int(float(at) * 1000), lambda c=code: view.keyPressEvent(QKeyEvent(QEvent.KeyPress, c, Qt.NoModifier)))
    app.exec()
    if view.rec is not None:
        rec = view.rec
        rec.close()
        print('saved', rec.path, flush=True)
    if depth is not None:
        depth.stop()
    state['log'].close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='Osprey', help='capture device: name fragment (default Osprey) or /dev/videoN')
    ap.add_argument('--pose', choices=['alt2freed', 'bridge', 'freed', 'synthetic', 'none'], default='alt2freed')
    ap.add_argument('--alt2freed', default=str(DEFAULT_ALT2FREED))
    ap.add_argument('--alt-serial', default='')
    ap.add_argument('--freed-port', type=int, default=40000, help='listen port for --pose freed')
    ap.add_argument('--freed-sink-port', type=int, default=40999, help='where alt2freed sends its (unused) FreeD copy')
    ap.add_argument('--trust-freed', action='store_true', help='accept FreeD poses although FreeD has no tracking stage')
    ap.add_argument('--profile', default='', help='VPcamtracking calib profile JSON with lens (+ nodal) solve')
    ap.add_argument('--hfov', type=float, default=60.0, help='provisional horizontal FOV when no profile')
    ap.add_argument('--floor-z', type=float, default=0.0, help='W-frame floor height (m) if tracking origin is not on the floor')
    ap.add_argument('--video-delay-ms', type=float, default=None, help='video vs tracking delay; default: profile timing.video_delay_ms, else 0 (unmeasured)')
    ap.add_argument('--pose-wait-ms', type=float, default=40.0)
    ap.add_argument('--max-pose-gap-ms', type=float, default=60.0)
    ap.add_argument('--tracking-loss', choices=['hold', 'raw'], default='hold',
                    help='on invalid tracking: hold = keep compositing over the last valid pose (flagged); raw = show the camera frame')
    ap.add_argument('--matte', choices=['none', 'cpu', 'cuda'], default='none')
    ap.add_argument('--matte-ratio', type=float, default=0.5, help='RVM downsample ratio; 0.5 chosen 2026-10-24 (fewer soft-edge pixels than 0.375 for +0.8 ms on RTX 5070)')
    ap.add_argument('--view', choices=['split', 'overlay', 'composite'], default='overlay')
    ap.add_argument('--background', choices=['grid', 'splat'], default='grid')
    ap.add_argument('--scene', default=str(_P.DATA / 'scenes/samples/train.splat'), help='3DGS .ply or .splat')
    ap.add_argument('--scene-align', default=str(_P.DATA / 'scenes/samples/train_align.json'), help='scene -> world alignment JSON')
    ap.add_argument('--overlay-opacity', type=float, default=0.5)
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--record', type=float, default=10.0, help='headless recording length (s)')
    ap.add_argument('--no-video', action='store_true', help='headless: telemetry only (no H.264 encoding load)')
    ap.add_argument('--pip-scale', type=float, default=0.10, help='each picture-in-picture width as a fraction of the screen width')
    ap.add_argument('--text-scale', type=float, default=0.3, help='operator text size relative to the original layout (0.3 = 30%%)')
    ap.add_argument('--no-depth', action='store_true', help='operator view without the Astra depth picture-in-picture')
    ap.add_argument('--windowed', action='store_true', help='start in a window instead of fullscreen')
    ap.add_argument('--take-folder', default='', help='where R saves Take<N>_<date_time>.mp4 (default ~/Desktop)')
    ap.add_argument('--snapshot', default='', help='test aid: save the operator view to this PNG path (timestamped) every 3 s')
    ap.add_argument('--script-keys', default='', help="test aid: e.g. 'R:2,R:8,I:9' presses keys at those seconds")
    ap.add_argument('--seconds', type=float, default=7200, help='GUI runtime limit (s)')
    a = ap.parse_args()
    if a.video_delay_ms is None:
        a.video_delay_ms = 0.0
        if a.profile:
            a.video_delay_ms = float(json.loads(Path(a.profile).expanduser().read_text()).get('timing', {}).get('video_delay_ms', 0.0))
    if not _P.disk_reserve_ok():
        sys.exit('25 GiB free-space reserve failed')
    if not 1 <= a.record <= 300 or not 1 <= a.seconds <= 7200:
        ap.error('finite runtime limits required')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = _P.DATA / 'recordings' / f'rig_live_{stamp}'
    out_dir.mkdir(parents=True)
    engine = Engine(a)
    engine.start()
    try:
        rows = run_headless(engine, a, out_dir) if a.headless else run_gui(engine, a, out_dir)
    finally:
        engine.stop()
    rig = engine.rig
    summary = {'created': datetime.datetime.now().astimezone().isoformat(), 'args': vars(a), 'version': VERSION,
               'capture': engine.capture.desc, 'stats': engine.stats,
               'driver_dropped_frames': engine.capture.cap.dropped_by_driver if engine.capture.cap else None,
               'skipped_stale_by_rema': engine.capture.cap.skipped_by_rema if engine.capture.cap else None,
               'pose_samples': engine.pose.count if engine.pose else 0,
               'pose_status': engine.pose.status if engine.pose else None,
               'pose_events': getattr(engine.pose, 'events', [])[-10:] if engine.pose else [],
               'calibration': {'source': rig.source, 'qualified': rig.qualified, 'notes': rig.notes} if rig else None,
               'matte_error': engine.matte_error, 'background': a.background,
               'scene': a.scene if a.background == 'splat' else None, 'splat_error': engine.splat_error,
               'scope': 'Software pipeline record. Not calibration, exposure-sync or glass-to-glass latency qualification.'}
    if a.headless and rows:
        proc = np.array([(r['done_ns'] - r['process_start_ns']) / 1e6 for r in rows])
        age = np.array([(r['done_ns'] - r['kernel_ns']) / 1e6 for r in rows])
        summary['processing_ms_p50_p95_max'] = [round(float(x), 2) for x in (*np.percentile(proc, [50, 95]), proc.max())]
        summary['sof_to_output_ms_p50_p95_max'] = [round(float(x), 2) for x in (*np.percentile(age, [50, 95]), age.max())]
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print('output', out_dir)


if __name__ == '__main__':
    main()
