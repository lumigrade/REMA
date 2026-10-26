"""Live Astra Pro depth for the operator view (picture-in-picture), via tools/astra_depth_helper.py.

Raw Astra depth in the Astra's own view: it is NOT reprojected into the RED camera (that needs the
Astra->RED calibration, still pending). Latest frame only; errors are reported, never hidden.
"""
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from rema import paths as _P

HEADER_SIZE = 32


class DepthStream(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.latest = None            # (seq, host_ns, depth metres float32 480x640); None while reconnecting
        self.error = None
        self.status = 'starting'
        self.restarts = 0
        self.frames = 0
        self.proc = None
        self.stop_event = threading.Event()

    def run(self):
        """Keep a depth helper running; if it times out or the Astra disappears, retry with back-off."""
        backoff = 1.0
        while not self.stop_event.is_set():
            got = self._run_once()
            if self.stop_event.is_set():
                break
            self.restarts += 1
            self.status = f'reconnecting (attempt {self.restarts}): {self.error or "stream ended"}'
            backoff = 1.0 if got else min(backoff * 2, 10.0)
            self.stop_event.wait(backoff)

    def _run_once(self):
        env = dict(os.environ, LD_PRELOAD=str(_P.WORK / 'astra_usb_compat.so'))
        try:
            self.proc = subprocess.Popen([sys.executable, str(_P.ROOT / 'tools/astra_depth_helper.py')], env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            self.error = f'cannot start depth helper: {e}'
            return False
        pipe = self.proc.stdout
        got = False
        while not self.stop_event.is_set():
            head = pipe.read(HEADER_SIZE)
            if len(head) < HEADER_SIZE:
                break
            magic, n, seq, host_ns, scale = np.frombuffer(head[:4], 'S4')[0], *np.frombuffer(head[4:28], '<u4,<u8,<u8,<f4')[0]
            if magic != b'RDEP' or n != 640 * 480:
                self.error = 'bad depth stream header'
                break
            body = pipe.read(n * 2)
            if len(body) < n * 2:
                break
            depth = np.frombuffer(body, np.uint16).reshape(480, 640).astype(np.float32) * float(scale)
            self.latest = (int(seq), int(host_ns), depth)
            self.frames += 1
            self.error, self.status, got = None, 'streaming', True
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        msg = (self.proc.stderr.read() or b'').decode(errors='replace').strip()
        self.error = (msg.splitlines()[-1] if msg else 'depth stream ended')
        self.latest = None
        return got

    def stop(self):
        self.stop_event.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def colorize(depth, near=0.5, far=6.0):
    """Turbo colour map, near = red/warm; invalid (0) = black."""
    v = np.clip((depth - near) / (far - near), 0, 1)
    img = cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth <= 0] = 0
    return img
