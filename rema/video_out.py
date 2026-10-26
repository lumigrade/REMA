"""Labeled H.264 progress videos via GStreamer x264 (ffmpeg is not installed on the rig host).

Frames are placed on their real capture timeline: when the source delivers fewer frames than the
output rate, the previous frame repeats, so playback speed stays truthful.
"""
import subprocess

import cv2


class TimelineH264:
    def __init__(self, path, width, height, fps=25, bitrate_kbps=12000):
        self.path, self.fps = str(path), fps
        self.size = (width, height)
        cmd = ['gst-launch-1.0', '-q', 'fdsrc', 'fd=0', '!', 'rawvideoparse', 'format=bgr', f'width={width}',
               f'height={height}', f'framerate={int(fps)}/1', '!', 'videoconvert', '!',
               'x264enc', 'speed-preset=veryfast', f'bitrate={int(bitrate_kbps)}', '!',
               'video/x-h264,profile=high', '!', 'mp4mux', '!', 'filesink', f'location={self.path}']
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.first_ns = None
        self.written = 0

    def write(self, bgr, t_ns):
        if bgr.shape[1::-1] != self.size:
            raise ValueError('frame size changed during recording')
        if self.first_ns is None:
            self.first_ns = t_ns
        slots = int((t_ns - self.first_ns) / 1e9 * self.fps) + 1 - self.written
        data = bgr.tobytes()
        for _ in range(max(slots, 0)):
            self.proc.stdin.write(data)
        self.written += max(slots, 0)

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        return self.proc.wait()


def put_lines(img, lines, origin=(14, 30), scale=0.7, colours=None):
    """White text with dark outline, one entry per line; colours optional per line (BGR)."""
    x, y = origin
    for i, text in enumerate(lines):
        c = (colours[i] if colours and i < len(colours) and colours[i] else (255, 255, 255))
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c, 1, cv2.LINE_AA)
        y += int(34 * scale / 0.7)
    return img


def notify(title, body):
    """Desktop notification (GNOME) so a finished background analysis is visible; never fails the caller."""
    import shutil
    if shutil.which('notify-send'):
        try:
            subprocess.run(['notify-send', '-a', 'REMA', title, body], timeout=5)
        except Exception:
            pass


class TakeRecorder:
    """R-key takes: ~/Desktop/Take<N>_<YYYYmmdd_HHMMSS>.mp4, H.265 (NVENC, else libx265), real-time
    timeline at `fps` (repeats fill gaps). Encoding runs in a thread with a bounded queue; frames
    that cannot be queued are counted, never silently lost."""

    def __init__(self, width, height, fps=25, folder=None, meta=None):
        import json, queue, re, shutil, threading, datetime
        from pathlib import Path
        self.folder = Path(folder or Path.home() / 'Desktop')
        self.folder.mkdir(parents=True, exist_ok=True)
        nums = [int(m.group(1)) for p in self.folder.glob('Take*_*.mp4') if (m := re.match(r'Take(\d+)_', p.name))]
        self.take = max(nums, default=0) + 1
        self.started = datetime.datetime.now()
        self.path = self.folder / f'Take{self.take}_{self.started.strftime("%Y%m%d_%H%M%S")}.mp4'
        self.size, self.fps = (width, height), fps
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            raise RuntimeError('ffmpeg not found')
        self.codec = 'hevc_nvenc' if 'hevc_nvenc' in subprocess.run([ffmpeg, '-hide_banner', '-encoders'], capture_output=True, text=True).stdout else 'libx265'
        enc = ['-c:v', 'hevc_nvenc', '-preset', 'p5', '-tune', 'hq', '-rc', 'vbr', '-cq', '21', '-b:v', '0'] if self.codec == 'hevc_nvenc' \
            else ['-c:v', 'libx265', '-preset', 'fast', '-crf', '21']
        self.proc = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
                                      '-framerate', str(fps), '-i', '-', *enc, '-pix_fmt', 'yuv420p', '-tag:v', 'hvc1', '-movflags', '+faststart',
                                      str(self.path)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.q = queue.Queue(maxsize=8)
        self.first_ns, self.written, self.frames, self.dropped = None, 0, 0, 0
        self.meta = dict(meta or {}, file=str(self.path), take=self.take, codec=self.codec, size=[width, height], fps=fps,
                         started=self.started.isoformat())
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            bgr, t_ns = item
            if self.first_ns is None:
                self.first_ns = t_ns
            slots = int((t_ns - self.first_ns) / 1e9 * self.fps) + 1 - self.written
            data = bgr.tobytes()
            try:
                for _ in range(max(slots, 0)):
                    self.proc.stdin.write(data)
            except BrokenPipeError:
                break
            self.written += max(slots, 0)

    def write(self, bgr, t_ns):
        if bgr.shape[1::-1] != self.size:
            raise ValueError('frame size changed during take')
        try:
            self.q.put_nowait((bgr, t_ns))
            self.frames += 1
        except Exception:
            self.dropped += 1

    def seconds(self):
        return self.written / self.fps

    def close(self):
        import json, datetime
        self.q.put(None)
        self.thread.join(timeout=30)
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        rc = self.proc.wait(timeout=60)
        err = (self.proc.stderr.read() or b'').decode(errors='replace').strip()
        self.meta.update(stopped=datetime.datetime.now().isoformat(), source_frames=self.frames, dropped_frames=self.dropped,
                         output_frames=self.written, ffmpeg_exit=rc, ffmpeg_error=err[-500:] or None)
        from rema import paths as _P
        d = _P.DATA / 'recordings' / 'takes'
        d.mkdir(parents=True, exist_ok=True)
        (d / (self.path.stem + '.json')).write_text(json.dumps(self.meta, indent=2, default=str))
        return rc
