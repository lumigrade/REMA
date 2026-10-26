"""Direct V4L2 (mmap) capture for UVC SDI bridges such as the Osprey VB-USL.

Why not cv2.VideoCapture: we need the kernel's per-buffer sequence number (drop detection)
and timestamp/source flags, kept separate from the application's arrival time.

Clock domains recorded per frame:
  kernel_ns   buffer timestamp from the driver. For uvcvideo this is CLOCK_MONOTONIC at
              start-of-frame reception on the host (V4L2_BUF_FLAG_TSTAMP_SRC_SOE), possibly
              corrected with UVC clock data. It is NOT the RED sensor exposure time.
  arrival_ns  time.monotonic_ns() when REMA dequeued the buffer (same clock as kernel_ns).
The difference is driver-to-application delay only; SDI/bridge/camera delays are upstream
and must be measured physically.
"""
import ctypes
import fcntl
import mmap
import os
import select
import time
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------- ioctl plumbing

def _ioc(direction, nr, size):
    return (direction << 30) | (size << 16) | (ord('V') << 8) | nr


class v4l2_capability(ctypes.Structure):
    _fields_ = [('driver', ctypes.c_char * 16), ('card', ctypes.c_char * 32), ('bus_info', ctypes.c_char * 32),
                ('version', ctypes.c_uint32), ('capabilities', ctypes.c_uint32), ('device_caps', ctypes.c_uint32),
                ('reserved', ctypes.c_uint32 * 3)]


class v4l2_pix_format(ctypes.Structure):
    _fields_ = [('width', ctypes.c_uint32), ('height', ctypes.c_uint32), ('pixelformat', ctypes.c_uint32),
                ('field', ctypes.c_uint32), ('bytesperline', ctypes.c_uint32), ('sizeimage', ctypes.c_uint32),
                ('colorspace', ctypes.c_uint32), ('priv', ctypes.c_uint32), ('flags', ctypes.c_uint32),
                ('ycbcr_enc', ctypes.c_uint32), ('quantization', ctypes.c_uint32), ('xfer_func', ctypes.c_uint32)]


class _fmt_union(ctypes.Union):
    # 200 raw bytes; the kernel union contains pointers, so it is 8-byte aligned.
    _fields_ = [('pix', v4l2_pix_format), ('raw_data', ctypes.c_uint8 * 200), ('_align', ctypes.c_uint64)]


class v4l2_format(ctypes.Structure):
    _fields_ = [('type', ctypes.c_uint32), ('fmt', _fmt_union)]


class v4l2_fract(ctypes.Structure):
    _fields_ = [('numerator', ctypes.c_uint32), ('denominator', ctypes.c_uint32)]


class v4l2_captureparm(ctypes.Structure):
    _fields_ = [('capability', ctypes.c_uint32), ('capturemode', ctypes.c_uint32), ('timeperframe', v4l2_fract),
                ('extendedmode', ctypes.c_uint32), ('readbuffers', ctypes.c_uint32), ('reserved', ctypes.c_uint32 * 4)]


class _parm_union(ctypes.Union):
    _fields_ = [('capture', v4l2_captureparm), ('raw_data', ctypes.c_uint8 * 200)]


class v4l2_streamparm(ctypes.Structure):
    _fields_ = [('type', ctypes.c_uint32), ('parm', _parm_union)]


class v4l2_requestbuffers(ctypes.Structure):
    _fields_ = [('count', ctypes.c_uint32), ('type', ctypes.c_uint32), ('memory', ctypes.c_uint32),
                ('capabilities', ctypes.c_uint32), ('flags', ctypes.c_uint8), ('reserved', ctypes.c_uint8 * 3)]


class timeval(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long), ('tv_usec', ctypes.c_long)]


class v4l2_timecode(ctypes.Structure):
    _fields_ = [('type', ctypes.c_uint32), ('flags', ctypes.c_uint32), ('frames', ctypes.c_uint8),
                ('seconds', ctypes.c_uint8), ('minutes', ctypes.c_uint8), ('hours', ctypes.c_uint8),
                ('userbits', ctypes.c_uint8 * 4)]


class _buf_m(ctypes.Union):
    _fields_ = [('offset', ctypes.c_uint32), ('userptr', ctypes.c_ulong), ('planes', ctypes.c_void_p), ('fd', ctypes.c_int32)]


class v4l2_buffer(ctypes.Structure):
    _fields_ = [('index', ctypes.c_uint32), ('type', ctypes.c_uint32), ('bytesused', ctypes.c_uint32),
                ('flags', ctypes.c_uint32), ('field', ctypes.c_uint32), ('timestamp', timeval),
                ('timecode', v4l2_timecode), ('sequence', ctypes.c_uint32), ('memory', ctypes.c_uint32),
                ('m', _buf_m), ('length', ctypes.c_uint32), ('reserved2', ctypes.c_uint32),
                ('request_fd', ctypes.c_int32)]


class v4l2_fmtdesc(ctypes.Structure):
    _fields_ = [('index', ctypes.c_uint32), ('type', ctypes.c_uint32), ('flags', ctypes.c_uint32),
                ('description', ctypes.c_char * 32), ('pixelformat', ctypes.c_uint32), ('mbus_code', ctypes.c_uint32),
                ('reserved', ctypes.c_uint32 * 3)]


class v4l2_frmsizeenum(ctypes.Structure):
    _fields_ = [('index', ctypes.c_uint32), ('pixel_format', ctypes.c_uint32), ('type', ctypes.c_uint32),
                ('u', ctypes.c_uint32 * 6), ('reserved', ctypes.c_uint32 * 2)]


class v4l2_frmivalenum(ctypes.Structure):
    _fields_ = [('index', ctypes.c_uint32), ('pixel_format', ctypes.c_uint32), ('width', ctypes.c_uint32),
                ('height', ctypes.c_uint32), ('type', ctypes.c_uint32), ('u', ctypes.c_uint32 * 6),
                ('reserved', ctypes.c_uint32 * 2)]


VIDIOC_QUERYCAP = _ioc(2, 0, ctypes.sizeof(v4l2_capability))
VIDIOC_ENUM_FMT = _ioc(3, 2, ctypes.sizeof(v4l2_fmtdesc))
VIDIOC_G_FMT = _ioc(3, 4, ctypes.sizeof(v4l2_format))
VIDIOC_S_FMT = _ioc(3, 5, ctypes.sizeof(v4l2_format))
VIDIOC_REQBUFS = _ioc(3, 8, ctypes.sizeof(v4l2_requestbuffers))
VIDIOC_QUERYBUF = _ioc(3, 9, ctypes.sizeof(v4l2_buffer))
VIDIOC_QBUF = _ioc(3, 15, ctypes.sizeof(v4l2_buffer))
VIDIOC_DQBUF = _ioc(3, 17, ctypes.sizeof(v4l2_buffer))
VIDIOC_STREAMON = _ioc(1, 18, ctypes.sizeof(ctypes.c_int))
VIDIOC_STREAMOFF = _ioc(1, 19, ctypes.sizeof(ctypes.c_int))
VIDIOC_G_PARM = _ioc(3, 21, ctypes.sizeof(v4l2_streamparm))
VIDIOC_S_PARM = _ioc(3, 22, ctypes.sizeof(v4l2_streamparm))
VIDIOC_ENUM_FRAMESIZES = _ioc(3, 74, ctypes.sizeof(v4l2_frmsizeenum))
VIDIOC_ENUM_FRAMEINTERVALS = _ioc(3, 75, ctypes.sizeof(v4l2_frmivalenum))

BUF_TYPE_VIDEO_CAPTURE = 1
MEMORY_MMAP = 1
FLAG_TIMESTAMP_MASK, FLAG_TIMESTAMP_MONOTONIC = 0xE000, 0x2000
FLAG_TSTAMP_SRC_MASK, FLAG_TSTAMP_SRC_SOE = 0x70000, 0x10000
FLAG_ERROR = 0x40


def fourcc(code):
    return sum(ord(c) << (8 * i) for i, c in enumerate(code))


def fourcc_str(value):
    return ''.join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


def _ioctl(fd, request, arg):
    fcntl.ioctl(fd, request, arg)
    return arg


# ---------------------------------------------------------------------------- capture

@dataclass
class SdiFrame:
    image_bgr: np.ndarray
    sequence: int            # kernel buffer sequence (gaps = frames the driver dropped)
    kernel_ns: int           # driver timestamp, see module docstring
    arrival_ns: int          # REMA dequeue time, CLOCK_MONOTONIC
    timestamp_source: str    # 'monotonic_soe', 'monotonic_eof', or 'unknown'
    error_flag: bool
    width: int
    height: int
    pixel_format: str


def find_video_device(name_part):
    """Capture node (/dev/videoN) whose V4L2 card name contains name_part (case-insensitive).
    Device numbers change across reboots and re-plugging, so tools select cameras by name."""
    from pathlib import Path as _Path
    hits = []
    for d in sorted(_Path('/sys/class/video4linux').glob('video*'), key=lambda x: int(x.name[5:])):
        try:
            name, index = (d / 'name').read_text().strip(), (d / 'index').read_text().strip()
        except OSError:
            continue
        if index == '0' and name_part.lower() in name.lower():
            hits.append(('/dev/' + d.name, name))
    if not hits:
        raise RuntimeError(f'no video capture device named like "{name_part}"')
    if len(hits) > 1:
        raise RuntimeError(f'several devices match "{name_part}": {hits}; pass --device /dev/videoN')
    return hits[0][0]


def resolve_device(spec):
    """'/dev/videoN' is used as given; anything else is a device-name fragment, e.g. 'Osprey'."""
    return spec if spec.startswith('/dev/') else find_video_device(spec)


def list_modes(path):
    """All (pixelformat, width, height, fps) the device advertises."""
    fd = os.open(resolve_device(path), os.O_RDWR | os.O_NONBLOCK)
    modes = []
    try:
        i = 0
        while True:
            d = v4l2_fmtdesc(index=i, type=BUF_TYPE_VIDEO_CAPTURE)
            try:
                _ioctl(fd, VIDIOC_ENUM_FMT, d)
            except OSError:
                break
            j = 0
            while True:
                s = v4l2_frmsizeenum(index=j, pixel_format=d.pixelformat)
                try:
                    _ioctl(fd, VIDIOC_ENUM_FRAMESIZES, s)
                except OSError:
                    break
                w, h = s.u[0], s.u[1]   # discrete type only (UVC)
                k = 0
                while True:
                    iv = v4l2_frmivalenum(index=k, pixel_format=d.pixelformat, width=w, height=h)
                    try:
                        _ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, iv)
                    except OSError:
                        break
                    num, den = iv.u[0], iv.u[1]
                    modes.append({'format': fourcc_str(d.pixelformat), 'width': w, 'height': h,
                                  'fps': round(den / num, 3) if num else None})
                    k += 1
                j += 1
            i += 1
    finally:
        os.close(fd)
    return modes


class V4L2Capture:
    """Minimal mmap capture. Always returns the newest frame when asked with latest_only=True,
    counting what it skipped, so the live path never builds a stale queue."""

    def __init__(self, path='Osprey', width=1360, height=768, pixel_format='YUYV', fps=25, buffers=4):
        self.path = resolve_device(path)
        self.fd = os.open(self.path, os.O_RDWR | os.O_NONBLOCK)
        self.maps = []
        self.streaming = False
        try:
            cap = _ioctl(self.fd, VIDIOC_QUERYCAP, v4l2_capability())
            self.card = cap.card.decode(errors='replace')
            self.driver = cap.driver.decode(errors='replace')
            self.bus_info = cap.bus_info.decode(errors='replace')
            f = v4l2_format(type=BUF_TYPE_VIDEO_CAPTURE)
            f.fmt.pix.width, f.fmt.pix.height, f.fmt.pix.pixelformat = width, height, fourcc(pixel_format)
            f.fmt.pix.field = 1  # V4L2_FIELD_NONE
            _ioctl(self.fd, VIDIOC_S_FMT, f)
            self.width, self.height = f.fmt.pix.width, f.fmt.pix.height
            self.pixel_format = fourcc_str(f.fmt.pix.pixelformat)
            self.bytesperline = f.fmt.pix.bytesperline
            self.colorspace = f.fmt.pix.colorspace
            self.quantization = f.fmt.pix.quantization
            self.requested = {'width': width, 'height': height, 'format': pixel_format, 'fps': fps}
            p = v4l2_streamparm(type=BUF_TYPE_VIDEO_CAPTURE)
            p.parm.capture.timeperframe.numerator, p.parm.capture.timeperframe.denominator = 1, int(fps)
            try:
                _ioctl(self.fd, VIDIOC_S_PARM, p)
            except OSError:
                pass
            _ioctl(self.fd, VIDIOC_G_PARM, p)
            tpf = p.parm.capture.timeperframe
            self.negotiated_fps = tpf.denominator / tpf.numerator if tpf.numerator else None
            if self.pixel_format not in ('YUYV', 'UYVY'):
                raise RuntimeError(f'Unsupported negotiated pixel format {self.pixel_format}')
            req = _ioctl(self.fd, VIDIOC_REQBUFS, v4l2_requestbuffers(count=buffers, type=BUF_TYPE_VIDEO_CAPTURE, memory=MEMORY_MMAP))
            for i in range(req.count):
                b = _ioctl(self.fd, VIDIOC_QUERYBUF, v4l2_buffer(index=i, type=BUF_TYPE_VIDEO_CAPTURE, memory=MEMORY_MMAP))
                self.maps.append(mmap.mmap(self.fd, b.length, mmap.MAP_SHARED, mmap.PROT_READ, offset=b.m.offset))
                _ioctl(self.fd, VIDIOC_QBUF, b)
            _ioctl(self.fd, VIDIOC_STREAMON, ctypes.c_int(BUF_TYPE_VIDEO_CAPTURE))
            self.streaming = True
        except BaseException:
            self.close()
            raise
        self.last_sequence = None
        self.dropped_by_driver = 0
        self.skipped_by_rema = 0

    def describe(self):
        return {'device': self.path, 'card': self.card, 'driver': self.driver, 'bus_info': self.bus_info,
                'requested': self.requested, 'negotiated': {'width': self.width, 'height': self.height,
                'format': self.pixel_format, 'fps': self.negotiated_fps, 'bytesperline': self.bytesperline,
                'v4l2_colorspace': self.colorspace, 'v4l2_quantization': self.quantization}}

    def _dequeue(self):
        b = v4l2_buffer(type=BUF_TYPE_VIDEO_CAPTURE, memory=MEMORY_MMAP)
        try:
            _ioctl(self.fd, VIDIOC_DQBUF, b)
        except BlockingIOError:
            return None
        arrival = time.monotonic_ns()
        # Read every field before re-queueing: QBUF writes the struct back with them cleared.
        sequence, flags = b.sequence, b.flags
        kernel_ns = b.timestamp.tv_sec * 1_000_000_000 + b.timestamp.tv_usec * 1000
        raw = np.frombuffer(self.maps[b.index], np.uint8, count=b.bytesused).copy()
        _ioctl(self.fd, VIDIOC_QBUF, b)
        if self.last_sequence is not None and sequence > self.last_sequence + 1:
            self.dropped_by_driver += sequence - self.last_sequence - 1
        self.last_sequence = sequence
        ts = flags & FLAG_TIMESTAMP_MASK
        src = flags & FLAG_TSTAMP_SRC_MASK
        source = ('monotonic_soe' if src == FLAG_TSTAMP_SRC_SOE else 'monotonic_eof') if ts == FLAG_TIMESTAMP_MONOTONIC else 'unknown'
        return raw, sequence, kernel_ns, arrival, source, bool(flags & FLAG_ERROR)

    def read(self, timeout=1.0, latest_only=True):
        """Block until a frame arrives; with latest_only, drain the queue and keep the newest."""
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return None
        got = self._dequeue()
        if got is None:
            return None
        while latest_only:
            newer = self._dequeue()
            if newer is None:
                break
            self.skipped_by_rema += 1
            got = newer
        raw, seq, kns, arr, src, err = got
        expected = self.bytesperline * self.height
        if raw.size < expected:
            err = True
            raw = np.pad(raw, (0, expected - raw.size))
        image = yuv422_to_bgr(raw[:expected].reshape(self.height, self.bytesperline // 2, 2)[:, :self.width],
                              self.pixel_format, limited_range=self.quantization != 1)
        return SdiFrame(image, seq, kns, arr, src, err, self.width, self.height, self.pixel_format)

    def close(self):
        if self.streaming:
            try:
                _ioctl(self.fd, VIDIOC_STREAMOFF, ctypes.c_int(BUF_TYPE_VIDEO_CAPTURE))
            except OSError:
                pass
            self.streaming = False
        for m in self.maps:
            m.close()
        self.maps = []
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def yuv422_to_bgr(packed, pixel_format='YUYV', limited_range=True):
    """Packed 4:2:2 (H x W x 2) -> BGR uint8 with BT.709 (HD SDI) coefficients.
    cv2's YUYV conversion uses BT.601, which shifts HD colours, so convert explicitly."""
    import cv2
    code = cv2.COLOR_YUV2BGR_YUYV if pixel_format == 'YUYV' else cv2.COLOR_YUV2BGR_UYVY
    if not limited_range:
        return cv2.cvtColor(np.ascontiguousarray(packed), code)
    p = packed.astype(np.float32)
    if pixel_format == 'YUYV':
        y = p[..., 0]
        c = p[..., 1]
    else:
        y = p[..., 1]
        c = p[..., 0]
    cb = np.repeat(c[:, 0::2], 2, axis=1)
    cr = np.repeat(c[:, 1::2], 2, axis=1)
    y = (y - 16.0) * (255.0 / 219.0)
    cb = (cb - 128.0) * (255.0 / 224.0)
    cr = (cr - 128.0) * (255.0 / 224.0)
    r = y + 1.5748 * cr
    g = y - 0.1873 * cb - 0.4681 * cr
    b = y + 1.8556 * cb
    return np.clip(np.dstack([b, g, r]) + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------- signal state

def colour_bars_score(bgr, max_bars=10):
    """Detect a synthetic vertical colour-bar card (e.g. the Osprey 'no input' pattern).
    Returns (is_bars, n_bands). Real camera video practically never has every column
    constant from top to bottom with only a handful of distinct values."""
    import cv2
    small = cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    column_std = small.std(axis=0).max(axis=1)              # per-column vertical variation
    vertically_uniform = float(np.mean(column_std < 4.0))
    cols = small.mean(axis=0)
    jumps = np.linalg.norm(np.diff(cols, axis=0), axis=1) > 20
    # A bar edge can straddle two resampled columns; count each run of edge columns once.
    bands = int(np.count_nonzero(np.diff(np.r_[False, jumps].astype(np.int8)) == 1)) + 1
    return vertically_uniform > 0.95 and 3 <= bands <= max_bars, bands


def picture_inset(bgr, dark=2.5, min_border=0.03, max_asymmetry=0.015):
    """Detect a camera monitoring-overlay feed: the picture shrunk into a window with black borders
    (e.g. RED SDI with overlays on: status bars top/bottom, frame lines, picture inset).
    Returns (x0, y0, x1, y1) of the active picture, or None for a clean full-frame feed."""
    g = bgr.astype(np.float32).mean(2)
    h, w = g.shape
    cols = g[h // 4: 3 * h // 4].mean(0)              # middle rows: avoid status bars when finding side borders
    left = int(np.argmax(cols > dark))
    right = w - int(np.argmax(cols[::-1] > dark))
    if left < w * min_border or (w - right) < w * min_border:
        return None                                     # picture reaches both sides: clean feed
    if abs(left - (w - right)) > w * max_asymmetry:
        return None                                     # dark scenery on one side, not a centred inset
    if g[h // 4: 3 * h // 4, :left].max(initial=0) > 4 * dark or g[h // 4: 3 * h // 4, right:].max(initial=0) > 4 * dark:
        return None                                     # borders must be pure black, not dark picture
    band = g[:, left + 5:right - 5]
    rows_dark = band.mean(1) < dark
    top = int(np.argmax(~rows_dark))
    bottom = h - int(np.argmax(~rows_dark[::-1]))
    return left, top, right, bottom


class SignalMonitor:
    """Classifies the feed as 'live', 'colour_bars' (no SDI input) or 'frozen'."""

    def __init__(self, frozen_after=25):
        self.prev = None
        self.identical = 0
        self.frozen_after = frozen_after

    def update(self, bgr):
        small = bgr[::8, ::8]
        if self.prev is not None and small.shape == self.prev.shape and np.abs(small.astype(np.int16) - self.prev).max() <= 2:
            self.identical += 1
        else:
            self.identical = 0
        self.prev = small.astype(np.int16)
        bars, _ = colour_bars_score(bgr)
        if bars:
            return 'colour_bars'
        if self.identical >= self.frozen_after:
            return 'frozen'
        return 'live'
