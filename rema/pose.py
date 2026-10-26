"""Antilatency camera pose for the RED rig: samples, timestamped history, rig transform.

Coordinate frames (all metres):
  A  Antilatency SDK raw: left-handed, X right, Y up, Z forward (Unity style). Quaternion (x, y, z, w).
  S  VPcamtracking calibration frame: A with Y negated -> right-handed, Y down. Matches
     alt2freed/ui/calib.py so its lens + nodal (Alt -> camera) results load unchanged.
  W  REMA render world: right-handed, X right, Y forward, Z up (same axes as alt2freed FreeD output).
  C  OpenCV camera: X right, Y down, Z along the optical axis.

REMA uses the RAW tracker pose (identity Antilatency placement, no SDK extrapolation) plus its own
calibrated T_tracker_cam. Never also apply an Antilatency placement, or the offset is applied twice.

Time: every sample carries a host CLOCK_MONOTONIC time and a label for what that time means.
History lookups interpolate to the requested video time; they never hold a stale pose.
"""
import bisect
import json
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

S = np.diag([1.0, -1.0, 1.0])
W_FROM_S = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])   # (xs, ys, zs) -> (xs, zs, -ys)
VALID_STAGES = frozenset({'Tracking6Dof'})


# ---------------------------------------------------------------------------- math

def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(m):
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0.0] * 4
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def slerp(q0, q1, a):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


def rt(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return rt(R.T, -R.T @ t)


def unity_euler_to_mat(x, y, z):
    """Unity Euler (radians): R = Ry(y) @ Rx(x) @ Rz(z), in Antilatency's left-handed frame."""
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def mat_to_unity_euler(m):
    """Inverse of unity_euler_to_mat; identical to alt2freed quatToYawPitchRoll (pitch, yaw, roll)."""
    x = math.asin(max(-1.0, min(1.0, -m[1, 2])))
    y = math.atan2(m[0, 2], m[2, 2])
    z = math.atan2(m[1, 0], m[1, 1])
    return x, y, z


# ---------------------------------------------------------------------------- samples

@dataclass(frozen=True)
class PoseSample:
    t_ns: int                 # host CLOCK_MONOTONIC
    position: tuple           # A frame, metres (raw tracker, identity placement)
    rotation: tuple           # A frame quaternion (x, y, z, w)
    stage: str                # Antilatency stability stage
    quality: float = float('nan')
    clock: str = 'unknown'    # what t_ns means, e.g. 'host_arrival_pipe' or 'sdk_query_monotonic'
    source: str = 'unknown'


def tracker_rh(position, rotation):
    """Raw Antilatency tracker pose -> 4x4 world<-tracker in the right-handed S frame."""
    return rt(S @ quat_to_mat(rotation) @ S, S @ np.asarray(position, float))


def parse_alt2freed_json(line, arrival_ns):
    """One `alt2freed stream --json` state line -> PoseSample using its raw 'tp'/'tq' tracker pose.
    Those lines are printed at most every 33 ms and carry no SDK time, so the timestamp is the
    pipe arrival time (clock 'host_arrival_pipe'). Returns None for event lines."""
    try:
        msg = json.loads(line)
    except ValueError:
        return None
    if 'tp' not in msg or 'tq' not in msg:
        return None
    return PoseSample(int(arrival_ns), tuple(msg['tp']), tuple(msg['tq']), msg.get('stage', 'unknown'),
                      float(msg.get('q', 'nan')), 'host_arrival_pipe', 'alt2freed_json')


def parse_bridge_json(line):
    """One line from tools/alt_pose_bridge (REMA native bridge): sampled in-process with
    CLOCK_MONOTONIC at the SDK query (clock 'sdk_query_monotonic')."""
    try:
        msg = json.loads(line)
    except ValueError:
        return None
    if 't_ns' not in msg or 'p' not in msg:
        return None
    return PoseSample(int(msg['t_ns']), tuple(msg['p']), tuple(msg['q']), msg.get('stage', 'unknown'),
                      float(msg.get('quality', 'nan')), 'sdk_query_monotonic', 'alt_pose_bridge')


def decode_freed_d1(packet):
    """FreeD D1 (29 bytes) -> dict(pan, tilt, roll degrees; x, y, z metres; zoom, focus raw). Checks checksum."""
    if len(packet) != 29 or packet[0] != 0xD1:
        raise ValueError('not a FreeD D1 packet')
    if (0x40 - sum(packet[:28])) & 0xFF != packet[28]:
        raise ValueError('FreeD checksum mismatch')

    def s24(b):
        v = int.from_bytes(b, 'big')
        return v - (1 << 24) if v & 0x800000 else v
    return {'camera_id': packet[1], 'pan': s24(packet[2:5]) / 32768.0, 'tilt': s24(packet[5:8]) / 32768.0,
            'roll': s24(packet[8:11]) / 32768.0, 'x': s24(packet[11:14]) / 64000.0, 'y': s24(packet[14:17]) / 64000.0,
            'z': s24(packet[17:20]) / 64000.0, 'zoom': int.from_bytes(packet[20:23], 'big'),
            'focus': int.from_bytes(packet[23:26], 'big')}


def freed_to_sample(d, arrival_ns, source='freed_udp'):
    """Invert alt2freed's FreeD mapping (default signs/offsets): pan=yaw, tilt=-pitch, roll=roll,
    X=x, Y=z, Z=y. FreeD carries no tracking stage, timestamp or quality: the stage is reported as
    'FreeD_unknown' so the history treats it as untrusted unless explicitly allowed.
    Note alt2freed applies its smoothing and any placement before sending FreeD."""
    R = unity_euler_to_mat(math.radians(-d['tilt']), math.radians(d['pan']), math.radians(d['roll']))
    return PoseSample(int(arrival_ns), (d['x'], d['z'], d['y']), tuple(mat_to_quat(R)), 'FreeD_unknown',
                      float('nan'), 'host_arrival_udp', source)


# ---------------------------------------------------------------------------- history

@dataclass
class PoseQuery:
    valid: bool
    reason: str
    sample: PoseSample | None = None
    bracket_gap_ms: float | None = None


class PoseHistory:
    """Bounded history; interpolates pose to a requested time (e.g. an estimated RED exposure)."""

    def __init__(self, seconds=2.0, max_gap_ms=60.0, max_extrapolate_ms=4.0, valid_stages=VALID_STAGES):
        self.seconds = seconds
        self.max_gap_ns = int(max_gap_ms * 1e6)
        self.max_extrapolate_ns = int(max_extrapolate_ms * 1e6)
        self.valid_stages = frozenset(valid_stages)
        self.samples = deque()
        self.times = deque()
        self.rejected_non_monotonic = 0

    def add(self, s):
        if self.times and s.t_ns <= self.times[-1]:
            self.rejected_non_monotonic += 1
            return False
        self.samples.append(s)
        self.times.append(s.t_ns)
        horizon = s.t_ns - int(self.seconds * 1e9)
        while self.times and self.times[0] < horizon:
            self.times.popleft()
            self.samples.popleft()
        return True

    def latest(self):
        return self.samples[-1] if self.samples else None

    def at(self, t_ns):
        if not self.samples:
            return PoseQuery(False, 'no_pose_samples')
        times = list(self.times)
        if t_ns < times[0]:
            return PoseQuery(False, 'pose_expired')
        i = bisect.bisect_right(times, t_ns)
        if i == len(times):
            s = self.samples[-1]
            if t_ns - s.t_ns > self.max_extrapolate_ns:
                return PoseQuery(False, 'pose_not_yet_available', s)
            if s.stage not in self.valid_stages:
                return PoseQuery(False, 'tracking_stage:' + s.stage, s)
            return PoseQuery(True, 'nearest_newest', s, (t_ns - s.t_ns) / 1e6)
        a, b = self.samples[i - 1], self.samples[i]
        gap = b.t_ns - a.t_ns
        for s in (a, b):
            if s.stage not in self.valid_stages:
                return PoseQuery(False, 'tracking_stage:' + s.stage, s, gap / 1e6)
        if gap > self.max_gap_ns:
            return PoseQuery(False, 'pose_gap', a, gap / 1e6)
        u = (t_ns - a.t_ns) / gap
        p = tuple((1 - u) * np.asarray(a.position) + u * np.asarray(b.position))
        q = tuple(slerp(a.rotation, b.rotation, u))
        return PoseQuery(True, 'interpolated', PoseSample(int(t_ns), p, q, a.stage, min(a.quality, b.quality),
                                                          a.clock, a.source), gap / 1e6)


# ---------------------------------------------------------------------------- operator quality

def tracking_quality(history, now_ns, window_s=0.5, baseline_value=None):
    """Operator-facing tracking quality 0..1 plus the reason for the weakest factor.
    Factors: stage (Tracking6Dof good, blind 6DoF weak, 3DoF/initialising zero), freshness (samples still
    arriving: radio dropouts), position jitter (noise from second differences of the 250 Hz stream:
    ~1 mm is normal, >=5 mm is bad) and the SDK stability value relative to its usual level.
    A display aid, not a qualification metric."""
    s = history.samples
    if not s:
        return 0.0, 'no tracker samples'
    last = s[-1]
    age_ms = (now_ns - last.t_ns) / 1e6
    fresh = float(np.clip(1.0 - (age_ms - 60.0) / 140.0, 0.0, 1.0))      # full until 60 ms, zero at 200 ms
    stage = {'Tracking6Dof': 1.0, 'TrackingBlind6Dof': 0.3}.get(last.stage, 0.0)
    win = [x for x in s if x.t_ns >= last.t_ns - window_s * 1e9]
    jitter_mm = float('nan')
    jit = 1.0
    if len(win) >= 10:
        Pp = np.array([x.position for x in win])
        d2 = Pp[2:] - 2 * Pp[1:-1] + Pp[:-2]                               # motion cancels at 4 ms steps
        jitter_mm = float(np.sqrt(np.mean(np.sum(d2 ** 2, 1)) / 6.0) * 1000)   # var(d2) = 6 sigma^2 per axis-sum
        jit = float(np.clip(1.0 - (jitter_mm - 1.5) / 3.5, 0.0, 1.0))     # full <=1.5 mm, zero at 5 mm
    sdk = 1.0
    if baseline_value and np.isfinite(last.quality) and baseline_value > 0:
        sdk = float(np.clip(last.quality / baseline_value, 0.0, 1.0))
    factors = {'tracking stage ' + last.stage: stage, 'tracker samples late (%.0f ms)' % age_ms: fresh,
               'position jitter %.1f mm' % jitter_mm: jit, 'SDK stability low': sdk}
    worst = min(factors, key=factors.get)
    score = stage * fresh * min(jit, sdk)
    return float(score), (worst if factors[worst] < 0.999 else 'good')


# ---------------------------------------------------------------------------- rig

@dataclass
class RigCalibration:
    """Lens + Alt->camera + world floor. `qualified` stays False until independently validated."""
    K: np.ndarray
    dist: np.ndarray
    size: tuple                                      # (width, height) the lens was solved at
    T_tracker_cam: np.ndarray = field(default_factory=lambda: np.eye(4))   # OpenCV camera in S-tracker frame
    floor_z_m: float = 0.0                           # W-frame height of the physical floor
    source: str = 'provisional'
    qualified: bool = False
    notes: list = field(default_factory=list)

    def world_from_camera(self, sample):
        T_S_cam = tracker_rh(sample.position, sample.rotation) @ self.T_tracker_cam
        T = rt(W_FROM_S, np.zeros(3)) @ T_S_cam
        T[2, 3] -= self.floor_z_m                    # render floor at z = 0
        return T

    def camera_from_world(self, sample):
        return inv(self.world_from_camera(sample))

    def scaled_K(self, width, height):
        """Intrinsics for a different delivered resolution (same framing only, never a crop)."""
        sx, sy = width / self.size[0], height / self.size[1]
        K = self.K.copy()
        K[0] *= sx
        K[1] *= sy
        return K


def provisional_rig(width, height, hfov_deg=60.0):
    """Uncalibrated placeholder so the pipeline can run; renders WILL NOT line up with reality."""
    f = width / (2 * math.tan(math.radians(hfov_deg) / 2))
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    return RigCalibration(K, np.zeros(5), (width, height), source='provisional_uncalibrated',
                          notes=['Nominal %.0f deg HFOV, zero distortion, identity Alt->camera.' % hfov_deg])


def rig_from_vpcamtracking(profile):
    """Load a VPcamtracking calib profile (alt2freed/calib/profiles/<name>.json).
    Uses its lens solve (K, dist at the solved size) and nodal T_tracker_cam if present."""
    if isinstance(profile, (str, bytes)) or hasattr(profile, 'read_text'):
        from pathlib import Path
        profile = json.loads(Path(profile).read_text())
    lens = profile.get('lens')
    if not lens:
        raise ValueError('profile has no lens calibration')
    K = np.asarray(lens['K'], float)
    centre = profile.get('imageCenter') or {}
    K[0, 2] += float(centre.get('dx', 0.0))
    K[1, 2] += float(centre.get('dy', 0.0))
    rig = RigCalibration(K, np.asarray(lens['dist'], float), tuple(lens['size']), source='vpcamtracking:' + profile.get('name', '?'))
    rig.notes.append('lens rms %.3f px over %d samples' % (lens.get('rms', float('nan')), lens.get('samples', 0)))
    nodal = profile.get('nodal')
    if nodal and 'T_tracker_cam' in nodal:
        rig.T_tracker_cam = np.asarray(nodal['T_tracker_cam'], float)
        rig.notes.append('nodal reprojection rms %.2f px over %d samples' % (nodal.get('reprojRmsPx', float('nan')), nodal.get('samples', 0)))
    else:
        rig.notes.append('No nodal (Alt->camera) solve: identity offset, overlay will not register.')
    return rig
