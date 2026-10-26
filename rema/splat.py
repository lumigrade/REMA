"""3D Gaussian Splatting background renderer (OpenGL 4.3 via ModernGL, headless EGL).

No CUDA compilation: each Gaussian is drawn as an instanced quad whose shape is the EWA-projected 2D
covariance, alpha-blended back to front (the method used by the common real-time web splat viewers).
Rendering uses REMA's camera model exactly: OpenCV camera_from_world, intrinsics K, then the calibrated
lens distortion applied by a precomputed remap, so splats register like the grid does.

Scope / limits (labelled, not hidden):
  - Colour = SH degree 0 (base colour). View-dependent SH terms are ignored (the .splat format has none).
  - Depth order: 16-bit radix sort on CPU every frame (~7 ms for 1 M splats); an optional background-thread
    mode reuses the previous order between refreshes (sort_async=True).
Scene files: 3DGS .ply (x,y,z,f_dc_*,opacity,scale_*,rot_*) or antimatter15 .splat (32 bytes/splat).
Alignment scene -> REMA world (Z up, metres): JSON {"matrix": 4x4} or {"scale", "rotation_deg", "translation"},
optionally "clear_cylinders" to empty the stage area around camera and actor.
"""
import json
import math
import threading
from pathlib import Path

import cv2
import numpy as np

SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------- loading

def load_splat(path):
    """antimatter15 .splat: pos f32x3, scale f32x3 (linear), rgba u8x4 (a = opacity), rot u8x4 (w,x,y,z)*128+128."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 32:
        raise ValueError('not a .splat file (size not a multiple of 32)')
    n = raw.size // 32
    rec = raw.reshape(n, 32)
    pos = rec[:, 0:12].copy().view(np.float32).reshape(n, 3)
    scale = rec[:, 12:24].copy().view(np.float32).reshape(n, 3)
    rgba = rec[:, 24:28].astype(np.float32) / 255.0
    q = (rec[:, 28:32].astype(np.float32) - 128.0) / 128.0
    return {'pos': pos.astype(np.float32), 'scale': scale.astype(np.float32), 'rgb': rgba[:, :3], 'alpha': rgba[:, 3],
            'quat_wxyz': q, 'format': 'splat'}


def load_ply(path):
    """Standard 3DGS binary little-endian PLY."""
    with open(path, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            line = f.readline()
            if not line:
                raise ValueError('bad PLY header')
            header += line
        lines = header.decode('ascii', 'replace').splitlines()
        if 'format binary_little_endian 1.0' not in lines:
            raise ValueError('only binary_little_endian PLY is supported')
        n = next(int(l.split()[2]) for l in lines if l.startswith('element vertex'))
        types = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'uchar': 'u1', 'uint8': 'u1', 'int': 'i4', 'uint': 'u4', 'short': 'i2', 'ushort': 'u2'}
        props = [(l.split()[2], types[l.split()[1]]) for l in lines if l.startswith('property ')]
        data = np.fromfile(f, dtype=np.dtype([(name, '<' + t) for name, t in props]), count=n)
    pos = np.stack([data['x'], data['y'], data['z']], 1).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * np.stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], 1), 0, 1).astype(np.float32)
    alpha = (1.0 / (1.0 + np.exp(-data['opacity'].astype(np.float32)))).astype(np.float32)
    scale = np.exp(np.stack([data['scale_0'], data['scale_1'], data['scale_2']], 1)).astype(np.float32)
    q = np.stack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']], 1).astype(np.float32)
    return {'pos': pos, 'scale': scale, 'rgb': rgb, 'alpha': alpha, 'quat_wxyz': q, 'format': 'ply'}


def load_scene(path):
    path = Path(path).expanduser()
    return load_splat(path) if path.suffix == '.splat' else load_ply(path)


def quat_wxyz_to_mat(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def alignment_matrix(align):
    """4x4 world <- scene (similarity). Either {"matrix": 4x4} or {"scale", "rotation_deg":[rx,ry,rz]
    (R = Rz Ry Rx), "translation"}."""
    if 'matrix' in align:
        return np.asarray(align['matrix'], float)
    rx, ry, rz = [math.radians(v) for v in align.get('rotation_deg', [0, 0, 0])]
    Rx = cv2.Rodrigues(np.array([rx, 0, 0]))[0]
    Ry = cv2.Rodrigues(np.array([0, ry, 0]))[0]
    Rz = cv2.Rodrigues(np.array([0, 0, rz]))[0]
    T = np.eye(4)
    T[:3, :3] = float(align.get('scale', 1.0)) * (Rz @ Ry @ Rx)
    T[:3, 3] = align.get('translation', [0, 0, 0])
    return T


def world_gaussians(scene, align, min_alpha=1.0 / 255):
    """Transform splats into REMA world: centres, 3D covariances (6 unique terms), rgba."""
    keep = scene['alpha'] >= min_alpha
    S = alignment_matrix(align)
    # Stage clearing: drop splats inside vertical cylinders (world coords) so nothing floats in the
    # camera/actor area. {"clear_cylinders": [{"center": [x, y], "radius": r, "z_max": h}, ...]}
    if align.get('clear_cylinders'):
        wpos = scene['pos'] @ S[:3, :3].T + S[:3, 3]
        for cyl in align['clear_cylinders']:
            c = np.asarray(cyl['center'], float)
            inside = (np.hypot(wpos[:, 0] - c[0], wpos[:, 1] - c[1]) < float(cyl['radius'])) & (wpos[:, 2] < float(cyl.get('z_max', 1e9)))
            keep &= ~inside
    sR = S[:3, :3]
    s = float(np.cbrt(np.linalg.det(sR)))
    Rw = sR / s
    pos = scene['pos'][keep] @ sR.T + S[:3, 3]
    R = Rw[None] @ quat_wxyz_to_mat(scene['quat_wxyz'][keep])
    M = R * (scene['scale'][keep] * s)[:, None, :]
    cov = M @ M.transpose(0, 2, 1)
    cov6 = np.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
    rgba = np.concatenate([scene['rgb'][keep], scene['alpha'][keep][:, None]], 1)
    return pos.astype(np.float32), cov6.astype(np.float32), rgba.astype(np.float32)


def estimate_floor(pos, iters=300, thresh=0.02, seed=0):
    """RANSAC dominant plane of splat centres (scene units): (normal, d) with normal.p + d = 0."""
    rng = np.random.default_rng(seed)
    P = pos[rng.choice(len(pos), min(len(pos), 200000), replace=False)].astype(np.float64)
    span = np.percentile(P, 95, 0) - np.percentile(P, 5, 0)
    t = thresh * float(np.linalg.norm(span))
    best = (0, None)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        inl = np.abs(P @ n + d) < t
        if inl.sum() > best[0]:
            best = (int(inl.sum()), (n, d, inl))
    n, d, inl = best[1]
    c = P[inl].mean(0)                       # refine by least squares on inliers
    _, _, vt = np.linalg.svd(P[inl] - c, full_matrices=False)
    n = vt[2]
    if np.mean((P - c) @ n > 0) < 0.5:       # most of the scene should be above the floor
        n = -n
    return n, float(-n @ c), float(inl.mean())


# ---------------------------------------------------------------------------- renderer

VS_SRC = """
#version 430
layout(std430, binding = 0) buffer Centers { float centers[]; };
layout(std430, binding = 1) buffer Covs { float covs[]; };
layout(std430, binding = 2) buffer Colors { float colors[]; };
in vec2 corner;            // quad corner in [-2, 2]
in uint idx;               // splat index (instanced, depth-sorted)
uniform mat4 view;         // camera <- world (OpenCV: x right, y down, z forward)
uniform vec4 intr;         // fx, fy, cx, cy (pixels of the render canvas)
uniform vec2 canvas;       // width, height
uniform float near_z;
out vec4 v_color;
out vec2 v_pos;
void main() {
    vec3 c = vec3(centers[3*idx], centers[3*idx+1], centers[3*idx+2]);
    vec4 t = view * vec4(c, 1.0);
    if (t.z < near_z) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
    float a0 = covs[6*idx], a1 = covs[6*idx+1], a2 = covs[6*idx+2], a3 = covs[6*idx+3], a4 = covs[6*idx+4], a5 = covs[6*idx+5];
    mat3 V = mat3(a0, a1, a2, a1, a3, a4, a2, a4, a5);
    mat3 W = mat3(view);                                       // world -> camera rotation
    float z2 = t.z * t.z;
    mat3 J = mat3(intr.x / t.z, 0.0, 0.0,
                  0.0, intr.y / t.z, 0.0,
                  -intr.x * t.x / z2, -intr.y * t.y / z2, 0.0);   // column-major
    mat3 T = J * W;
    mat3 cov2 = T * V * transpose(T);
    float a = cov2[0][0] + 0.3, b = cov2[0][1], d = cov2[1][1] + 0.3;   // + low-pass filter
    float mid = 0.5 * (a + d);
    float rad = length(vec2(0.5 * (a - d), b));
    float l1 = mid + rad, l2 = max(mid - rad, 0.1);
    vec2 v1 = (abs(b) > 1e-9) ? normalize(vec2(b, l1 - a)) : ((a >= d) ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
    vec2 ax1 = min(sqrt(2.0 * l1), 1024.0) * v1;
    vec2 ax2 = min(sqrt(2.0 * l2), 1024.0) * vec2(v1.y, -v1.x);
    vec2 px = vec2(intr.x * t.x / t.z + intr.z, intr.y * t.y / t.z + intr.w);
    vec2 off = corner.x * ax1 + corner.y * ax2;
    vec2 p = px + off;
    gl_Position = vec4(2.0 * p.x / canvas.x - 1.0, 2.0 * p.y / canvas.y - 1.0, 0.0, 1.0);   // y down kept; read back as-is
    v_color = vec4(colors[4*idx], colors[4*idx+1], colors[4*idx+2], colors[4*idx+3]);
    v_pos = corner;
}
"""

FS_SRC = """
#version 430
in vec4 v_color;
in vec2 v_pos;
out vec4 frag;
void main() {
    float A = -dot(v_pos, v_pos);
    if (A < -4.0) discard;
    float B = exp(A) * v_color.a;
    if (B < 1.0 / 255.0) discard;
    frag = vec4(v_color.rgb * B, B);      // premultiplied
}
"""


class SplatRenderer:
    """Usage: r = SplatRenderer(scene, align, K, dist, (w, h)); bgr = r.render(camera_from_world)."""

    def __init__(self, scene, align, K, dist, size, background_bgr=(46, 44, 42), pad=0.25, near=0.05, sort_async=False):
        import moderngl
        self.w, self.h = size
        self.K = np.asarray(K, float)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, float)
        self.background = np.array(background_bgr, np.float32) / 255.0
        self.near = near
        self.pos, cov6, rgba = world_gaussians(scene, align)
        self.n = len(self.pos)
        # Pinhole canvas with margin, so lens distortion can be applied by remapping into the real frame.
        self.has_dist = bool(np.any(np.abs(self.dist) > 1e-9))
        mx, my = (int(self.w * pad), int(self.h * pad)) if self.has_dist else (0, 0)
        self.cw, self.ch = self.w + 2 * mx, self.h + 2 * my
        self.Kc = self.K.copy()
        self.Kc[0, 2] += mx
        self.Kc[1, 2] += my
        if self.has_dist:
            ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
            pts = np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2)
            und = cv2.undistortPoints(pts, self.K, self.dist, P=self.Kc).reshape(self.h, self.w, 2)
            self.map1, self.map2 = cv2.convertMaps(und[..., 0], und[..., 1], cv2.CV_16SC2)
        self.ctx = moderngl.create_standalone_context(require=430, backend='egl')
        self.prog = self.ctx.program(vertex_shader=VS_SRC, fragment_shader=FS_SRC)
        self.buf_c = self.ctx.buffer(self.pos.astype(np.float32).tobytes())
        self.buf_v = self.ctx.buffer(cov6.tobytes())
        self.buf_col = self.ctx.buffer(rgba.tobytes())
        quad = np.array([-2, -2, 2, -2, 2, 2, -2, -2, 2, 2, -2, 2], np.float32)
        self.buf_quad = self.ctx.buffer(quad.tobytes())
        self.buf_idx = self.ctx.buffer(reserve=self.n * 4, dynamic=True)
        self.vao = self.ctx.vertex_array(self.prog, [(self.buf_quad, '2f', 'corner'), (self.buf_idx, '1u4/i', 'idx')])
        self.fbo = self.ctx.simple_framebuffer((self.cw, self.ch), components=4)
        self.prog['intr'].value = tuple(float(v) for v in (self.Kc[0, 0], self.Kc[1, 1], self.Kc[0, 2], self.Kc[1, 2]))
        self.prog['canvas'].value = (float(self.cw), float(self.ch))
        self.prog['near_z'].value = float(near)
        self.order_view = None
        self.lock = threading.Lock()
        self.pending = None
        self.sort_async = sort_async
        self.sort_event = threading.Event()
        self.stop = False
        self.next_order = None
        if sort_async:
            threading.Thread(target=self._sort_loop, daemon=True).start()

    def _order(self, Tcw):
        """Back-to-front order: 16-bit depth keys so NumPy uses its O(n) radix sort."""
        z = self.pos @ Tcw[2, :3].astype(np.float32) + np.float32(Tcw[2, 3])
        lo, hi = float(z.min()), float(z.max())
        key = (65535 - np.clip((z - lo) * (65535.0 / max(hi - lo, 1e-6)), 0, 65535)).astype(np.uint16)
        return np.argsort(key, kind='stable').astype(np.uint32)   # far -> near

    def _sort_loop(self):
        while not self.stop:
            self.sort_event.wait(0.5)
            self.sort_event.clear()
            with self.lock:
                Tcw = self.pending
            if Tcw is not None:
                order = self._order(Tcw)
                with self.lock:
                    self.next_order = order

    def render(self, camera_from_world):
        Tcw = np.asarray(camera_from_world, float)
        if self.order_view is None or not self.sort_async:
            self.buf_idx.write(self._order(Tcw).tobytes())
            self.order_view = Tcw
        else:
            with self.lock:
                order, self.next_order = self.next_order, None
                self.pending = Tcw
            if order is not None:
                self.buf_idx.write(order.tobytes())
            self.sort_event.set()
        self.buf_c.bind_to_storage_buffer(0)
        self.buf_v.bind_to_storage_buffer(1)
        self.buf_col.bind_to_storage_buffer(2)
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.cw, self.ch)
        bg = self.background[::-1]                                   # BGR -> RGB
        self.ctx.clear(float(bg[0]), float(bg[1]), float(bg[2]), 1.0)  # back-to-front 'over' composites onto it
        self.ctx.enable(self.ctx.BLEND)
        self.ctx.blend_func = (self.ctx.ONE, self.ctx.ONE_MINUS_SRC_ALPHA)
        self.prog['view'].write(Tcw.T.astype(np.float32).tobytes())          # GLSL is column-major
        self.vao.render(mode=self.ctx.TRIANGLES, vertices=6, instances=self.n)
        rgb = np.frombuffer(self.fbo.read(components=3, dtype='f1'), np.uint8).reshape(self.ch, self.cw, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.has_dist:
            bgr = cv2.remap(bgr, self.map1, self.map2, cv2.INTER_LINEAR)
        return bgr

    def close(self):
        self.stop = True
        self.sort_event.set()
        self.ctx.release()


def load_alignment(path):
    return json.loads(Path(path).expanduser().read_text())
