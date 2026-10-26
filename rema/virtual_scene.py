"""Milestone-1 virtual background: metric floor grid + simply shaded blocks in the REMA world
frame W (right-handed, X right, Y forward, Z up, floor at z = 0, metres).

Rendering uses the same camera_from_world, K and distortion as the video frame it will be
composited with. Geometry behind the near plane is dropped rather than wrapped.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

NEAR_M = 0.05


@dataclass
class Block:
    x: float          # min corner on the floor, metres
    y: float
    size_x: float
    size_y: float
    height: float
    bgr: tuple


@dataclass
class Scene:
    grid_half_extent_m: float = 6.0
    grid_step_m: float = 0.5
    major_every: int = 2                       # every 1 m
    sky_top_bgr: tuple = (52, 40, 32)
    sky_horizon_bgr: tuple = (96, 82, 70)
    floor_bgr: tuple = (46, 44, 42)
    line_bgr: tuple = (92, 90, 86)
    major_bgr: tuple = (120, 118, 112)
    x_axis_bgr: tuple = (60, 60, 210)
    y_axis_bgr: tuple = (80, 170, 80)
    light_dir: tuple = (-0.4, -0.6, 0.7)       # towards the light, W frame
    blocks: list = field(default_factory=lambda: [
        Block(0.8, 1.5, 0.5, 0.5, 0.5, (70, 130, 200)),
        Block(-1.6, 2.5, 0.6, 0.4, 1.0, (180, 120, 70)),
        Block(0.2, 3.6, 0.8, 0.8, 0.35, (100, 170, 120)),
    ])


def _project(points_w, camera_from_world, K, dist):
    pts = np.asarray(points_w, float)
    cam = pts @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    uv, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return cam, uv.reshape(-1, 2)


def _clip_segment(a, b):
    """Clip a camera-space segment to z >= NEAR_M. Returns None if fully behind."""
    if a[2] < NEAR_M and b[2] < NEAR_M:
        return None
    if a[2] >= NEAR_M and b[2] >= NEAR_M:
        return a, b
    u = (NEAR_M - a[2]) / (b[2] - a[2])
    c = a + u * (b - a)
    return (c, b) if a[2] < NEAR_M else (a, c)


def _draw_line(img, a_w, b_w, camera_from_world, K, dist, colour, thickness, samples=24):
    R, t = camera_from_world[:3, :3], camera_from_world[:3, 3]
    a, b = R @ a_w + t, R @ b_w + t
    seg = _clip_segment(a, b)
    if seg is None:
        return
    # Sample along the segment so lens distortion bends long lines correctly.
    u = np.linspace(0, 1, samples)[:, None]
    pts = seg[0] * (1 - u) + seg[1] * u
    uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    uv = uv.reshape(-1, 2)
    ok = np.isfinite(uv).all(axis=1) & (np.abs(uv).max(axis=1) < 1e5)
    if ok.sum() >= 2:
        cv2.polylines(img, [np.rint(uv[ok] * 16).astype(np.int32)], False, colour, thickness, cv2.LINE_AA, shift=4)


def _clip_halfplane(poly, a, b, c):
    """Sutherland-Hodgman: keep the part of a polygon where a*x + b*y + c <= 0 (below the horizon)."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        fp, fq = a * p[0] + b * p[1] + c, a * q[0] + b * q[1] + c
        if fp <= 0:
            out.append(p)
        if (fp < 0 < fq) or (fq < 0 < fp):
            u = fp / (fp - fq)
            out.append((p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1])))
    return out


def _draw_lines(img, segs, styles, camera_from_world, K, dist, samples=24):
    """Batched version of _draw_line: one projectPoints call for every sampled grid point."""
    R, t = camera_from_world[:3, :3], camera_from_world[:3, 3]
    u = np.linspace(0, 1, samples)[:, None]
    keep, pts = [], []
    for k, (a_w, b_w) in enumerate(segs):
        seg = _clip_segment(R @ a_w + t, R @ b_w + t)
        if seg is None:
            continue
        keep.append(k)
        pts.append(seg[0] * (1 - u) + seg[1] * u)
    if not pts:
        return
    uv, _ = cv2.projectPoints(np.concatenate(pts).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    uv = uv.reshape(len(keep), samples, 2)
    for k, line in zip(keep, uv):
        ok = np.isfinite(line).all(axis=1) & (np.abs(line).max(axis=1) < 1e5)
        if ok.sum() >= 2:
            colour, th = styles[k]
            cv2.polylines(img, [np.rint(line[ok] * 16).astype(np.int32)], False, colour, th, cv2.LINE_AA, shift=4)


def render(camera_from_world, K, dist, size, scene=None):
    """Return a BGR uint8 background image (h, w, 3) for this exact camera pose."""
    scene = scene or Scene()
    w, h = size
    K = np.asarray(K, float)
    dist = np.asarray(dist, float) if dist is not None else np.zeros(5)
    img = np.empty((h, w, 3), np.uint8)

    # Sky/floor split at the projected horizon (direction-only points at infinity).
    R = camera_from_world[:3, :3]
    up_c = R @ np.array([0, 0, 1.0])
    # Sky/floor colours are smooth except at the horizon: build them on a coarse grid (pinhole,
    # ignoring distortion) and upsample once in uint8.
    step = 8
    ys, xs = np.mgrid[0:h + step:step, 0:w + step:step].astype(np.float32)
    rays = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs)], -1)
    elev = ((rays @ up_c) / np.linalg.norm(rays, axis=-1)).astype(np.float32)
    t = np.clip(elev, 0, 1)[..., None]
    sky_col = np.asarray(scene.sky_horizon_bgr, np.float32) * (1 - t) + np.asarray(scene.sky_top_bgr, np.float32) * t
    full = (elev.shape[1] * step, elev.shape[0] * step)
    img[:] = cv2.resize(sky_col.astype(np.uint8), full, interpolation=cv2.INTER_LINEAR)[:h, :w]
    # For a pinhole camera the horizon is exactly the image line a*x + b*y + c = 0 (sign of up . ray):
    # fill the floor half-plane as a polygon - sharp, exact and fast.
    a, b = up_c[0] / K[0, 0], up_c[1] / K[1, 1]
    c = up_c[2] - a * K[0, 2] - b * K[1, 2]
    poly = _clip_halfplane([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)], a, b, c)
    if len(poly) >= 3:
        cv2.fillPoly(img, [np.rint(np.array(poly) * 16).astype(np.int32)], scene.floor_bgr, cv2.LINE_AA, shift=4)

    e, s = scene.grid_half_extent_m, scene.grid_step_m
    n = int(round(e / s))
    segs, styles = [], []
    for i in range(-n, n + 1):
        v = i * s
        major = i % scene.major_every == 0
        col = scene.major_bgr if major else scene.line_bgr
        th = (2 if major else 1) + (i == 0)
        segs.append((np.array([v, -e, 0.0]), np.array([v, e, 0.0])))
        styles.append((scene.y_axis_bgr if i == 0 else col, th))
        segs.append((np.array([-e, v, 0.0]), np.array([e, v, 0.0])))
        styles.append((scene.x_axis_bgr if i == 0 else col, th))
    _draw_lines(img, segs, styles, camera_from_world, K, dist)

    light = np.asarray(scene.light_dir, float)
    light /= np.linalg.norm(light)
    cam_centre_w = -R.T @ camera_from_world[:3, 3]
    faces = []
    for bk in scene.blocks:
        x0, y0, x1, y1, z1 = bk.x, bk.y, bk.x + bk.size_x, bk.y + bk.size_y, bk.height
        v = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],
                      [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
        quads = [((0, 1, 5, 4), (0, -1, 0)), ((1, 2, 6, 5), (1, 0, 0)), ((2, 3, 7, 6), (0, 1, 0)),
                 ((3, 0, 4, 7), (-1, 0, 0)), ((4, 5, 6, 7), (0, 0, 1))]
        cam, uv = _project(v, camera_from_world, K, dist)
        for ids, normal in quads:
            ids = list(ids)
            centre = v[ids].mean(0)
            if np.dot(np.asarray(normal), cam_centre_w - centre) <= 0:
                continue                         # back face
            if cam[ids, 2].min() < NEAR_M:
                continue
            poly = uv[ids]
            if not np.isfinite(poly).all() or np.abs(poly).max() > 1e5:
                continue
            shade = 0.45 + 0.55 * max(0.0, float(np.dot(normal, light)))
            faces.append((float(np.linalg.norm(cam[ids].mean(0))), np.rint(poly * 16).astype(np.int32),
                          tuple(int(c * shade) for c in bk.bgr)))
    for _, poly, colour in sorted(faces, key=lambda f: -f[0]):
        cv2.fillConvexPoly(img, poly, colour, cv2.LINE_AA, shift=4)
        cv2.polylines(img, [poly], True, (25, 25, 25), 1, cv2.LINE_AA, shift=4)
    return img


def overlay(video_bgr, background_bgr, opacity=0.5):
    """Registration check view: virtual scene blended over the real frame."""
    return cv2.addWeighted(video_bgr, 1 - opacity, background_bgr, opacity, 0)


def composite(video_bgr, alpha, background_bgr):
    """Actor (original pixels) over background. alpha: float32 HxW in [0, 1]."""
    a = alpha[..., None].astype(np.float32)
    return np.rint(video_bgr.astype(np.float32) * a + background_bgr.astype(np.float32) * (1 - a)).astype(np.uint8)
