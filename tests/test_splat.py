"""3DGS background: loaders, alignment, and GPU rendering geometry vs OpenCV projection.
Synthetic splats only; GPU tests skip when no EGL/OpenGL 4.3 context is available."""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rema import splat as S

K = np.array([[1173.6, 0, 679.2], [0, 1178.1, 387.1], [0, 0, 1.0]])
DIST = np.array([-0.2992, 0.1275, 0.0003, 0.0004, 0.0])


def scene(points, scales, quats=None):
    n = len(points)
    return {'pos': np.array(points, np.float32), 'scale': np.array(scales, np.float32), 'rgb': np.ones((n, 3), np.float32),
            'alpha': np.ones(n, np.float32), 'quat_wxyz': np.array(quats if quats is not None else [[1, 0, 0, 0]] * n, np.float32)}


def gpu_available():
    try:
        import moderngl
        moderngl.create_standalone_context(require=430, backend='egl').release()
        return True
    except Exception:
        return False


class Loaders(unittest.TestCase):
    def test_splat_and_ply_round_trip(self):
        rng = np.random.default_rng(0)
        n = 50
        pos = rng.normal(size=(n, 3)).astype(np.float32)
        scale = rng.uniform(0.01, 0.2, (n, 3)).astype(np.float32)
        q = rng.normal(size=(n, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        rgb = rng.uniform(0, 1, (n, 3))
        alpha = rng.uniform(0.05, 1, n)
        with tempfile.TemporaryDirectory() as d:
            # .splat
            rec = np.zeros((n, 32), np.uint8)
            rec[:, 0:12] = pos.view(np.uint8).reshape(n, 12)
            rec[:, 12:24] = scale.view(np.uint8).reshape(n, 12)
            rec[:, 24:27] = np.round(rgb * 255)
            rec[:, 27] = np.round(alpha * 255)
            rec[:, 28:32] = np.clip(np.round(q * 128 + 128), 0, 255)
            rec.tofile(Path(d) / 's.splat')
            sp = S.load_scene(Path(d) / 's.splat')
            np.testing.assert_allclose(sp['pos'], pos)
            np.testing.assert_allclose(sp['scale'], scale)
            np.testing.assert_allclose(sp['alpha'], np.round(alpha * 255) / 255)
            # .ply (3DGS conventions: f_dc via SH_C0, logit opacity, log scale)
            names = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
            cols = np.column_stack([pos, (rgb - 0.5) / S.SH_C0, np.log(alpha / (1 - alpha + 1e-9)), np.log(scale), q]).astype(np.float32)
            header = 'ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n + ''.join('property float %s\n' % c for c in names) + 'end_header\n'
            with open(Path(d) / 's.ply', 'wb') as f:
                f.write(header.encode())
                f.write(cols.tobytes())
            pl = S.load_scene(Path(d) / 's.ply')
            np.testing.assert_allclose(pl['pos'], pos, atol=1e-6)
            np.testing.assert_allclose(pl['scale'], scale, rtol=1e-5)
            np.testing.assert_allclose(pl['rgb'], np.clip(rgb, 0, 1), atol=1e-5)
            np.testing.assert_allclose(pl['alpha'], alpha, atol=1e-4)

    def test_alignment_matrix_and_world_covariance(self):
        al = {'scale': 2.0, 'rotation_deg': [-90, 0, 0], 'translation': [1, 2, 3]}
        M = S.alignment_matrix(al)
        np.testing.assert_allclose(M[:3, :3] @ np.array([0, -1.0, 0]), [0, 0, 2.0], atol=1e-12)   # scene -Y up -> world +Z
        self.assertTrue(np.allclose(S.alignment_matrix({'matrix': M.tolist()}), M))
        pos, cov6, rgba = S.world_gaussians(scene([[0, -1, 0]], [[0.1, 0.2, 0.3]]), al)
        np.testing.assert_allclose(pos[0], [1, 2, 5], atol=1e-6)
        # scale 2: variances scale by 4; the scene Y axis (sigma 0.2) becomes world Z
        np.testing.assert_allclose([cov6[0, 0], cov6[0, 3], cov6[0, 5]], [0.04, 0.36, 0.16], atol=1e-6)

    def test_stage_clearing(self):
        sc = scene([[0, 0, 0.5], [3, 0, 0.5], [0, 0, 5.0]], [[0.1] * 3] * 3)
        pos, _, _ = S.world_gaussians(sc, {'clear_cylinders': [{'center': [0, 0], 'radius': 1.0, 'z_max': 4.0}]})
        self.assertEqual(sorted(map(tuple, pos.round(3))), [(0.0, 0.0, 5.0), (3.0, 0.0, 0.5)])   # only the low splat in the stage is removed


@unittest.skipUnless(gpu_available(), 'no EGL / OpenGL 4.3 context')
class Render(unittest.TestCase):
    def centroid_errors(self, dist):
        pts = [[0.3, -0.2, 3.0], [-0.6, 0.25, 2.5], [0.9, 0.5, 4.0]]
        r = S.SplatRenderer(scene(pts, [[0.01] * 3] * 3), {}, K, dist, (1360, 768), background_bgr=(0, 0, 0))
        img = r.render(np.eye(4)).astype(np.float32).sum(2)
        r.close()
        exp, _ = cv2.projectPoints(np.array(pts), np.zeros(3), np.zeros(3), K, dist)
        errs = []
        for u, v in exp.reshape(-1, 2):
            x0, y0 = int(u) - 15, int(v) - 15
            patch = img[y0:y0 + 31, x0:x0 + 31]
            ys, xs = np.mgrid[0:31, 0:31]
            errs.append(np.hypot((patch * xs).sum() / patch.sum() + x0 - u + 0.5, (patch * ys).sum() / patch.sum() + y0 - v + 0.5))
        return errs

    def test_projection_matches_opencv(self):
        self.assertLess(max(self.centroid_errors(np.zeros(5))), 0.1)
        self.assertLess(max(self.centroid_errors(DIST)), 0.1)

    def test_orientation_and_behind_camera(self):
        r = S.SplatRenderer(scene([[0, 0, 3.0], [0, 0, -3.0]], [[0.2, 0.01, 0.01]] * 2), {}, K, None, (1360, 768), background_bgr=(0, 0, 0))
        ys, xs = np.nonzero(r.render(np.eye(4)).sum(2) > 20)
        r.close()
        self.assertGreater(np.ptp(xs), 10 * np.ptp(ys))          # X-elongated -> wide; the splat behind is culled
        q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
        r = S.SplatRenderer(scene([[0, 0, 3.0]], [[0.2, 0.01, 0.01]], [q]), {}, K, None, (1360, 768), background_bgr=(0, 0, 0))
        ys, xs = np.nonzero(r.render(np.eye(4)).sum(2) > 20)
        r.close()
        self.assertGreater(np.ptp(ys), 10 * np.ptp(xs))

    def test_back_to_front_occlusion(self):
        red_near = scene([[0, 0, 2.0]], [[0.1] * 3])
        red_near['rgb'][:] = [1, 0, 0]
        both = {k: np.concatenate([red_near[k], scene([[0, 0, 4.0]], [[0.2] * 3])[k]]) for k in red_near}
        r = S.SplatRenderer(both, {}, K, None, (1360, 768), background_bgr=(0, 0, 0))
        px = r.render(np.eye(4))[387, 679].astype(int)
        r.close()
        self.assertGreater(px[2], 200)                            # near red splat in front (BGR)
        self.assertLess(px[1], 60)


if __name__ == '__main__':
    unittest.main()
