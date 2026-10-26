"""Nodal (Alt -> camera) calibration: hand-eye with a board fixed in the room.

Per still sample i:
  A_i = world <- tracker   (Antilatency raw pose in the right-handed S frame, rema.pose.tracker_rh)
  B_i = camera <- board    (board PnP with the calibrated lens)
Model: B_i = X @ inv(A_i) @ Z,  X = camera <- tracker,  Z = world <- board (both constant).

Pipeline: diversity check -> OpenCV calibrateRobotWorldHandEye (Shah and Li) -> pick lower
reprojection -> Gauss-Newton refinement of X, Z on all corner reprojections -> outlier removal ->
held-out evaluation (every 4th sample is never used for fitting).
Output T_tracker_cam = inv(X) uses the same convention as VPcamtracking's nodal result.
"""
import math

import cv2
import numpy as np

from rema.pose import inv, rt

ROTATION_FIRST_MAX_STD_DEG = 0.1   # rule fixed in advance (2026-10-24), not tuned on held-out data


# ---------------------------------------------------------------------------- helpers

def rotation_angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def to_vec(T):
    rv, _ = cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))
    return np.r_[rv.ravel(), T[:3, 3]]


def from_vec(v):
    R, _ = cv2.Rodrigues(np.asarray(v[:3], float))
    return rt(R, v[3:6])


def rotation_diversity(tracker_rotations, min_angle_deg=3.0):
    """How well the sample rotations constrain hand-eye.
    Returns dict(max_relative_deg, axis_spread, n_pairs, ok, advice). axis_spread is the second
    singular value / first of the angle-weighted relative rotation axes (0 = single axis, e.g. pan only)."""
    axes, max_angle = [], 0.0
    Rs = list(tracker_rotations)
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            Rij = Rs[i].T @ Rs[j]
            ang = rotation_angle_deg(Rij)
            max_angle = max(max_angle, ang)
            if ang >= min_angle_deg:
                rv, _ = cv2.Rodrigues(Rij)
                n = rv.ravel() / np.linalg.norm(rv)
                axes.append(n * min(ang, 45.0))
    if len(axes) < 2:
        return {'max_relative_deg': max_angle, 'axis_spread': 0.0, 'n_pairs': len(axes), 'ok': False,
                'advice': 'Rotate the camera more between positions (pan, tilt and some roll).'}
    A = np.array(axes)
    A = np.vstack([A, -A])                      # axis sign is arbitrary
    s = np.linalg.svd(A, compute_uv=False)
    spread = float(s[1] / s[0]) if s[0] > 0 else 0.0
    ok = max_angle >= 20.0 and spread >= 0.3
    advice = 'Rotation variety OK.' if ok else (
        'Rotations are mostly around ONE axis. Add tilt up/down and roll the camera, not only pan.' if spread < 0.3
        else 'Rotate further: largest change between positions is only %.0f deg (want >= 20).' % max_angle)
    return {'max_relative_deg': max_angle, 'axis_spread': spread, 'n_pairs': len(axes), 'ok': ok, 'advice': advice}


def board_pose(obj, img, K, dist, max_rms_px=1.5, min_ambiguity_ratio=1.6, noise_px=0.3):
    """Planar PnP with IPPE. Rejects poses whose two IPPE solutions explain the corners similarly
    well (mirror-tilt ambiguity: frequent for small, far, frontal boards). Returns (T_cam_board, info) or (None, info)."""
    obj = np.asarray(obj, np.float64).reshape(-1, 3)
    img = np.asarray(img, np.float64).reshape(-1, 2)
    if len(obj) < 8:
        return None, {'reason': 'too_few_corners', 'n': len(obj)}
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if n == 0:
        return None, {'reason': 'pnp_failed'}
    errs = np.ravel(errs)
    order = np.argsort(errs)
    rv, tv = rvecs[order[0]], tvecs[order[0]]
    rv, tv = cv2.solvePnPRefineLM(obj, img, K, dist, rv.copy(), tv.copy())
    proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, 1))))
    ratio = float(errs[order[1]] / max(errs[order[0]], 1e-9)) if n > 1 else float('inf')
    info = {'rms_px': rms, 'ambiguity_ratio': ratio, 'n': len(obj)}
    if rms > max_rms_px:
        info['reason'] = 'board_reprojection_high'
        return None, info
    # An ambiguous (mirror-tilt) pose is still a valid observation: refinement uses the corner
    # pixels, not this pose. It is only excluded from seeding the hand-eye solve.
    info['ambiguous'] = ratio < min_ambiguity_ratio
    R, _ = cv2.Rodrigues(rv)
    # Orientation precision of this board pose: re-solve with simulated corner noise.
    rng = np.random.default_rng(0)
    angs = []
    for _ in range(12):
        ok, rv2, tv2 = cv2.solvePnP(obj, img + rng.normal(0, noise_px, img.shape), K, dist, rv.copy(), tv.copy(),
                                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            angs.append(rotation_angle_deg(R.T @ cv2.Rodrigues(rv2)[0]))
    info['rot_std_deg'] = float(np.sqrt(np.mean(np.square(angs)))) if angs else float('inf')
    info['reason'] = 'ok'
    return rt(R, tv.ravel()), info


# ---------------------------------------------------------------------------- solve

def _project_all(X, Z, samples, K, dist, idx):
    out = []
    for i in idx:
        s = samples[i]
        T = X @ inv(s['A']) @ Z
        rv, _ = cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))
        p, _ = cv2.projectPoints(s['obj'], rv, T[:3, 3], K, dist)
        out.append((p.reshape(-1, 2) - s['img']).ravel())
    return out


def sample_rms(X, Z, samples, K, dist, idx):
    """Per-sample RMS corner reprojection error (px), predicting the board from tracking only."""
    return [float(np.sqrt(np.mean(np.sum(r.reshape(-1, 2) ** 2, 1)))) for r in _project_all(X, Z, samples, K, dist, idx)]


def _opencv_init(samples, idx, method):
    R_w2c, t_w2c, R_b2g, t_b2g = [], [], [], []
    for i in idx:
        s = samples[i]
        B, Ai = s['B'], inv(s['A'])
        R_w2c.append(B[:3, :3]); t_w2c.append(B[:3, 3].reshape(3, 1))
        R_b2g.append(Ai[:3, :3]); t_b2g.append(Ai[:3, 3].reshape(3, 1))
    R_bw, t_bw, R_gc, t_gc = cv2.calibrateRobotWorldHandEye(R_w2c, t_w2c, R_b2g, t_b2g, method=method)
    X = rt(R_gc, t_gc)          # camera <- tracker
    Zi = rt(R_bw, t_bw)         # board <- world
    for T in (X, Zi):
        if abs(np.linalg.det(T[:3, :3]) - 1) > 1e-3:
            raise cv2.error('non-rotation result')
    return X, inv(Zi)


def _gauss_newton(p, resid, iters):
    r = resid(p)
    lam = 1e-3
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for k in range(p.size):
            d = np.zeros(p.size)
            d[k] = 1e-6
            J[:, k] = (resid(p + d) - r) / 1e-6
        H = J.T @ J
        g = J.T @ r
        step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
        r2 = resid(p + step)
        if r2 @ r2 < r @ r:
            p, r, lam = p + step, r2, lam * 0.3
            if np.abs(step).max() < 1e-10:
                break
        else:
            lam *= 10
            if lam > 1e8:
                break
    return p


def _refine_3d(X, Z, samples, idx, iters=40):
    """Fit X, Z so predicted board corners match each PnP board pose in camera space (metres).
    Unlike pixel reprojection this has no 'push the board to infinity' collapse, so it turns a
    rough linear seed into a reliable start for the pixel refinement."""
    def resid(q):
        Xq, Zq = from_vec(q[:6]), from_vec(q[6:])
        out = []
        for i in idx:
            s = samples[i]
            P = Xq @ inv(s['A']) @ Zq
            out.append(((s['obj'] @ P[:3, :3].T + P[:3, 3]) - (s['obj'] @ s['B'][:3, :3].T + s['B'][:3, 3])).ravel())
        return np.concatenate(out)
    p = _gauss_newton(np.r_[to_vec(X), to_vec(Z)], resid, iters)
    return from_vec(p[:6]), from_vec(p[6:])


def _refine(X, Z, samples, K, dist, idx, iters=30):
    """Gauss-Newton on 12 params (X, Z as rotation vector + translation), numeric Jacobian."""
    p = np.r_[to_vec(X), to_vec(Z)]

    def resid(q):
        return np.concatenate(_project_all(from_vec(q[:6]), from_vec(q[6:]), samples, K, dist, idx))
    r = resid(p)
    lam = 1e-3
    for _ in range(iters):
        J = np.empty((r.size, 12))
        for k in range(12):
            d = np.zeros(12)
            d[k] = 1e-6
            J[:, k] = (resid(p + d) - r) / 1e-6
        H = J.T @ J
        g = J.T @ r
        step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
        r2 = resid(p + step)
        if r2 @ r2 < r @ r:
            p, r, lam = p + step, r2, lam * 0.3
            if np.abs(step).max() < 1e-9:
                break
        else:
            lam *= 10
            if lam > 1e8:
                break
    return from_vec(p[:6]), from_vec(p[6:])


def _chordal_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def solve_rotation_first(samples, idx):
    """Hand-eye from relative motions, rotation first (Park & Martin style), then translations.
    Pixel reprojection with a close board weights millimetre tracker-position noise above rotation;
    for a distant virtual background rotation dominates, so it is solved from rotations alone:
      C_ij X = X T_ij with C_ij = B_i B_j^-1 (camera motion), T_ij = A_i^-1 A_j (tracker motion).
    Returns X (camera <- tracker), Z (world <- board)."""
    a_c, a_t, w = [], [], []
    for ii, i in enumerate(idx):
        for j in idx[ii + 1:]:
            C = samples[i]['B'] @ inv(samples[j]['B'])
            T = inv(samples[i]['A']) @ samples[j]['A']
            rc, _ = cv2.Rodrigues(np.ascontiguousarray(C[:3, :3]))
            rt_, _ = cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))
            ang = np.linalg.norm(rt_)
            if math.degrees(ang) < 3.0:
                continue                                     # axis undefined for tiny rotations
            a_c.append(rc.ravel())
            a_t.append(rt_.ravel())
            w.append(1.0)
    if len(a_c) < 3:
        raise ValueError('Too few rotation pairs for a rotation-first solve.')
    a_c, a_t = np.array(a_c), np.array(a_t)
    H = (a_t * np.array(w)[:, None]).T @ a_c                 # Kabsch: a_c ~ R_X a_t
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R_X = Vt.T @ D @ U.T

    def rot_resid(rv):
        R = cv2.Rodrigues(rv)[0]
        out = []
        for ii, i in enumerate(idx):
            for j in idx[ii + 1:]:
                C = samples[i]['B'] @ inv(samples[j]['B'])
                T = inv(samples[i]['A']) @ samples[j]['A']
                E = C[:3, :3] @ R @ T[:3, :3].T @ R.T
                out.append(cv2.Rodrigues(np.ascontiguousarray(E))[0].ravel())
        return np.concatenate(out)
    rv = _gauss_newton(cv2.Rodrigues(R_X)[0].ravel(), rot_resid, 30)
    R_X = cv2.Rodrigues(rv)[0]
    # Board orientation in the world, then translations by linear least squares.
    R_Z = _chordal_mean(np.array([samples[i]['A'][:3, :3] @ R_X.T @ samples[i]['B'][:3, :3] for i in idx]))
    rows, rhs = [], []
    for i in idx:
        A, B = samples[i]['A'], samples[i]['B']
        M = R_X @ A[:3, :3].T                              # t_B = M t_Z + t_X - M t_A
        rows.append(np.hstack([np.eye(3), M]))
        rhs.append(B[:3, 3] + M @ A[:3, 3])
    sol, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    return rt(R_X, sol[:3]), rt(R_Z, sol[3:])


def metric_errors(X, Z, samples, idx):
    """Predicted-from-tracking vs PnP board pose per sample: (translation mm, rotation deg)."""
    out = []
    for i in idx:
        Pr = X @ inv(samples[i]['A']) @ Z
        B = samples[i]['B']
        out.append((float(np.linalg.norm(Pr[:3, 3] - B[:3, 3]) * 1000), rotation_angle_deg(Pr[:3, :3].T @ B[:3, :3])))
    return out


def solve_nodal(samples, K, dist, holdout_every=4, outlier_factor=3.0, min_fit=6):
    """samples: list of dict(A=4x4 world<-tracker S frame, B=4x4 camera<-board, obj=Nx3, img=Nx2).
    Returns a result dict; raises ValueError with a readable reason when the data cannot support a solve."""
    K, dist = np.asarray(K, float), np.asarray(dist, float)
    for s in samples:
        s['obj'] = np.asarray(s['obj'], np.float64).reshape(-1, 3)
        s['img'] = np.asarray(s['img'], np.float64).reshape(-1, 2)
    all_idx = list(range(len(samples)))
    held = [i for i in all_idx if holdout_every and i % holdout_every == holdout_every - 1]
    fit = [i for i in all_idx if i not in held]
    if len(fit) < min_fit:
        raise ValueError(f'Need at least {min_fit} fitting samples (+ every {holdout_every}th held out); have {len(fit)}.')
    div = rotation_diversity([samples[i]['A'][:3, :3] for i in fit])
    if not div['ok']:
        raise ValueError('Insufficient rotation variety: ' + div['advice'] +
                         f" (max {div['max_relative_deg']:.1f} deg, axis spread {div['axis_spread']:.2f})")
    # Seed only from unambiguous board poses (fall back to all if too few).
    seed = [i for i in fit if not samples[i].get('ambiguous')]
    if len(seed) < min_fit:
        seed = list(fit)

    def try_seed(subset, iters):
        found = None
        for name, method in (('shah', cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH), ('li', cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)):
            try:
                X0, Z0 = _opencv_init(samples, subset, method)
            except cv2.error:
                continue              # e.g. the 'determinant(R) is null' failure seen in the UI
            X0, Z0 = _refine_3d(X0, Z0, samples, subset)
            X, Z = _refine(X0, Z0, samples, K, dist, fit, iters=iters)
            err = float(np.median(sample_rms(X, Z, samples, K, dist, fit)))   # median: robust to one bad sample
            if found is None or err < found[0]:
                found = (err, name, X, Z)
        return found

    best = try_seed(seed, 30)
    if best is None or best[0] > 2.0:
        # A single inconsistent sample can wreck the linear seed: retry leaving each one out.
        for k in range(len(seed)):
            cand = try_seed(seed[:k] + seed[k + 1:], 8)
            if cand and (best is None or cand[0] < best[0]):
                best = cand
        if best is not None:
            best = (best[0], best[1], *_refine(best[2], best[3], samples, K, dist, fit))
    if best is None:
        raise ValueError('Both hand-eye initialisations failed; check that board and tracker poses are consistent.')
    _, method, X, Z = best
    dropped = []
    while True:
        errs = sample_rms(X, Z, samples, K, dist, fit)
        med = float(np.median(errs))
        worst = int(np.argmax(errs))
        if errs[worst] > max(outlier_factor * med, 2.0) and len(fit) - 1 >= min_fit:
            dropped.append({'sample': fit[worst], 'rms_px': errs[worst]})
            fit = [i for k, i in enumerate(fit) if k != worst]
            X, Z = _refine(X, Z, samples, K, dist, fit)
        else:
            break
    # Rotation-first alternative. Chosen BEFORE looking at held-out results, by board-orientation
    # precision: when board poses are precise (close board), rotation pairs pin R_X far better than
    # pixel reprojection, which then mostly absorbs millimetre tracker-position noise.
    alternatives = {'pixel': (X, Z)}
    rot_std = [samples[i].get('rot_std_deg', float('inf')) for i in fit if not samples[i].get('ambiguous')]
    precise = len(rot_std) >= min_fit and float(np.median(rot_std)) < ROTATION_FIRST_MAX_STD_DEG
    unamb = [i for i in fit if not samples[i].get('ambiguous')]
    if len(unamb) >= min_fit:
        try:
            alternatives['rotation_first'] = solve_rotation_first(samples, unamb)
        except ValueError:
            pass
    chosen = 'rotation_first' if precise and 'rotation_first' in alternatives else 'pixel'
    X, Z = alternatives[chosen]
    report = {}
    for name, (Xa, Za) in alternatives.items():
        ef, eh = metric_errors(Xa, Za, samples, fit), metric_errors(Xa, Za, samples, held)
        report[name] = {'fit_mm_deg_median': [float(np.median([e[0] for e in ef])), float(np.median([e[1] for e in ef]))],
                        'heldout_mm_deg_median': [float(np.median([e[0] for e in eh])), float(np.median([e[1] for e in eh]))] if eh else None,
                        'heldout_deg_max': float(max(e[1] for e in eh)) if eh else None,
                        'heldout_px_median': float(np.median(sample_rms(Xa, Za, samples, K, dist, held))) if held else None,
                        'offset_m': float(np.linalg.norm(inv(Xa)[:3, 3]))}
    fit_errs = sample_rms(X, Z, samples, K, dist, fit)
    held_errs = sample_rms(X, Z, samples, K, dist, held) if held else []
    T_tracker_cam = inv(X)
    return {
        'method': chosen, 'board_rot_std_deg_median': float(np.median(rot_std)) if rot_std else None,
        'methods': report,
        'method_init': method, 'samples_total': len(samples), 'fit_samples': fit, 'heldout_samples': held,
        'dropped_outliers': dropped, 'diversity': div,
        'fit_rms_px_median_max': [float(np.median(fit_errs)), float(np.max(fit_errs))],
        'heldout_rms_px_median_max': [float(np.median(held_errs)), float(np.max(held_errs))] if held_errs else None,
        'per_sample_rms_px': dict(zip([*fit, *held], [*fit_errs, *held_errs])),
        'T_tracker_cam': T_tracker_cam.tolist(), 'T_world_board': Z.tolist(),
        'offset_m': float(np.linalg.norm(T_tracker_cam[:3, 3])),
        'camera_axis_vs_tracker_deg': rotation_angle_deg(T_tracker_cam[:3, :3]),
    }
