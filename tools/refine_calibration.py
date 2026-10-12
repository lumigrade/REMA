#!/usr/bin/env python3
"""Refine saved calibration with explicit fit/evaluation split; no camera or activation."""
import sys, json, argparse, hashlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from rema.calibration import validate_depth
from rema.calibration_refine import check_observation, fit_rigid_edges


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(folder, training, heldout):
    folder = Path(folder).resolve()
    if len(training) < 6 or len(heldout) < 3 or len(set(training+heldout)) != len(training+heldout):
        raise ValueError('Need >=6 training and >=3 evaluation captures, distinct and disjoint')
    config_path = folder/'candidate.json'
    config = json.loads(config_path.read_text())
    if not config.get('stereo_pass'):
        raise ValueError('Original plane fit must pass before edge refinement')
    raw_paths = [folder/f'depth_check_{i:03}.npz' for i in training+heldout]
    pose_paths = sorted(folder.glob('pose_*.npz'))
    hashes = {str(p): digest(p) for p in [config_path]+raw_paths+pose_paths}
    observations, raw, planes = {}, {}, []
    for i, path in zip(training+heldout, raw_paths):
        with np.load(path, allow_pickle=False) as data:
            rgb, depth = data['rgb'].copy(), data['depth'].copy()
            detected = (data['rgb_points'].copy(), data['rgb_ids'].copy()) if 'rgb_points' in data else None
        observation = check_observation(config, rgb, depth, detected)
        observations[i] = observation
        raw[i] = rgb, depth, observation['detected']
        if i in training:
            planes.append(observation['plane'])
    original_test_planes = []
    for i, path in enumerate(pose_paths):
        with np.load(path, allow_pickle=False) as data:
            ok, rotation, origin = cv2.solvePnP(data['obj'], data['rgb'], np.array(config['K_rgb']), np.array(config['dist_rgb']))
            if not ok:
                raise ValueError('Saved fitting board pose failed')
            points = data['plane_points'][::max(1, len(data['plane_points'])//100)].copy()
            plane = (points, cv2.Rodrigues(rotation)[0][:, 2], origin.ravel())
            (original_test_planes if i % 4 == 3 else planes).append(plane)
    refined, metrics = fit_rigid_edges(config, [observations[i] for i in training], planes)
    checks = []
    for i in sorted(raw):
        rgb, depth, detected = raw[i]
        checks.append(dict(index=i, split='training' if i in training else 'evaluation',
                           before=validate_depth(config, rgb, depth, detected=detected),
                           after=validate_depth(refined, rgb, depth, detected=detected)))
    plane_errors = []
    for points, normal, origin in original_test_planes:
        plane_errors.extend(np.abs((points@np.array(refined['R_depth_to_rgb']).T+np.array(refined['t_depth_to_rgb_metres'])-origin)@normal))
    evaluation = [c for c in checks if c['split'] == 'evaluation']
    report = dict(schema='rema_offline_edge_refinement_v1', created_unix=time.time(),
                  source_candidate_sha256=hashes[str(config_path)], source_hashes=hashes,
                  training=training, evaluation=heldout, checks=checks, optimizer=metrics,
                  evaluation_passed=sum(bool(c['after']['passed']) for c in evaluation),
                  evaluation_total=len(evaluation), qualified=False,
                  original_heldout_plane_p50_p95_m=np.percentile(plane_errors, [50, 95]).tolist(),
                  limitations='Evaluation captures were not used by this optimizer; some were previously inspected. This is an offline comparison, not a new six-check qualification. A fresh independent set is required before activation.')
    refined['refinement'] = {k: report[k] for k in ('schema', 'training', 'evaluation', 'source_candidate_sha256', 'limitations')}
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError('Source evidence changed during refinement')
    output = folder/('refinement_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir()
    (output/'candidate.json').write_text(json.dumps(refined, indent=2))
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(output=str(output), evaluation_passed=report['evaluation_passed'],
                         evaluation_total=len(evaluation), plane_p95_mm=report['original_heldout_plane_p50_p95_m'][1]*1000,
                         errors=[dict(index=c['index'], before=c['before']['edge_p95_px'], after=c['after']['edge_p95_px']) for c in evaluation]), indent=2))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session');p.add_argument('--train', type=int, nargs='+', required=True)
    p.add_argument('--evaluate', type=int, nargs='+', required=True)
    args = p.parse_args();cv2.setNumThreads(2)
    run(args.session, args.train, args.evaluate)
