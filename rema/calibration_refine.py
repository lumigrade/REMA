"""Offline rigid RGB/depth refinement. Never activates a calibration.

Fit full native depth contours against RGB paper edges, together with board
planes. Intrinsics, depth scale and qualification thresholds remain fixed.
"""
import copy
import cv2
import numpy as np
from .calibration import Projection, board, detect_robust


def check_observation(config, rgb, depth, detected=None):
    projection = Projection(config)
    corners, ids = detect_robust(rgb, config['square_metres']) if detected is None else detected
    if len(ids) < 12:
        raise ValueError('Need at least 12 RGB corners for refinement')
    ok, rotation, origin = cv2.solvePnP(board(config['square_metres']).getChessboardCorners()[ids],
                                      corners, projection.K, projection.dist)
    if not ok:
        raise ValueError('Board pose solve failed')
    normal = cv2.Rodrigues(rotation)[0][:, 2]
    depth = np.asarray(depth)
    z = depth.ravel()
    xyz = projection.rays * z[:, None]
    transformed = xyz @ projection.R.T + projection.t
    near = np.isfinite(z) & (z > .25) & (z < 10) & (np.abs((transformed-origin.ravel()) @ normal) < .025)
    mask = cv2.morphologyEx(near.reshape(480, 640).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels = cv2.connectedComponents(mask)
    # Select by overlap with the board interior; never crop its measured outline.
    hull = np.zeros((480, 640), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(corners.astype(np.float32)).astype(np.int32), 1)
    hull = cv2.erode(hull, np.ones((11, 11), np.uint8))
    uv, _ = cv2.projectPoints(transformed[near], np.zeros(3), np.zeros(3), projection.K, projection.dist)
    px = np.rint(uv.reshape(-1, 2)).astype(int)
    inside = (px[:, 0] >= 0) & (px[:, 0] < 640) & (px[:, 1] >= 0) & (px[:, 1] < 480)
    locations = np.flatnonzero(near)[inside]
    px = px[inside]
    locations = locations[hull[px[:, 1], px[:, 0]] > 0]
    overlap = np.bincount(labels.ravel()[locations], minlength=count)
    overlap[0] = 0
    if not overlap.any():
        raise ValueError('No separated depth board surface')
    mask = np.uint8(labels == overlap.argmax())
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError('Board surface connects to image boundary; cannot refine')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pixels = max(contours, key=cv2.contourArea)[:, 0]
    valid = near.reshape(480, 640)[pixels[:, 1], pixels[:, 0]]
    pixels = pixels[valid]
    pixels = pixels[::max(1, len(pixels)//400)]
    edge_points = xyz[pixels[:, 1]*640+pixels[:, 0]]
    if len(edge_points) < 100:
        raise ValueError('Too few measured contour samples')
    inner = cv2.erode(mask, np.ones((15, 15), np.uint8)).ravel().astype(bool) & near
    points = xyz[inner][::max(1, np.count_nonzero(inner)//100)]
    square = config['square_metres']
    ox, oy = (.297-10*square)/2, (.210-7*square)/2
    sheet = np.float32([[-ox, -oy, 0], [.297-ox, -oy, 0], [.297-ox, .210-oy, 0], [-ox, .210-oy, 0]])
    polygon, _ = cv2.projectPoints(sheet, rotation, origin, projection.K, projection.dist)
    return dict(edge_points=edge_points, polygon=polygon.reshape(4, 2),
                plane=(points, normal, origin.ravel()), detected=(corners, ids))


def fit_rigid_edges(config, observations, planes):
    """Only training observations belong here; evaluation happens in the caller."""
    if len(observations) < 3 or not planes:
        raise ValueError('Need multiple separated board observations and training planes')
    projection = Projection(config)
    start = np.r_[cv2.Rodrigues(projection.R)[0].ravel(), projection.t]

    def residual(parameters):
        rotation = cv2.Rodrigues(parameters[:3])[0]
        translation = parameters[3:]
        errors = []
        for observation in observations:
            points = observation['edge_points'] @ rotation.T + translation
            uv, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), projection.K, projection.dist)
            uv = uv.reshape(-1, 2)
            polygon = observation['polygon']
            distances = []
            for j in range(4):
                origin = polygon[j]
                vector = polygon[(j+1) % 4]-origin
                factor = np.clip(((uv-origin)*vector).sum(axis=1)/(vector@vector), 0, 1)
                distances.append(np.linalg.norm(uv-origin-factor[:, None]*vector, axis=1))
            errors.extend(np.min(distances, axis=0))
        for points, normal, origin in planes:
            errors.extend(((points@rotation.T+translation-origin)@normal)/.004)
        return np.asarray(errors)

    def cost(parameters):
        return float(np.sum(2*(np.sqrt(1+residual(parameters)**2)-1)))

    parameters = start.copy()
    for iteration in range(60):
        error = residual(parameters)
        if not np.isfinite(error).all():
            raise ValueError('Nonfinite refinement residual')
        jacobian = np.column_stack([(residual(parameters+np.eye(6)[j]*1e-6)-error)/1e-6 for j in range(6)])
        weights = 1/np.sqrt(1+error**2)
        step = np.linalg.lstsq(jacobian*np.sqrt(weights[:, None]), -error*np.sqrt(weights), rcond=None)[0]
        previous = cost(parameters)
        for factor in [1, .5, .25, .125, .0625, .03125]:
            trial = parameters+step*factor
            if np.isfinite(trial).all() and cost(trial) < previous:
                parameters = trial
                break
        else:
            break
        if np.linalg.norm(step*factor) < 1e-7:
            break
    if np.linalg.norm(parameters[3:]) >= .3 or np.linalg.norm(parameters[:3]) >= .4:
        raise ValueError('Implausible refined rigid transform')
    result = copy.deepcopy(config)
    result.update(R_depth_to_rgb=cv2.Rodrigues(parameters[:3])[0].tolist(),
                  t_depth_to_rgb_metres=parameters[3:].tolist(), qualified=False,
                  depth_validation_pass=False, exposure_sync_qualified=False)
    for key in ('depth_checks', 'timing_proxy', 'calibration_id'):
        result.pop(key, None)
    metrics = dict(before_cost=cost(start), after_cost=cost(parameters), iterations=iteration+1,
                   rigid_parameter_delta=(parameters-start).tolist())
    return result, metrics
