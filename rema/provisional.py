"""Explicitly unqualified calibration for user-authorized prototype tracking.
Never writes active.json and never grants measured-calibration qualification.
"""
import json,time
from pathlib import Path
import cv2,numpy as np
from .calibration import ROOT, Projection
from .calibration_overlay import estimate_guide_K

def create_provisional():
    samples=[];source=[];intrinsics=None
    for folder in sorted(ROOT.glob('session_*'),reverse=True):
        for path in sorted(folder.glob('pose_*.npz')):
            with np.load(path,allow_pickle=False) as data:
                required=('obj','rgb','square_metres','plane_points','depth_intrinsics')
                if not all(k in data for k in required):continue
                sample={k:data[k].copy() for k in required}
                if not np.isfinite(sample['plane_points']).all():continue
                samples.append(sample);source.append(str(path));intrinsics=sample['depth_intrinsics']
        if samples:break
    if intrinsics is None:
        intrinsics=np.array([570.3421630859375,570.3422241210938,320.,240.])
    fx,fy,cx,cy=map(float,intrinsics)
    config=dict(schema='rema_provisional_rgbd_v1',provisional=True,user_authorized=True,qualified=False,
         stereo_pass=False,depth_validation_pass=False,exposure_sync_qualified=False,created_unix=time.time(),
         calibration_id='provisional_sdk_rays_v1',image_size=[640,480],
         K_depth=[[fx,0,cx],[0,fy,cy],[0,0,1]],K_rgb=estimate_guide_K(samples).tolist(),
         dist_depth=[0]*5,dist_rgb=[0]*5,R_depth_to_rgb=np.eye(3).tolist(),t_depth_to_rgb_metres=[0,0,0],
         square_metres=float(samples[0]['square_metres']) if samples else .02317,
         source_frames=source,assumptions=[
             'Depth intrinsics from OpenNI CoordinateConverter; extra distortion assumed zero.',
             'RGB focal lengths are only a rough estimate from the existing print, whose size/quality is unreliable.',
             'Principal point assumed at image centre; RGB distortion assumed zero.',
             'Identity depth-to-RGB rotation and zero translation are temporary priors, NOT measured extrinsics.',
             'Relative tracking scale, RGB/depth edges, floor height and physical latency are unqualified.'])
    path=ROOT/'provisional.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(config,indent=2));tmp.replace(path)
    return config

def load_provisional():
    p=ROOT/'provisional.json'
    if not p.exists():return None
    c=json.loads(p.read_text())
    if c.get('schema')!='rema_provisional_rgbd_v1' or not c.get('user_authorized') or not c.get('provisional') or c.get('qualified'):
        raise ValueError('Invalid provisional calibration contract')
    for key in ('K_rgb','K_depth'):
        K=np.asarray(c[key],float)
        if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0:raise ValueError('Invalid provisional intrinsics')
    if c.get('alignment_model') == 'rigid_reprojection_v1':
        R=np.asarray(c['R_depth_to_rgb'],float);t=np.asarray(c['t_depth_to_rgb_metres'],float)
        if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-5) or np.linalg.det(R)<.99 or np.linalg.norm(t)>=.3:
            raise ValueError('Invalid provisional rigid transform')
    return c

def align_provisional(depth,config,projection=None):
    # This homography is valid ONLY for the explicit zero-baseline/zero-distortion
    # prior. It is fast and avoids sparse reprojection holes during prototyping.
    if not config.get('provisional'):raise ValueError('Use metric reprojection for measured calibration')
    if config.get('alignment_model')=='rigid_reprojection_v1':
        return fill_small_depth_holes((projection or Projection(config)).align(depth))
    if not np.allclose(config.get('R_depth_to_rgb',np.eye(3)),np.eye(3)) or not np.allclose(config.get('t_depth_to_rgb_metres',[0,0,0]),0) or not np.allclose(config.get('dist_rgb',[0]*5),0) or not np.allclose(config.get('dist_depth',[0]*5),0):
        raise ValueError('Nonidentity provisional geometry requires rigid reprojection')
    H=np.array(config['K_rgb'])@np.linalg.inv(np.array(config['K_depth']))
    return cv2.warpPerspective(depth,H,(640,480),flags=cv2.INTER_NEAREST)


def fill_small_depth_holes(depth):
    """Current-frame interpolation for isolated holes on locally flat surfaces.

    Does not bridge a depth discontinuity, extrapolate a silhouette, or overwrite
    measured depth. Used only for the unqualified live prototype, never checks.
    """
    valid=np.isfinite(depth)&(depth>.25)
    safe=np.where(valid,depth,0).astype(np.float32)
    kernel=np.ones((3,3),np.uint8)
    count=cv2.filter2D(valid.astype(np.float32),-1,kernel,borderType=cv2.BORDER_CONSTANT)
    low=cv2.erode(np.where(valid,safe,np.inf),kernel)
    high=cv2.dilate(safe,kernel)
    supported=(~valid)&(count>=5)&((high-low)<=.025)
    median=cv2.medianBlur(safe,3)
    result=safe.copy();result[supported]=median[supported]
    return result
