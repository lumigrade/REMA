"""Metric geometry and replay contracts. Optical axes: x right, y down, z forward."""
from dataclasses import dataclass
import json, hashlib
from pathlib import Path
import numpy as np
import cv2
@dataclass(frozen=True)
class Frame:
 stream: str
 frame_id: int
 sensor_timestamp_ns: int | None
 host_arrival_ns: int
 exposure_host_ns: int | None
 clock_domain: str
 calibration_version: str
 model_version: str
 payload: str
 sha256: str

def pair_frames(rgb, depth, max_skew_ns=10_000_000):
 """One-to-one monotonic exposure pairing; arrival time is never a sync substitute."""
 if any(f.exposure_host_ns is None for f in rgb+depth): raise ValueError('exposure-time alignment unqualified')
 for frames in (rgb,depth):
  if any(b.exposure_host_ns <= a.exposure_host_ns or b.frame_id <= a.frame_id for a,b in zip(frames,frames[1:])): raise ValueError('non-monotonic stream')
 out=[]; j=0
 for r in rgb:
  while j+1<len(depth) and abs(depth[j+1].exposure_host_ns-r.exposure_host_ns)<abs(depth[j].exposure_host_ns-r.exposure_host_ns): j+=1
  if j<len(depth) and abs(depth[j].exposure_host_ns-r.exposure_host_ns)<=max_skew_ns:
   out.append((r,depth[j])); j+=1
 return out

def validate_calibration(c):
 if c['units']!='metres' or c['transform_direction']!='depth_to_rgb': raise ValueError('explicit metre units and depth_to_rgb required')
 for name in ['rgb','depth']:
  k=np.asarray(c[name]['K'],float); d=np.asarray(c[name]['distortion'],float)
  if c[name]['distortion_model']!='brown_conrady' or k.shape!=(3,3) or k[0,0]<=0 or k[1,1]<=0 or not np.isfinite(k).all() or d.size!=5 or not np.isfinite(d).all(): raise ValueError('invalid intrinsics/distortion')
 t=np.asarray(c['T_rgb_from_depth'],float)
 if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(t[:3,:3]),1): raise ValueError('invalid rigid transform')
 if not 0<c['depth_scale_metres_per_unit']<=1: raise ValueError('invalid depth scale')
 return c

def project_depth(points_metres,c):
 validate_calibration(c); xyz=np.asarray(points_metres,float); t=np.asarray(c['T_rgb_from_depth'],float)
 q=xyz@t[:3,:3].T+t[:3,3]
 if np.any(q[:,2]<=0): raise ValueError('behind RGB camera')
 uv,_=cv2.projectPoints(q,np.zeros(3),np.zeros(3),np.asarray(c['rgb']['K'],float),np.asarray(c['rgb']['distortion'],float))
 return uv.reshape(-1,2)

def load_replay(path):
 path=Path(path); rows=[Frame(**json.loads(line)) for line in (path/'frames.jsonl').read_text().splitlines()]
 last={}
 for f in rows:
  if f.stream in last and (f.frame_id<=last[f.stream].frame_id or f.host_arrival_ns<=last[f.stream].host_arrival_ns): raise ValueError('replay order invalid')
  last[f.stream]=f
  payload=(path/f.payload).resolve()
  if not payload.is_relative_to(path.resolve()) or hashlib.sha256(payload.read_bytes()).hexdigest()!=f.sha256: raise ValueError('payload containment/integrity failed')
 return rows
