"""CPU temporal alpha + conservative, uncalibrated depth-assisted actor selection.
No generated foreground colour: only alpha is used; composite keeps source RGB.
"""
from rema import paths as _P
import time,os
import cv2
import numpy as np
import onnxruntime as ort
MODEL=str(_P.DATA)+'/models/rvm/rvm_mobilenetv3_fp32.onnx'
VERSION='rvm_mobilenetv3_fp32_v1.0.0+conservative_actor_v2'
class TemporalMatte:
 def __init__(self,ratio=.375):
  options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
  options.add_session_config_entry('session.intra_op.allow_spinning','0')
  self.backend=os.environ.get('REMA_MATTE_BACKEND','cpu')
  providers=['CPUExecutionProvider']
  if self.backend=='cuda':
   ort.preload_dlls(directory='')
   providers=[('CUDAExecutionProvider',{'device_id':0,'gpu_mem_limit':1024**3,'arena_extend_strategy':'kSameAsRequested','cudnn_conv_algo_search':'HEURISTIC','cudnn_conv_use_max_workspace':'0','use_tf32':'0'}),'CPUExecutionProvider']
  self.session=ort.InferenceSession(MODEL,options,providers=providers)
  if self.backend=='cuda' and self.session.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('CUDA matting requested but not active')
  self.session.disable_fallback()
  self.ratio=ratio;self.reset()
 def reset(self):self.rec=[np.zeros((1,1,1,1),np.float32) for _ in range(4)];self.last_time=None
 def infer(self,bgr,timestamp=None):
  now=time.monotonic() if timestamp is None else timestamp
  if self.last_time is not None and (now-self.last_time>.5 or now<=self.last_time):self.reset()
  self.last_time=now
  src=np.ascontiguousarray(bgr[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
  alpha,*self.rec=self.session.run(['pha','r1o','r2o','r3o','r4o'],dict(src=src,r1i=self.rec[0],r2i=self.rec[1],r3i=self.rec[2],r4i=self.rec[3],downsample_ratio=np.array([self.ratio],np.float32)))
  return alpha[0,0]
def components(mask):return cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
class ActorMask:
 def __init__(self):self.previous_depth=None;self.last_valid=None
 def reset(self):self.previous_depth=None;self.last_valid=None
 def apply(self,alpha,depth,max_range=4.,tolerance=.9,timestamp=None,background_match=None):
  now=time.monotonic() if timestamp is None else timestamp
  if self.last_valid is not None and now-self.last_valid>.3:self.reset()
  # Find substantial high-confidence body cores. A detached nearer hand is not a new actor.
  cores=(alpha>.65).astype(np.uint8)
  cores=cv2.morphologyEx(cores,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
  n,labels,stats,_=components(cores)
  largest=max(stats[1:,cv2.CC_STAT_AREA],default=0)
  minimum=max(500,largest*.18);candidates=[]
  for i in range(1,n):
   area=stats[i,cv2.CC_STAT_AREA]
   if area<minimum:continue
   inside=cv2.erode((labels==i).astype(np.uint8),np.ones((7,7),np.uint8))>0
   d=depth[inside&(depth>.35)&(depth<max_range)]
   if d.size<80:continue
   z=float(np.percentile(d,40));candidates.append((z,i))
  if candidates:
   z,chosen=min(candidates)
   # Modest depth hysteresis prevents noise swapping similarly placed separated actors.
   if self.previous_depth is not None:
    prior=min(candidates,key=lambda item:abs(item[0]-self.previous_depth))
    if prior[0]-z<.20:z,chosen=prior
   self.previous_depth=z;self.last_valid=now
  else:
   # Brief depth holes may preserve a still strongly observed RGB body, never a stale alpha.
   if self.previous_depth is None or largest<minimum or self.last_valid is None or now-self.last_valid>.25:
    return np.zeros_like(alpha),None
   z=self.previous_depth;chosen=int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))+1
  body=labels==chosen;body_area=stats[chosen,cv2.CC_STAT_AREA]
  # Select connected soft alpha, then recover nearby smaller fragments such as a separated hand.
  nsoft,soft,softstats,_=components(alpha>.08)
  main_ids=np.unique(soft[body]);main_ids=main_ids[main_ids!=0]
  keep=np.isin(soft,main_ids)
  distance=cv2.distanceTransform((~keep).astype(np.uint8),cv2.DIST_L2,3)
  radius=min(48,max(16,alpha.shape[1]*.06))
  for i in range(1,nsoft):
   area=softstats[i,cv2.CC_STAT_AREA]
   if area<12 or area>body_area*.4 or i in main_ids:continue
   fragment=soft==i
   if distance[fragment].min()>radius:continue
   d=depth[fragment&(depth>.35)&(depth<max_range)]
   supported=d.size>=8 and abs(float(np.median(d))-z)<tolerance
   if supported or float(alpha[fragment].max())>.9:keep|=fragment
  # Reject other substantial person cores. Approximate pixel depth never punches holes in
  # a confidently identified selected arm/body. Depth ranks actors, not individual skin pixels.
  for other_z,other in candidates:
   if other!=chosen and other_z>z+.20:
    rejected=cv2.dilate((labels==other).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    keep[rejected]=False
  output=np.where(keep,alpha,0).astype(np.float32)
  if background_match is not None:
   # Frozen empty-room agreement only removes uncertain pixels. Strong body RGB wins.
   output[(background_match)&(alpha<.85)]=0
  # Small noise floor; retain soft hair/edge alpha, make nearly opaque interiors exact RGB.
  output[output<.055]=0;output[output>.985]=1
  return output,z
class EmptyRoom:
 """Explicit frozen clean plate; never updates underneath a stationary actor."""
 def __init__(self):self.clear()
 def clear(self):self.rgb=None;self.depth=None;self.pending=None;self.samples=[];self.status='No empty-room reference'
 def request(self,now):self.pending=now+5;self.samples=[];self.status='Step out: capture begins in 5 seconds'
 def update(self,rgb,depth,alpha,now):
  if self.pending is None:return
  if now<self.pending:self.status=f'Step out: {self.pending-now:.0f}s';return
  if (alpha>.5).mean()>.008:
   self.samples=[];self.status='Waiting for an empty room — person still detected'
   if now-self.pending>20:self.pending=None;self.status='Capture cancelled: room was not empty'
   return
  self.samples.append((rgb.copy(),depth.copy()));self.status=f'Capturing empty room {len(self.samples)}/12'
  if len(self.samples)==12:
   self.rgb=np.median(np.stack([x[0] for x in self.samples]),axis=0).astype(np.uint8)
   self.depth=np.median(np.stack([x[1] for x in self.samples]),axis=0).astype(np.float32)
   self.samples=[];self.pending=None;self.status='Empty-room reference active; keep camera fixed'
 def match(self,rgb,depth):
  if self.rgb is None:return None
  # Compensate small global exposure drift using a robust channel offset.
  diff=rgb.astype(np.float32)-self.rgb.astype(np.float32);offset=np.median(diff[::8,::8],axis=(0,1))
  colour=np.max(np.abs(diff-offset),axis=2)<18
  agreement=(depth>.35)&(self.depth>.35)&(np.abs(depth-self.depth)<.10)
  return colour&agreement
