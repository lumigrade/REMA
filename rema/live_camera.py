import ctypes as C
import collections,threading,time
from pathlib import Path
import cv2,numpy as np
class Cameras:
 def __init__(self,mode="depth",rgb_mode=None):
  self.rgb_mode=rgb_mode or dict(width=640,height=480,fourcc="MJPG",fps=30)
  self.depth_intrinsics=None;self.mode=mode;self.projector_request=None;self.projector_on=True;self.control_status='';self.rgb_device=None;self.stop_event=threading.Event();self.lock=threading.Lock();self.rgb=None;self.rgb_history=collections.deque(maxlen=12);self.pair_offset_ns=0;self.depth=collections.deque(maxlen=16);self.error='';self.threads=[]
 def start(self):
  for fn in (self.capture_rgb,self.capture_depth):
   t=threading.Thread(target=self.guarded,args=(fn,),daemon=True);self.threads.append(t);t.start()
 def guarded(self,fn):
  try:fn()
  except Exception as e:self.error=str(e);self.stop_event.set()
 def pair(self):
  with self.lock:
   if self.rgb is None or not self.depth:return None
   rgb=self.rgb
   if self.pair_offset_ns>0:
    ready=[r for r in self.rgb_history if r[1]+self.pair_offset_ns<=self.depth[-1][1]]
    if not ready:return None
    rgb=ready[-1]
   depth=min(self.depth,key=lambda d:abs(d[1]-(rgb[1]+self.pair_offset_ns)))
   return rgb,depth
 def capture_rgb(self):
  devices=[]
  for p in sorted(Path('/sys/class/video4linux').glob('video*')):
   for parent in (p/'device').resolve().parents:
    if (parent/'idVendor').exists():
     if (parent/'idVendor').read_text().strip()=='2bc5' and (parent/'idProduct').read_text().strip()=='0501':devices.append('/dev/'+p.name)
     break
  cap=None
  try:
   for dev in devices:
    candidate=cv2.VideoCapture(dev,cv2.CAP_V4L2)
    if candidate.isOpened():cap=candidate;self.rgb_device=dev;break
    candidate.release()
   if cap is None:raise RuntimeError('Astra Pro RGB endpoint 2bc5:0501 unavailable')
   cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*self.rgb_mode['fourcc']));cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.rgb_mode['width']);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.rgb_mode['height']);cap.set(cv2.CAP_PROP_FPS,self.rgb_mode['fps']);cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
   seq=0
   while not self.stop_event.is_set():
    ok,frame=cap.read();now=time.monotonic_ns()
    if not ok:raise RuntimeError('RGB frame read failed')
    if frame.shape!=(self.rgb_mode['height'],self.rgb_mode['width'],3):raise RuntimeError('RGB negotiated unexpected resolution')
    seq+=1
    with self.lock:self.rgb=(seq,now,frame);self.rgb_history.append(self.rgb)
  finally:
   if cap:cap.release()
 def capture_depth(self):
  lib=C.CDLL('/home/pc/rema-data/work/openni_live_bridge.so')
  lib.rema_open.argtypes=[C.c_char_p,C.c_int];lib.rema_open.restype=C.c_void_p
  lib.rema_read.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),C.POINTER(C.c_float),C.c_char_p,C.c_int];lib.rema_read.restype=C.c_int
  lib.rema_close.argtypes=[C.c_void_p];lib.rema_close.restype=None
  lib.rema_ir_open.argtypes=[C.c_char_p,C.c_int];lib.rema_ir_open.restype=C.c_void_p
  lib.rema_projector.argtypes=[C.c_void_p,C.c_int];lib.rema_projector.restype=C.c_int
  err=C.create_string_buffer(1024);ptr=(lib.rema_ir_open if self.mode=='ir' else lib.rema_open)(err,len(err))
  if not ptr:raise RuntimeError(err.value.decode())
  if self.mode=='depth':
   lib.rema_intrinsics.argtypes=[C.c_void_p,C.c_void_p];lib.rema_intrinsics.restype=C.c_int
   intr=np.zeros(4,np.float32)
   if lib.rema_intrinsics(ptr,intr.ctypes.data)==0 and np.isfinite(intr).all() and (intr[:2]>0).all():self.depth_intrinsics=intr.tolist()
  try:
   misses=0
   while not self.stop_event.is_set():
    if self.projector_request is not None:
     desired=self.projector_request;self.projector_request=None;rc=lib.rema_projector(ptr,int(desired));self.control_status='Projector changed' if rc==0 else 'Projector setting rejected'
     if rc==0:self.projector_on=desired
    raw=np.empty((480,640),np.uint16);seq=C.c_uint64();stamp=C.c_uint64();scale=C.c_float()
    result=lib.rema_read(ptr,raw.ctypes.data,C.byref(seq),C.byref(stamp),C.byref(scale),err,len(err));now=time.monotonic_ns()
    if result<0:raise RuntimeError(err.value.decode())
    if not result:
     misses+=1
     if misses>=3:raise RuntimeError('Depth stream timeout')
     continue
    misses=0
    with self.lock:self.depth.append((seq.value,now,raw if self.mode=="ir" else raw.astype(np.float32)*(scale.value/1000),stamp.value))
  finally:lib.rema_close(ptr)
 def close(self):
  self.stop_event.set()
  for t in self.threads:t.join(2)
