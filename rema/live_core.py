"""CPU prototype: semantic person mask, approximate depth association, static grid.
Depth units: metres. No calibrated reprojection, exposure synchronization or pose.
"""
from rema import paths as _P
import cv2
import numpy as np
cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(False)
MODEL=str(_P.DATA)+'/models/pphumanseg/human_segmentation_pphumanseg_2023mar.onnx'
class PersonModel:
 def __init__(self):
  self.net=cv2.dnn.readNet(MODEL)
  self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
  self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
 def infer(self,bgr):
  rgb=cv2.cvtColor(cv2.resize(bgr,(192,192)),cv2.COLOR_BGR2RGB).astype(np.float32)/127.5-1
  self.net.setInput(cv2.dnn.blobFromImage(rgb))
  out=self.net.forward()[0]
  return cv2.resize(out[1],(bgr.shape[1],bgr.shape[0]),interpolation=cv2.INTER_LINEAR)
def align_depth(depth,shape,scale=1,dx=0,dy=0):
 h,w=shape;d=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
 mat=np.float32([[scale,0,(1-scale)*w/2+dx],[0,scale,(1-scale)*h/2+dy]])
 return cv2.warpAffine(d,mat,(w,h),flags=cv2.INTER_NEAREST)
def select_person(prob,depth,max_range=4.0,tolerance=0.9):
 """Select closest separated semantic component by robust valid interior depth.
 Invalid depth may preserve edges of a selected person, never select an unmeasured one.
 Touching/overlapping people can merge; this is not instance segmentation.
 """
 binary=(prob>.5).astype(np.uint8)
 binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
 n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
 candidates=[]
 for i in range(1,n):
  if stats[i,cv2.CC_STAT_AREA]<500:continue
  inside=cv2.erode((labels==i).astype(np.uint8),np.ones((9,9),np.uint8))>0
  valid=depth[inside&(depth>.35)&(depth<max_range)]
  if len(valid)<80:continue
  z=float(np.percentile(valid,35));candidates.append((z,i))
 if not candidates:return np.zeros_like(prob),None
 z,idx=min(candidates)
 region=cv2.dilate((labels==idx).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
 # Broad range support; tolerate missing depth and a thin uncertain registration boundary.
 allowed=(depth==0)|((depth>max(.35,z-tolerance))&(depth<min(max_range,z+tolerance)))
 allowed=cv2.dilate(allowed.astype(np.uint8),np.ones((13,13),np.uint8)).astype(bool)
 alpha=np.clip((prob-.35)/.3,0,1)*region*allowed
 alpha=cv2.GaussianBlur(alpha.astype(np.float32),(3,3),.6)
 alpha[alpha>.98]=1;alpha[alpha<.02]=0
 return alpha,z
def grid(w=640,h=480,horizon=.40):
 # Illustrative pinhole projection. x right, y up, z forward; metres arbitrary until calibrated.
 image=np.full((h,w,3),57,np.uint8);hy=int(h*horizon)
 image[hy:]=np.array([65,65,65],np.uint8)
 f=w*.82;camera_height=1.5
 def point(x,z):return (int(w/2+f*x/z),int(hy+f*camera_height/z))
 for x in np.arange(-20,20.1,.5):
  color=(92,92,92) if abs(x-round(x))<.01 else (76,76,76)
  if abs(x)<.01:color=(88,142,88)
  cv2.line(image,point(x,1),point(x,60),color,1,cv2.LINE_AA)
 for z in np.arange(1,60,.5):
  color=(92,92,92) if abs(z-round(z))<.01 else (76,76,76)
  if z==3:color=(85,85,176)
  cv2.line(image,point(-30,z),point(30,z),color,1,cv2.LINE_AA)
 return image
def composite(rgb,alpha,background):
 return np.rint(rgb.astype(np.float32)*alpha[...,None]+background.astype(np.float32)*(1-alpha[...,None])).astype(np.uint8)
def depth_preview(depth,max_range=5):
 values=np.clip(depth/max_range*255,0,255).astype(np.uint8)
 color=cv2.applyColorMap(255-values,cv2.COLORMAP_TURBO);color[depth<=0]=0
 return color
