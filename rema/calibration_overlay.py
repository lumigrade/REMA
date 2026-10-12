"""40% filled RGB-white target guide. Display-only; raw calibration is untouched."""
import cv2
import numpy as np
from .calibration_guide import mirror_points

COLORS=np.float32([[0,0,255],[0,255,0],[255,0,0],[255,255,255]]) # BGR
OPACITY=.40

def estimate_guide_K(samples):
    """Provisional RGB focal lengths from saved metric board homographies.
    Centred principal point, no distortion and depth Z ~= RGB Z are assumptions
    for the guide only. This estimate is NEVER an activated calibration.
    """
    focals=[]
    for sample in samples:
        H,_=cv2.findHomography(sample['obj'][:,:2],sample['rgb'])
        if H is None:continue
        square=float(sample['square_metres'])
        center=np.array([5*square,3.5*square,1.])
        z=float(np.median(sample['plane_points'][:,2]))
        H=H*z/(H@center)[2]
        h=H.copy();h[0]-=320*h[2];h[1]-=240*h[2]
        A=np.array([[h[0,0]**2,h[1,0]**2],[h[0,1]**2,h[1,1]**2],[h[0,0]*h[0,1],h[1,0]*h[1,1]]])
        b=np.array([1-h[2,0]**2,1-h[2,1]**2,-h[2,0]*h[2,1]])
        solution=np.linalg.lstsq(A,b,rcond=None)[0]
        if np.all(solution>0):
            f=1/np.sqrt(solution)
            if np.isfinite(f).all() and np.all((f>300)&(f<1200)):focals.append(f)
    fx,fy=np.median(focals,axis=0) if focals else (600.,600.)
    return np.array([[fx,0,320],[0,fy,240],[0,0,1]],float)

def target_polygon(pose,K,dist=None,R_depth_to_rgb=None):
    """A4 board with target normal, distance and image centre; native RGB pixels."""
    tx,ty=np.radians(pose.tilt)
    rotation=cv2.Rodrigues(np.array([-ty,tx,0.]))[0]
    if R_depth_to_rgb is not None:rotation=np.array(R_depth_to_rgb)@rotation
    dist=np.zeros(5) if dist is None else np.array(dist)
    ray=cv2.undistortPoints(np.array([[pose.center]],np.float64),K,dist)[0,0]
    t=np.r_[ray,1.]*pose.distance
    corners=np.float32([[-.1485,-.105,0],[.1485,-.105,0],[.1485,.105,0],[-.1485,.105,0]])
    pixels,_=cv2.projectPoints(corners,cv2.Rodrigues(rotation)[0],t,K,dist)
    return pixels.reshape(-1,2)

def filled_overlay(raw,polygon,seconds,mirror=True):
    base=cv2.flip(raw,1) if mirror else raw.copy()
    if polygon is None:return base
    vertices=mirror_points(polygon,raw.shape[1]) if mirror else np.asarray(polygon)
    mask=np.zeros(raw.shape[:2],np.uint8)
    cv2.fillConvexPoly(mask,np.rint(vertices).astype(np.int32),1)
    y,x=np.nonzero(mask)
    if not len(x):return base
    # Smooth R -> G -> B -> white -> R, with a diagonal colour gradient across
    # a uniformly filled polygon. Alpha remains exactly .40 through the cycle.
    span=np.maximum(np.ptp(vertices,axis=0),1)
    gradient=((x-vertices[:,0].min())/span[0]+(y-vertices[:,1].min())/span[1])*.22
    phase=(float(seconds)+gradient)%4
    index=np.floor(phase).astype(int);fraction=(phase-index)[:,None]
    color=COLORS[index]*(1-fraction)+COLORS[(index+1)%4]*fraction
    base[y,x]=np.rint(base[y,x]*(1-OPACITY)+color*OPACITY).astype(np.uint8)
    return base
