"""Explicit native-HD to existing calibration coordinate mapping.
The measured map is registration evidence, not a qualified calibration.
Native pixels are always retained alongside the virtual 640x480 image.
"""
import cv2,numpy as np
from .calibration import detect_robust

def prepare_rgb(native,profile,square):
    if native.shape[:2]!=(profile['height'],profile['width']):
        raise ValueError('RGB profile resolution mismatch')
    H=np.asarray(profile['H_native_to_calibration'],np.float64)
    points,ids=detect_robust(native,square)
    if len(ids):
        points=cv2.perspectiveTransform(points.astype(np.float32)[:,None],H)[:,0]
        keep=np.isfinite(points).all(axis=1)&(points[:,0]>=0)&(points[:,0]<640)&(points[:,1]>=0)&(points[:,1]<480)
        points=points[keep];ids=ids[keep]
    virtual=cv2.warpPerspective(native,H,(640,480),flags=cv2.INTER_LINEAR)
    return virtual,points,ids
