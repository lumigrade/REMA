"""P1 marker-assisted world anchor, using the calibrated printed floor target.
Object/world axes are the printed board X/Y; -Z is above the printed surface.
R,t map board/world to the CURRENT RGB camera. No stale pose, optical-flow
prediction or assumed room/floor registration when the target is missing.
"""
import cv2
import numpy as np
from .calibration import board,detect

class FloorTargetTracker:
    def __init__(self,calibration):
        if not all(calibration.get(k) for k in ('qualified','stereo_pass','depth_validation_pass')):
            raise ValueError('A qualified RGB/depth calibration is required for floor tracking')
        self.K=np.array(calibration['K_rgb'],float);self.dist=np.array(calibration['dist_rgb'],float)
        if self.K.shape!=(3,3) or not np.isfinite(self.K).all() or min(self.K[0,0],self.K[1,1])<=0:
            raise ValueError('Invalid camera intrinsics')
        self.square=float(calibration['square_metres']);self.object_points=board(self.square).getChessboardCorners()
        self.calibration_id=calibration.get('calibration_id','synthetic-test')
        self.reset()

    def reset(self):
        self.previous=None;self.last_frame=None;self.last_time=None
        self.pending=None;self.lock_count=0

    def lost(self,reason,frame_id,timestamp,corners=0):
        self.previous=None;self.pending=None;self.lock_count=0
        return dict(valid=False,reason=reason,provider='fixed_charuco_floor_target',
                    frame_id=int(frame_id),rgb_host_arrival_ns=int(timestamp),corners=int(corners),
                    calibration_id=self.calibration_id,world_anchor='printed_board_xy_floor_negative_z_up')

    def update(self,rgb,person_alpha,frame_id,timestamp):
        points,ids=detect(rgb,self.square)
        if person_alpha is not None and len(ids):
            mask=cv2.dilate((person_alpha>.3).astype(np.uint8),np.ones((9,9),np.uint8))
            px=np.rint(points).astype(int);good=(px[:,0]>=0)&(px[:,0]<640)&(px[:,1]>=0)&(px[:,1]<480)
            keep=np.zeros(len(ids),bool);idx=np.flatnonzero(good)
            keep[idx]=mask[px[idx,1],px[idx,0]]==0
            points=points[keep];ids=ids[keep]
        return self.update_points(points,ids,frame_id,timestamp)

    def update_points(self,points,ids,frame_id,timestamp):
        if self.last_frame is not None and (frame_id<=self.last_frame or timestamp<=self.last_time):
            return self.lost('Non-monotonic RGB frame; refusing stale pose',frame_id,timestamp)
        previous_time=self.last_time;self.last_time=int(timestamp);self.last_frame=int(frame_id)
        if previous_time is not None and timestamp-previous_time>500_000_000:
            self.previous=None;self.pending=None;self.lock_count=0
        points=np.asarray(points,np.float32);ids=np.asarray(ids,int)
        if len(ids)<20 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(self.object_points)):
            return self.lost('Keep at least 20 target corners visible; actor must not cover the target',frame_id,timestamp,len(ids))
        if points.shape!=(len(ids),2) or not np.isfinite(points).all():
            return self.lost('Invalid corner coordinates',frame_id,timestamp,len(ids))
        if cv2.contourArea(cv2.convexHull(points))<1200:
            return self.lost('Floor target too small in the image',frame_id,timestamp,len(ids))
        obj=self.object_points[ids]
        # Reject outliers in undistorted pixels before planar pose estimation.
        undist=cv2.undistortPoints(points[:,None],self.K,self.dist,P=self.K).reshape(-1,2)
        _,inliers=cv2.findHomography(obj[:,:2],undist,cv2.RANSAC,2.5)
        if inliers is None or inliers.sum()<20 or inliers.mean()<.85:
            return self.lost('Target corners inconsistent',frame_id,timestamp,len(ids))
        keep=inliers.ravel().astype(bool);obj=obj[keep];pixels=points[keep]
        answer=cv2.solvePnPGeneric(obj,pixels,self.K,self.dist,flags=cv2.SOLVEPNP_IPPE)
        candidates=[]
        if answer[0]:
            for r,t in zip(answer[1],answer[2]):
                R=cv2.Rodrigues(r)[0];camera_points=obj@R.T+t.ravel()
                if not np.isfinite(camera_points).all() or np.min(camera_points[:,2])<.15:continue
                projected,_=cv2.projectPoints(obj,r,t,self.K,self.dist)
                errors=np.linalg.norm(projected.reshape(-1,2)-pixels,axis=1)
                candidates.append((float(np.mean(errors)),r,t,float(np.median(errors)),float(np.percentile(errors,95))))
        if not candidates:return self.lost('No forward-facing target pose',frame_id,timestamp,len(ids))
        candidates.sort(key=lambda p:p[0]);best=candidates[0]
        if len(candidates)>1 and candidates[1][0]-best[0]<.10:
            relative=cv2.Rodrigues(best[1])[0]@cv2.Rodrigues(candidates[1][1])[0].T
            angle=np.linalg.norm(cv2.Rodrigues(relative)[0])
            if angle>np.radians(8):
                return self.lost('Planar pose ambiguous; view the floor target at a clearer angle',frame_id,timestamp,len(ids))
        _,r,t,median,p95=best
        r,t=cv2.solvePnPRefineLM(obj,pixels,self.K,self.dist,r,t)
        projected,_=cv2.projectPoints(obj,r,t,self.K,self.dist)
        errors=np.linalg.norm(projected.reshape(-1,2)-pixels,axis=1);median=float(np.median(errors));p95=float(np.percentile(errors,95))
        R=cv2.Rodrigues(r)[0];position=-R.T@t.ravel()
        if median>1 or p95>3 or np.linalg.norm(t)>5 or not np.isfinite(position).all():
            return self.lost('Pose reprojection / range check failed',frame_id,timestamp,len(ids))
        current=(R,t.ravel(),position)
        reference=self.previous or self.pending
        dt=(timestamp-previous_time)/1e9 if previous_time is not None else 0
        if reference is not None and 0<dt<.5:
            translation=np.linalg.norm(position-reference[2])
            angle=np.linalg.norm(cv2.Rodrigues(R@reference[0].T)[0])
            if translation>max(.08,2.0*dt) or angle>max(np.radians(15),np.radians(180)*dt):
                return self.lost('Pose jump rejected; keep target fixed and move camera smoothly',frame_id,timestamp,len(ids))
        self.pending=current;self.lock_count+=1
        if self.lock_count<2:
            return dict(valid=False,reason='Acquiring floor target',provider='fixed_charuco_floor_target',
                        frame_id=int(frame_id),rgb_host_arrival_ns=int(timestamp),corners=len(ids),calibration_id=self.calibration_id)
        self.previous=current
        world_from_camera=np.eye(4);world_from_camera[:3,:3]=R.T;world_from_camera[:3,3]=position
        return dict(valid=True,reason='Floor target tracked',provider='fixed_charuco_floor_target',
                    frame_id=int(frame_id),rgb_host_arrival_ns=int(timestamp),corners=int(len(obj)),
                    rvec=r.ravel().tolist(),tvec_metres=t.ravel().tolist(),world_from_camera=world_from_camera.tolist(),
                    reprojection_median_px=median,reprojection_p95_px=p95,calibration_id=self.calibration_id,
                    world_anchor='printed_board_xy_floor_negative_z_up',hardware_tracking_qualified=False)


def tracked_scene(pose,K,dist,size=(640,480),square=.02317):
    """Metric 25 cm grid and three shaded blocks; actor is composited over this.
    All geometry is projected with the current frame pose and calibrated distortion.
    Near-plane / image clipping prevents giant lines when the floor crosses camera Z.
    """
    if not pose.get('valid'):raise ValueError('Cannot render an untracked or stale pose')
    w,h=size;image=np.full((h,w,3),(41,34,29),np.uint8)
    R=cv2.Rodrigues(np.array(pose['rvec'],float))[0];t=np.array(pose['tvec_metres'],float)
    K=np.asarray(K,float);dist=np.asarray(dist,float)
    center=np.array([5*square,3.5*square,0.])
    def projected(points):
        camera=np.asarray(points)@R.T+t
        uv,_=cv2.projectPoints(camera,np.zeros(3),np.zeros(3),K,dist)
        return camera,uv.reshape(-1,2)
    def line(points,color,thickness=1):
        camera,pixels=projected(points)
        valid=(camera[:,2]>=.08)&np.isfinite(pixels).all(axis=1)&(np.max(np.abs(pixels),axis=1)<1e6)
        # Draw continuous valid runs in native OpenCV instead of ~2400 Python
        # per-segment clipping calls. OpenCV clips raster strokes to the image.
        boundaries=np.flatnonzero(np.diff(np.r_[False,valid,False]))
        for first,last in zip(boundaries[::2],boundaries[1::2]):
            if last-first>=2:cv2.polylines(image,[np.rint(pixels[first:last]).astype(np.int32)],False,color,thickness,cv2.LINE_AA)
    axis=np.linspace(-3,3,49)
    for k in range(-12,13):
        offset=k*.25
        line(np.c_[axis,np.full_like(axis,offset),np.zeros_like(axis)]+center,(82,79,72) if k else (70,110,190),2 if k==0 else 1)
        line(np.c_[np.full_like(axis,offset),axis,np.zeros_like(axis)]+center,(82,79,72) if k else (95,155,90),2 if k==0 else 1)
    faces=[]
    for x,y,width,height,color in [(.45,.25,.25,.30,(188,126,70)),(-.55,.50,.30,.20,(89,158,197)),(.0,1.,.35,.45,(130,171,101))]:
        vertices=np.array([[x,y,0],[x+width,y,0],[x+width,y+width,0],[x,y+width,0],
                           [x,y,-height],[x+width,y,-height],[x+width,y+width,-height],[x,y+width,-height]])+center
        camera,pixels=projected(vertices)
        for index,face in enumerate([(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7),(4,5,6,7)]):
            ids=list(face)
            if np.min(camera[ids,2])<.08:continue
            poly=pixels[ids]
            if not np.isfinite(poly).all() or np.max(np.abs(poly))>1e6:continue
            shade=[.65,.8,.65,.8,1.0][index]
            faces.append((float(camera[ids,2].mean()),np.rint(poly).astype(np.int32),tuple(int(c*shade) for c in color)))
    for _,polygon,color in sorted(faces,key=lambda f:-f[0]):
        cv2.fillConvexPoly(image,polygon,color,cv2.LINE_AA);cv2.polylines(image,[polygon],True,(30,30,30),1,cv2.LINE_AA)
    return image
