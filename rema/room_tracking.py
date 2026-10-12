"""Bounded marker-free RGB-D prototype: static ORB landmarks + RANSAC PnP.
A small keyframe map keeps the initial world frame on relocalization. No loop
closure, global optimization, inertial gravity, or physical accuracy claim.
"""
import cv2,numpy as np


def inverse(T):
    result=np.eye(4);result[:3,:3]=T[:3,:3].T;result[:3,3]=-T[:3,:3].T@T[:3,3]
    return result


def floor_from_depth(depth,K,person=None,dist=None):
    """Estimate a lower-view plane with a camera-down normal. No IMU gravity:
    this assumes the starting camera is reasonably upright and roughly level.
    Returns None for walls/weak support; accepted plane is frozen by the caller.
    """
    y,x=np.indices(depth.shape);mask=(depth>.35)&(depth<4)&(y>depth.shape[0]*.52)
    if person is not None:mask &= person<.15
    v,u=np.where(mask);z=depth[v,u]
    if len(z)<1500:return None
    stride=max(1,len(z)//4000);v,u,z=v[::stride],u[::stride],z[::stride]
    rays=cv2.undistortPoints(np.c_[u,v].astype(np.float32)[:,None],K,np.zeros(5) if dist is None else dist).reshape(-1,2)
    points=np.c_[rays*z[:,None],z];rng=np.random.default_rng(41);best=None;best_score=0
    for _ in range(90):
        tri=points[rng.choice(len(points),3,replace=False)];n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);length=np.linalg.norm(n)
        if length<1e-7:continue
        n/=length
        if n[1]<0:n=-n
        if n[1]<.80:continue
        offset=-n@tri[0];distance=-offset
        if not .25<distance<2.5:continue
        use=np.abs(points@n+offset)<.025;count=int(use.sum())
        if count<400:continue
        span=np.ptp(points[use],axis=0)
        if span[0]<.5 or span[2]<.3:continue
        score=count*(1+distance*.1)
        if score>best_score:best_score=score;best=(n,offset,use)
    if best is None:return None
    n,d,use=best;p=points[use];center=np.median(p,axis=0)
    _,_,vt=np.linalg.svd(p-p.mean(0),full_matrices=False);n=vt[-1]
    if n[1]<0:n=-n
    d=-n@p.mean(0)
    if n[1]<.8 or np.percentile(np.abs(p@n+d),95)>.035:return None
    # Floor frame uses +Z down through the floor, so renderer -Z is above it.
    axis_x=np.array([1.,0,0]);axis_x-=n*(axis_x@n);axis_x/=np.linalg.norm(axis_x)
    axis_y=np.cross(n,axis_x)
    origin=np.array([0.,0.,float(center[2])]);origin-=n*(n@origin+d)
    T=np.eye(4);T[:3,:3]=np.column_stack([axis_x,axis_y,n]);T[:3,3]=origin
    return T


class RoomTracker:
    def __init__(self,calibration):
        if not (calibration.get('qualified') or (calibration.get('provisional') and calibration.get('user_authorized'))):
            raise ValueError('Tracking requires measured calibration or explicit provisional mode')
        self.K=np.array(calibration['K_rgb'],float);self.dist=np.array(calibration['dist_rgb'],float)
        self.calibration_id=calibration['calibration_id'];self.provisional=bool(calibration.get('provisional'))
        self.orb=cv2.ORB_create(nfeatures=1100,scaleFactor=1.2,nlevels=6,edgeThreshold=19,fastThreshold=14)
        self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        self.preview_height=1.2;self.origin_generation=-1;self.reset()

    def reset(self):
        self.origin_generation+=1
        self.keyframes=[];self.last_frame=None;self.last_time=None;self.last_pose=None;self.pending_pose=None
        self.flow_reference=None;self.flow_steps=0
        self.good_count=0;self.floor=None;self.floor_mode='preview_height';self.keyframe_clock=0

    def extract(self,rgb,depth,person=None):
        gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        # Depth boundaries are deliberately excluded under provisional alignment.
        valid=np.isfinite(depth)&(depth>.35)&(depth<5)
        safe=np.where(valid,depth,0).astype(np.float32)
        lower=cv2.erode(safe,np.ones((5,5),np.uint8));upper=cv2.dilate(safe,np.ones((5,5),np.uint8))
        mask=valid&(lower>.35)&((upper-lower)<.07)
        if person is not None:
            excluded=cv2.dilate((person>.20).astype(np.uint8),np.ones((21,21),np.uint8))
            mask &= excluded==0
        keypoints,descriptors=self.orb.detectAndCompute(gray,mask.astype(np.uint8)*255)
        if descriptors is None:return dict(pixels=np.empty((0,2),np.float32),points=np.empty((0,3),np.float32),descriptors=None)
        pixels=np.array([k.pt for k in keypoints],np.float32);uv=np.rint(pixels).astype(int)
        z=depth[uv[:,1],uv[:,0]];rays=cv2.undistortPoints(pixels[:,None],self.K,self.dist).reshape(-1,2)
        points=np.c_[rays*z[:,None],z].astype(np.float32)
        return dict(pixels=pixels,points=points,descriptors=descriptors)

    def coverage(self,pixels):
        if len(pixels)==0:return 0
        cells=np.minimum([3,2],np.maximum([0,0],(pixels/[160,160]).astype(int)))
        return len(np.unique(cells,axis=0))

    def status(self,valid,reason,frame_id,timestamp,**extra):
        return dict(valid=valid,reason=reason,frame_id=int(frame_id),rgb_host_arrival_ns=int(timestamp),
                    provider='rgbd_orb_keyframe_map',provisional=self.provisional,calibration_id=self.calibration_id,
                    tracking_qualified=False,origin_generation=self.origin_generation,floor_mode=self.floor_mode,keyframes=len(self.keyframes),**extra)

    def fail(self,reason,frame_id,timestamp,**extra):
        self.good_count=0;self.last_pose=None;self.pending_pose=None;self.flow_reference=None;self.flow_steps=0
        return self.status(False,reason,frame_id,timestamp,**extra)

    def add_keyframe(self,features,Tworld_camera,frame_id):
        self.keyframes.append(dict(pixels=features['pixels'],descriptors=features['descriptors'],
                              world_points=features['points']@Tworld_camera[:3,:3].T+Tworld_camera[:3,3],frame_id=frame_id))
        # Keep original anchor plus seven recent keyframes. Memory stays bounded.
        if len(self.keyframes)>8:del self.keyframes[1]
        self.keyframe_clock=frame_id

    def solve_matches(self,world_points,pixels,current_points):
        if len(pixels)<30 or self.coverage(pixels)<4:return None
        ok,r,t,inliers=cv2.solvePnPRansac(np.asarray(world_points,np.float32),np.asarray(pixels,np.float32),
                    self.K,self.dist,iterationsCount=100,reprojectionError=2.5,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers)<25 or len(inliers)/len(pixels)<.5:return None
        ids=inliers.ravel()
        if self.coverage(pixels[ids])<4:return None
        r,t=cv2.solvePnPRefineLM(world_points[ids].astype(np.float32),pixels[ids].astype(np.float32),self.K,self.dist,r,t)
        R=cv2.Rodrigues(r)[0];camera=world_points[ids]@R.T+t.ravel()
        projected,_=cv2.projectPoints(world_points[ids],r,t,self.K,self.dist)
        errors=np.linalg.norm(projected.reshape(-1,2)-pixels[ids],axis=1)
        residual=np.linalg.norm(camera-current_points[ids],axis=1)
        if not np.isfinite(camera).all() or np.min(camera[:,2])<.2:return None
        if np.percentile(errors,95)>2.5 or np.median(residual)>.05 or np.percentile(residual,90)>.12:return None
        Tcamera_world=np.eye(4);Tcamera_world[:3,:3]=R;Tcamera_world[:3,3]=t.ravel()
        return dict(Tworld_camera=inverse(Tcamera_world),inliers=len(ids),matches=len(pixels),tracked_world=world_points[ids],tracked_pixels=pixels[ids],
                    reprojection_p95_px=float(np.percentile(errors,95)),depth_residual_p50_m=float(np.median(residual)))

    def follow_features(self,gray,depth,person):
        if self.flow_reference is None:return None
        old,pixels,world=self.flow_reference
        forward,ok,err=cv2.calcOpticalFlowPyrLK(old,gray,pixels.astype(np.float32)[:,None],None,winSize=(21,21),maxLevel=3)
        if forward is None:return None
        back,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,old,forward,None,winSize=(21,21),maxLevel=3)
        if back is None:return None
        uv=forward.reshape(-1,2);back=back.reshape(-1,2)
        good=ok.ravel().astype(bool)&back_ok.ravel().astype(bool)&(err.ravel()<25)&(np.linalg.norm(back-pixels,axis=1)<.8)
        good &= np.isfinite(uv).all(axis=1)&(uv[:,0]>=3)&(uv[:,0]<637)&(uv[:,1]>=3)&(uv[:,1]<477)
        idx=np.flatnonzero(good)
        if len(idx)<30:return None
        px=np.rint(uv[idx]).astype(int);z=depth[px[:,1],px[:,0]]
        safe=np.where(np.isfinite(depth),depth,0).astype(np.float32)
        lower=cv2.erode(safe,np.ones((5,5),np.uint8));upper=cv2.dilate(safe,np.ones((5,5),np.uint8))
        good=(z>.35)&(z<5)&(lower[px[:,1],px[:,0]]>.35)&((upper-lower)[px[:,1],px[:,0]]<.07)
        if person is not None:
            excluded=cv2.dilate((person>.20).astype(np.uint8),np.ones((21,21),np.uint8));good &= excluded[px[:,1],px[:,0]]==0
        idx=idx[good];z=z[good]
        if len(idx)<30:return None
        rays=cv2.undistortPoints(uv[idx,None],self.K,self.dist).reshape(-1,2)
        return self.solve_matches(world[idx],uv[idx],np.c_[rays*z[:,None],z]) if len(idx)>=30 else None

    def update(self,rgb,depth,person,frame_id,timestamp):
        if self.last_frame is not None and (frame_id<=self.last_frame or timestamp<=self.last_time):
            return self.fail('Stale RGB frame rejected',frame_id,timestamp)
        dt=(timestamp-self.last_time)/1e9 if self.last_time is not None else 0
        self.last_frame=int(frame_id);self.last_time=int(timestamp)
        if dt>.5:self.good_count=0;self.pending_pose=None;self.last_pose=None;self.flow_reference=None
        gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY);features=None;best=None;method='descriptor_map'
        # Current-frame optical measurements, never extrapolation or a held pose.
        # Refresh from descriptors every four flow frames to bound optical drift.
        if self.keyframes and 0<dt<.2 and self.flow_steps<4:
            best=self.follow_features(gray,depth,person)
            if best:method='optical_flow_rgbd'
        if best is None:
            features=self.extract(rgb,depth,person);count=len(features['pixels'])
            if not self.keyframes:
                if count<50 or self.coverage(features['pixels'])<4:
                    return self.fail('Need more static room texture / valid depth',frame_id,timestamp,features=count)
                self.add_keyframe(features,np.eye(4),frame_id)
                floor=floor_from_depth(depth,self.K,person,self.dist)
                if floor is not None:self.floor=floor;self.floor_mode='estimated_depth_plane'
                else:
                    self.floor=np.eye(4);self.floor[:3,:3]=np.array([[1,0,0],[0,0,1],[0,-1,0]])
                    self.floor[:3,3]=[0,self.preview_height,3.5];self.floor_mode='preview_height'
                self.pending_pose=np.eye(4);self.good_count=1
                return self.status(False,'Learning room origin — keep camera steady briefly',frame_id,timestamp,features=count)
            references=[self.keyframes[0]]+[ref for ref in reversed(self.keyframes[-3:]) if ref is not self.keyframes[0]]
            if count>=50:
                for reference in references:
                    pairs=self.matcher.knnMatch(reference['descriptors'],features['descriptors'],k=2)
                    chosen=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.72*b.distance and a.distance<60]
                    unique={}
                    for match in chosen:
                        if match.trainIdx not in unique or match.distance<unique[match.trainIdx].distance:unique[match.trainIdx]=match
                    chosen=list(unique.values())
                    if len(chosen)<30:continue
                    qi=np.array([m.queryIdx for m in chosen]);ti=np.array([m.trainIdx for m in chosen])
                    candidate=self.solve_matches(reference['world_points'][qi],features['pixels'][ti],features['points'][ti])
                    if candidate and (best is None or candidate['inliers']>best['inliers']):best=candidate
                    if best and best['inliers']>90:break
            # One failed descriptor refresh may use validated flow, but cannot
            # postpone descriptor refresh for more than eight measured frames.
            if best is None and 0<dt<.2 and self.flow_steps<8:
                best=self.follow_features(gray,depth,person)
                if best:method='optical_flow_rgbd';features=None
        if best is None:return self.fail('Tracking lost — return toward a previously seen view',frame_id,timestamp,features=count)
        count=len(features['pixels']) if features is not None else best['matches']
        current=best['Tworld_camera'];prior=self.last_pose if self.last_pose is not None else self.pending_pose
        if prior is not None and 0<dt<.5:
            shift=np.linalg.norm(current[:3,3]-prior[:3,3]);angle=np.linalg.norm(cv2.Rodrigues(current[:3,:3]@prior[:3,:3].T)[0])
            if shift>max(.07,2*dt) or angle>max(np.radians(10),np.radians(120)*dt):
                return self.fail('Pose jump rejected — move camera more slowly',frame_id,timestamp,features=count)
        self.pending_pose=current;self.good_count+=1
        if self.good_count<2:return self.status(False,'Confirming relocalization',frame_id,timestamp,features=count)
        self.last_pose=current
        self.flow_steps=self.flow_steps+1 if method=='optical_flow_rgbd' else 0
        ids=np.linspace(0,len(best['tracked_pixels'])-1,min(220,len(best['tracked_pixels']))).astype(int)
        self.flow_reference=(gray,best['tracked_pixels'][ids].copy(),best['tracked_world'][ids].copy())
        if features is not None and frame_id-self.keyframe_clock>=20 and best['inliers']>=40:self.add_keyframe(features,current,frame_id)
        floor=self.floor.copy()
        if self.floor_mode=='preview_height':floor[1,3]=self.preview_height
        Tcamera_floor=inverse(current)@floor
        r=cv2.Rodrigues(Tcamera_floor[:3,:3])[0].ravel();t=Tcamera_floor[:3,3]
        return self.status(True,'Room tracked (temporary calibration)' if self.provisional else 'Room tracked',frame_id,timestamp,
                           rvec=r.tolist(),tvec_metres=t.tolist(),world_from_camera=current.tolist(),features=count,method=method,
                           inliers=best['inliers'],matches=best['matches'],reprojection_p95_px=best['reprojection_p95_px'],
                           depth_residual_p50_m=best['depth_residual_p50_m'])

    def estimate_floor_now(self,depth,person=None):
        if self.last_pose is None:return False
        plane=floor_from_depth(depth,self.K,person,self.dist)
        if plane is None:return False
        self.floor=self.last_pose@plane;self.floor_mode='estimated_depth_plane';return True
