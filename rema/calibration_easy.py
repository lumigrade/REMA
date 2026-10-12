"""Free-pose calibration capture; final measured solver/checks stay mandatory."""
import math
import cv2
import numpy as np
from .calibration import board,detect,detect_robust
from .calibration_guide import Guide,Pose

def robust_detect(image,square):
    return detect_robust(image,square)

def print_quality(image,points,ids,square):
    if len(ids)<12:return None
    H,_=cv2.findHomography(board(square).getChessboardCorners()[ids,:2],points,cv2.RANSAC,2)
    if H is None:return None
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    warp=np.diag([50/square,50/square,1.])@np.linalg.inv(H)
    rect=cv2.warpPerspective(gray,warp,(500,350))
    present=cv2.warpPerspective(np.full(gray.shape,255,np.uint8),warp,(500,350))>250
    target=board(square).generateImage((500,350))
    black=(cv2.erode((target<60).astype(np.uint8),np.ones((5,5),np.uint8))>0)&present
    white=(cv2.erode((target>200).astype(np.uint8),np.ones((5,5),np.uint8))>0)&present
    if black.sum()<100 or white.sum()<100:return None
    return dict(contrast=float(np.median(rect[white])-np.median(rect[black])),washed_black_percent=float((rect[black]>220).mean()*100))

class EasyGuide(Guide):
    def __init__(self):
        super().__init__();self.views=[];self.pending_view=None;self.last_capture_at=None;self.started_at=None
        self.feedback='Confirm the NEW square measurement, then press START. Move naturally; pause to save each useful view.'
    def restore(self,samples):
        self.views=[]
        for sample in samples:
            normal=np.array(sample['plane_normal'],float);tilt=np.degrees(np.arctan2(normal[:2],normal[2]))
            self.views.append(dict(center=np.mean(sample['rgb'],axis=0).tolist(),tilt=tilt.tolist(),normal=normal.tolist(),distance=float(np.median(sample['plane_points'][:,2]))))
        self.index=len(self.views)
    def start(self,now,index=0):
        super().start(now,index);self.started_at=now+10;self.last_capture_at=None
    def dashboard(self,now):
        c=self.coverage();checks=dict(positions=c['cells']>=4 and min(c['span'])>=120,distances=c['range_span']>=.4,side_tilt=c['tilt_span'][0]>=30,vertical_tilt=c['tilt_span'][1]>=30,tilt_variety=c['condition']<=12)
        remaining=max(0,24-len(self.views));missing=[k for k,ok in checks.items() if not ok]
        anchor=self.last_capture_at if self.last_capture_at is not None else self.started_at
        idle=max(0,now-anchor) if anchor is not None else 0
        return dict(saved=len(self.views),minimum=24,remaining=remaining,checks=checks,missing=missing,coverage_complete=not missing,seconds_since_save=round(idle),stalled=bool(self.phase=='capture' and idle>=25),recent_save=bool(self.last_capture_at is not None and now-self.last_capture_at<4))
    @property
    def pose(self):
        if self.phase=='capture':return Pose(.8,(320,240),(0,0))
        return super().pose
    def coverage(self,views=None):
        records=self.views if views is None else views
        train=[v for i,v in enumerate(records) if i%4!=3]
        if not train:return dict(views=len(records),cells=0,span=[0,0],tilt_span=[0,0],range_span=0,condition=999,ready=False)
        centers=np.array([v['center'] for v in train]);tilt=np.array([v['tilt'] for v in train]);normals=np.array([v['normal'] for v in train]);ranges=[v['distance'] for v in train]
        cells=len(np.unique(np.floor(centers/[160,160]).astype(int),axis=0));span=np.ptp(centers,axis=0);tilt_span=np.ptp(tilt,axis=0);condition=float(np.linalg.cond(normals)) if len(normals)>=3 else 999
        ready=len(records)>=24 and cells>=4 and min(span)>=120 and min(tilt_span)>=30 and np.ptp(ranges)>=.4 and condition<=12
        return dict(views=len(records),cells=cells,span=span.tolist(),tilt_span=tilt_span.tolist(),range_span=float(np.ptp(ranges)),condition=condition,ready=bool(ready))
    def suggestion(self):
        c=self.coverage()
        if len(self.views)<3:return 'Show the full board; pause, then change its angle or position.'
        if c['span'][0]<120:return 'Move the board to the other SIDE of the picture.'
        if c['span'][1]<120:return 'Move the board HIGHER or LOWER in the picture.'
        if c['range_span']<.4:return 'Vary the distance: one view near 0.55 m, another near 1.05 m.'
        if c['tilt_span'][0]<30:return 'Tip one SIDE toward the camera, then the opposite side.'
        if c['tilt_span'][1]<30:return 'Tip the TOP toward the camera, then the BOTTOM.'
        if c['cells']<4:return 'Try a different area of the picture.'
        if c['condition']>12:return 'Use stronger side-to-side and top-to-bottom tilts.'
        remaining=max(0,24-len(self.views))
        if remaining:return f'Only {remaining} more view'+('s' if remaining!=1 else '')+' needed. Tip the TOP edge a little closer to the camera, then HOLD STILL for 1.4 seconds.'
        return 'Coverage complete — ready to fit and validate.'
    def update(self,obs,now):
        if self.phase not in ('capture','validation') or self.waiting:return super().update(obs,now)
        if now<self.until:
            self.feedback=f'{self.transition_message} ({math.ceil(self.until-now)} s)' if self.phase=='validation' else f'SAVED — change the view ({math.ceil(self.until-now)} s)';self.reset_hold();return None
        reason=obs.get('problem','')
        if not reason and obs.get('skew',999)>=80:reason='Waiting for a closer RGB/depth pair.'
        if not reason and not .45<=obs['distance']<=1.35:reason='Keep the board about 0.5–1.3 metres from the camera.'
        if not reason and self.phase=='capture':
            for v in self.views:
                if abs(v['distance']-obs['distance'])<.06 and np.linalg.norm(np.array(v['center'])-obs['center'])<35 and np.linalg.norm(np.array(v['tilt'])-obs['tilt'])<12:
                    reason='Already saved this view. '+self.suggestion();break
        if not reason and self.phase=='validation':
            # Broad ranges guarantee the independent six checks span >0.4 m.
            lo,hi=[(.50,.63),(.75,.90),(1.05,1.25)][self.index%3]
            if not lo<=obs['distance']<=hi:reason=f'Move the board into the {lo:.2f}–{hi:.2f} m range.'
            else:
                target=self.pose.tilt;axis=int(np.argmax(np.abs(target)));sign=1 if target[axis]>0 else -1
                if obs['tilt'][axis]*sign<15:
                    edge=(('LEFT' if sign>0 else 'RIGHT') if axis==0 else ('BOTTOM' if sign>0 else 'TOP'))
                    reason=f'Tip the {edge} edge shown on screen closer to the camera.'
        if reason:self.feedback=reason;self.cue={'kind':'find','message':reason};self.reset_hold();return None
        corners=obs['corners']
        if self.anchor is not None:
            common=set(corners)&set(self.anchor)
            shift=np.percentile([np.linalg.norm(corners[i]-self.anchor[i]) for i in common],90) if len(common)>=(10 if self.phase=='validation' else 18) else 999
            if shift>2:
                self.reset_hold();self.feedback='Board is still moving — stop completely for 1.4 seconds';self.cue={'kind':'moving','message':self.feedback};return None
        if self.hold_since is None:self.hold_since=now;self.anchor={i:p.copy() for i,p in corners.items()}
        self.progress=min(1.,(now-self.hold_since)/1.4);self.cue={'kind':'hold','message':'HOLD STILL'}
        self.feedback=f'HOLD STILL — saving in {max(0,1.4-(now-self.hold_since)):.1f} s'
        if self.progress>=1:
            tilt=np.array(obs['tilt'],float);normal=np.r_[np.tan(np.radians(tilt)),1.];normal/=np.linalg.norm(normal)
            self.pending_view=dict(center=np.mean(list(corners.values()),axis=0).tolist(),tilt=tilt.tolist(),normal=normal.tolist(),distance=float(obs['distance']))
            self.waiting=True;return 'save' if self.phase=='capture' else 'check'
        return None
    def captured(self,now):
        if self.phase!='capture':return super().captured(now)
        self.views.append(self.pending_view);self.pending_view=None;self.last_capture_at=now;self.index+=1;self.waiting=False;self.serial+=1;self.reset_hold();self.until=now+1.5
        if self.coverage()['ready']:
            self.phase='solving';self.feedback=f'{self.index} useful views saved. Fitting and checking calibration…';return 'solve'
        if self.index>=40:self.block('40 views saved, but coverage is still weak. Session retained for review.');return None
        self.feedback='SAVED — '+self.suggestion()
    def title(self):
        if self.phase=='capture':return f'AUTO CAPTURE • {self.index} views saved • 24 minimum'
        if self.phase=='validation':return f'INDEPENDENT DEPTH CHECK • {self.index+1} / 6'
        return super().title()
