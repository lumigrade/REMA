"""CPU-only hands-free calibration guidance. All thresholds are acquisition gates;
existing independent geometry/outline/timing qualification remains mandatory.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np
from .calibration import board, extract_board_plane

@dataclass(frozen=True)
class Pose:
    distance: float
    center: tuple
    tilt: tuple

    @property
    def instruction(self):
        x, y = self.tilt
        edge = ('RIGHT' if x > 0 else 'LEFT') if abs(x) > abs(y) else ('BOTTOM' if y > 0 else 'TOP')
        zone = 'CLOSE' if self.distance < .7 else 'MID' if self.distance < 1 else 'FAR'
        return f'{zone} {self.distance:.2f} m • Bring the {edge} edge toward the camera'

# Each distance includes all four image quadrants and both signs of both tilt axes.
# Every fourth view is held out by the solver, so training keeps both tilt signs.
PATTERN = [(-75,-95,28,0),(75,-95,-28,0),(-75,95,0,28),(75,95,0,-28),
           (-75,-95,0,-28),(75,-95,0,28),(75,95,-28,0),(-75,95,28,0)]
FIT_POSES = [Pose(z,(320+x,240+y),(tx,ty)) for z in (.60,.80,1.02) for x,y,tx,ty in PATTERN]
CHECK_POSES = [Pose(z,(320+x,240+y),tilt) for z,x,y,tilt in [
    (.56,-25,-20,(23,0)),(.80,25,20,(-23,0)),(1.06,-25,20,(0,23)),
    (.56,25,-20,(0,-23)),(.80,-25,-20,(0,23)),(1.06,25,20,(-23,0))]]

def rgb_outline(points, ids, square, min_corners=20):
    if len(ids) < min_corners:
        raise ValueError('Show the whole print; face it toward the camera and move slightly closer if it is too small')
    obj = board(square).getChessboardCorners()[ids,:2]
    if min_corners<20 and (np.ptp(obj[:,0])<4*square or np.ptp(obj[:,1])<3*square):
        raise ValueError('Need pattern corners spread across the board; face it more toward the camera')
    H, inliers = cv2.findHomography(obj, points, cv2.RANSAC, 2)
    if H is None or inliers.sum() < min_corners-2:
        raise ValueError('Pattern blurred — hold still with the print facing the camera')
    ox, oy = (.297-10*square)/2, (.210-7*square)/2
    sheet = np.float32([[-ox,-oy],[10*square+ox,-oy],[10*square+ox,7*square+oy],[-ox,7*square+oy]])
    outline = cv2.perspectiveTransform(sheet[None],H)[0]
    if not np.isfinite(outline).all() or not cv2.isContourConvex(outline):
        raise ValueError('Cannot locate the four board edges')
    if np.any(outline < [8,8]) or np.any(outline > [631,471]):
        raise ValueError('Keep ALL four board edges inside the picture')
    center = cv2.perspectiveTransform(np.float32([[[5*square,3.5*square]]]),H)[0,0]
    return center, outline

def auto_board_plane(depth, center, intrinsics, previous_seed=None):
    """Find a separated A4-sized depth patch near the RGB board, without a click.
    Pre-calibration correspondence is provisional. Reject merged torso/wall patches;
    independent final outline validation determines whether correspondence was right.
    """
    seeds = [center]
    if previous_seed is not None and np.linalg.norm(np.array(previous_seed)-center)<100:
        seeds.insert(0,previous_seed)
    seeds += [center+np.array(delta) for delta in [(-45,0),(45,0),(0,-35),(0,35)]]
    fx,fy,_,_ = intrinsics
    for seed in seeds:
        if not (5 < seed[0] < 635 and 5 < seed[1] < 475):
            continue
        try:
            plane = extract_board_plane(depth, seed, intrinsics)
            mask = plane['mask'].astype(np.uint8)
            contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours,key=cv2.contourArea)
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)
            rect_area = rect[1][0]*rect[1][1]
            z = float(np.median(plane['points'][:,2]))
            expected = fx*fy*.297*.210*plane['normal'][2]/(z*z)
            centroid = np.array(rect[0])
            # Plane region must look like a physical A4 rectangle, not a body/wall.
            if not (.55 < area/max(expected,1) < 1.40 and area/max(rect_area,1)>.77):
                continue
            if np.linalg.norm(centroid-center)>95 or plane['median_error_m']>.008:
                continue
            plane['range_m'] = z
            plane['center'] = centroid
            return plane
        except (ValueError,np.linalg.LinAlgError):
            pass
    raise ValueError('Hold the board 30 cm clear of your chest / wall; show its full depth surface')

def mirror_points(points, width=640):
    """Display transform only. Sensor pixels used for calibration stay untouched."""
    result=np.array(points,dtype=float,copy=True)
    result[...,0]=width-1-result[...,0]
    return result


def movement_cue(pose, obs):
    """One correction at a time, in the user's mirrored screen coordinates.
    Positive native normal.x means native RIGHT is nearer the camera. That edge
    appears LEFT in the mirror. Pushing it away from the user increases that tilt.
    """
    if obs.get('problem'):
        return dict(kind='find',message=obs['problem'])
    z=obs['distance'];dx,dy=np.array(obs['center'])-pose.center
    if abs(z-pose.distance)>.045:
        away=z>pose.distance
        return dict(kind='distance',away=bool(away),current=z,target=pose.distance,
                    message=('PUSH the WHOLE board away from you — CLOSER to camera' if away else
                             'PULL the WHOLE board TOWARD YOU — farther from camera'))
    if abs(dx)>22 or abs(dy)>12:
        horizontal=abs(dx)/22>abs(dy)/12
        direction=('RIGHT' if dx>0 else 'LEFT') if horizontal else ('UP' if dy>0 else 'DOWN')
        return dict(kind='position',direction=direction,delta=[float(dx),float(-dy)],
                    message=f'SLIDE the whole board {direction} — keep its angle for now')
    current=np.array(obs['tilt'],dtype=float);target=np.array(pose.tilt,dtype=float)
    error=target-current
    if np.max(np.abs(error))>7:
        axis=int(np.argmax(np.abs(error)))
        sign=1 if (target[axis] if target[axis] else current[axis])>0 else -1
        edge=(('LEFT' if sign>0 else 'RIGHT') if axis==0 else ('BOTTOM' if sign>0 else 'TOP'))
        away=bool(error[axis]*sign>0)
        return dict(kind='tilt',axis=axis,edge=edge,away=away,current=current.tolist(),target=target.tolist(),
                    message=(f'PUSH the ORANGE {edge} edge AWAY from you' if away else
                             f'PULL the ORANGE {edge} edge TOWARD YOU'))
    if obs.get('skew',999)>=80:
        return dict(kind='wait',message='Waiting for a closer RGB/depth frame pair')
    return dict(kind='hold',message='HOLD STILL')

class Guide:
    def __init__(self):
        self.phase='ready';self.index=0;self.until=0.;self.hold_since=None;self.anchor=None
        self.feedback='Measure the print, then press Start. All captures after that are automatic.'
        self.progress=0.;self.serial=0;self.deadline=0.;self.resume_phase=None
        self.cue={'kind':'find','message':'Show the printed board'};self.waiting=False;self.validation_attempt=0;self.transition_message="Move to the next pose"

    @property
    def pose(self):
        if self.phase in ('capture','validation'):
            return (FIT_POSES if self.phase=='capture' else CHECK_POSES)[self.index]
        return None

    def reset_hold(self):
        self.hold_since=None;self.anchor=None;self.progress=0.

    def start(self,now,index=0):
        self.phase='countdown';self.until=now+10;self.index=index;self.serial+=1;self.reset_hold()

    def pause(self,now):
        if self.phase=='paused':
            self.phase=self.resume_phase;self.until=now+3;self.serial+=1;self.transition_message='Resuming — get into position'
        elif self.phase in ('capture','validation','countdown','timing_ready'):
            self.resume_phase=self.phase;self.phase='paused';self.serial+=1;self.feedback='Paused. Press Resume when ready; your saved poses are retained.'
        self.reset_hold()

    def block(self,message):
        self.phase='blocked';self.feedback=message;self.waiting=False;self.reset_hold();self.serial+=1

    def update(self,obs,now):
        self.cue={'kind':'wait','message':''}
        if self.phase=='countdown':
            self.feedback=f'Pick up the board — starting in {max(0,math.ceil(self.until-now))} seconds'
            if now>=self.until:
                self.phase='capture';self.until=now;self.serial+=1
            return None
        if self.phase=='timing_ready':
            self.feedback=f'In {max(0,math.ceil(self.until-now))} s, move the board toward / away from the camera'
            if now>=self.until:
                self.phase='timing';self.deadline=now+12;self.serial+=1
                return 'timing'
            return None
        if self.phase=='timing':
            toward=int((now-(self.deadline-12))/2)%2==0
            self.feedback=('MOVE CLOSER toward 0.60 m' if toward else 'MOVE FARTHER toward 1.05 m')+f' • {max(0,math.ceil(self.deadline-now))} s remaining'
            return None
        if not self.pose or self.waiting:
            return None
        if now<self.until:
            self.feedback=f'{self.transition_message} ({math.ceil(self.until-now)} s)'
            self.reset_hold();return None
        pose=self.pose
        self.cue=movement_cue(pose,obs)
        reason=self.cue['message'] if self.cue['kind']!='hold' else ''
        if reason:
            self.feedback=reason;self.reset_hold();return None
        corners=obs['corners']
        if self.anchor is not None:
            common=set(corners)&set(self.anchor)
            displacement=np.percentile([np.linalg.norm(corners[i]-self.anchor[i]) for i in common],90) if len(common)>=18 else 999
            if displacement>2:
                self.reset_hold()
        if self.hold_since is None:
            self.hold_since=now;self.anchor={i:p.copy() for i,p in corners.items()}
        self.progress=min(1.,(now-self.hold_since)/1.4)
        self.feedback=f'HOLD STILL — capturing in {max(0,1.4-(now-self.hold_since)):.1f} s'
        if self.progress>=1:
            self.waiting=True
            return 'save' if self.phase=='capture' else 'check'
        return None

    def captured(self,now):
        self.waiting=False;self.index+=1;self.serial+=1;self.reset_hold();self.until=now+2.5;self.transition_message='SAVED ✓ — move to the next pose'
        if self.phase=='capture' and self.index==len(FIT_POSES):
            self.phase='solving';self.feedback='All 24 poses saved. Fitting calibration — please wait.'
            return 'solve'
        if self.phase=='validation' and self.index==len(CHECK_POSES):
            return 'validation_done'
        return None

    def solved(self,passed,now,message=''):
        if not passed:
            self.block('Geometry fit was not accepted. '+message+' Raw poses are saved.');return
        self.phase='validation';self.index=0;self.until=now+6;self.reset_hold();self.serial+=1
        self.feedback='Fit passed. Next: six NEW poses to check depth alignment.';self.transition_message=self.feedback

    def validated(self,passed,now,message=''):
        if not passed:
            self.block('Depth validation was not accepted. '+message);return
        self.phase='timing_ready';self.until=now+8;self.serial+=1;self.reset_hold()

    def title(self):
        if self.pose:
            n=24 if self.phase=='capture' else 6
            stage='CALIBRATION' if self.phase=='capture' else 'DEPTH CHECK'
            return f'{stage} {self.index+1} / {n} • BOARD DISTANCE {self.pose.distance:.2f} m'
        return {'ready':'HANDS-FREE CALIBRATION','countdown':'GET READY','solving':'CALCULATING',
                'timing_ready':'NEXT: MOVE THE BOARD IN / OUT','timing':'TIMING — KEEP THE PRINT FACING THE CAMERA',
                'paused':'PAUSED — put the board down if needed','blocked':'CALIBRATION NEEDS ATTENTION',
                'done':'FINISHED — you can put the board down'}.get(self.phase,'PLEASE WAIT')
