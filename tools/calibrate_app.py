#!/usr/bin/env python3
"""One-person, CPU-only measured RGB/depth calibration wizard."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', QT_XCB_GL_INTEGRATION='none',
                  LIBGL_ALWAYS_SOFTWARE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import sys, time, json, threading, fcntl, shutil, signal, subprocess, queue, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
cv2.setNumThreads(2)
import numpy as np
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QCheckBox, QDoubleSpinBox, QProgressBar)
from PySide6.QtCore import QTimer, Qt, QRect
from PySide6.QtGui import QPainter,QColor,QImage
from live_app import Views, label
from rema.live_camera import Cameras
from rema.live_core import depth_preview
from rema.calibration import (ROOT, detect, board, solve_planes, validate_depth,
                              depth_validation_summary, Projection, estimate_arrival_offset)
from rema.calibration_easy import EasyGuide,robust_detect,print_quality
from rema.calibration_rgb import prepare_rgb
from rema.calibration_guide import Guide, rgb_outline, auto_board_plane
from rema.calibration_overlay import filled_overlay,target_polygon,estimate_guide_K

class CalibrationWorker:
    def __init__(self,resume_folder=None,easy=False,new_print=False,square_mm=None,target_layout_pending=False):
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.commands = queue.Queue()
        self.cameras = None
        profile_path=Path(str(_P.DATA)+'/config/calibration_rgb.json')
        self.rgb_profile=json.loads(profile_path.read_text()) if profile_path.exists() else None
        self.target_settings = {}
        settings_path=ROOT/'target_settings.json'
        if settings_path.exists():
            self.target_settings=json.loads(settings_path.read_text())
        self.easy=easy;self.target_layout_pending=target_layout_pending
        if new_print:self.target_settings={'square_mm':25.,'measurement_confirmed':False}
        if square_mm is not None:self.target_settings={'square_mm':float(square_mm),'measurement_confirmed':True}
        if resume_folder and new_print:raise ValueError('Choose resume OR new print, not both')
        self.square = float(self.target_settings.get('square_mm',25))/1000
        self.result = None
        self.error = ''
        self.guide = EasyGuide() if easy else Guide()
        if self.target_layout_pending:self.guide.feedback=f'{self.square*1000:.2f} mm square confirmed. Specify squares across/down and pattern type to enable capture.'
        self.samples = []
        self.checks = []
        self.config = None
        self.retry_available = False
        self.projection = None
        self.seed = None
        self.timing = None
        self.timing_result = None
        self.folder = Path(resume_folder).resolve() if resume_folder else ROOT / ('session_' + time.strftime('%Y%m%d_%H%M%S') + ('_easy' if easy else '_guided'))
        if resume_folder:
            if self.folder.parent!=ROOT.resolve() or not self.folder.is_dir():
                raise ValueError('Resume session must be an existing REMA calibration folder')
            for index,path in enumerate(sorted(self.folder.glob('pose_*.npz'))):
                with np.load(path,allow_pickle=False) as data:
                    sample={k:data[k].copy() for k in data.files if k not in ('rgb_image','sensor_raw','rgb_native','rgb_profile_json')}
                if int(sample.get('guided_pose',-1))!=index or abs(float(sample['square_metres'])-self.square)>1e-7:
                    raise ValueError('Saved pose sequence / square size does not match this target')
                self.samples.append(sample)
            if easy:
                if not self.folder.name.endswith('_easy'):raise ValueError('Resume easy mode only from an easy-capture session')
                if len(self.samples)>=40:raise ValueError('Session reached40views; review its saved data before continuing')
                self.guide.restore(self.samples)
            elif len(self.samples)>=24:raise ValueError('Acquisition is complete; retain session for fitting')
        else:
            self.folder.mkdir()
        self.guide_K=estimate_guide_K(self.samples)
        self.observation={};self.preflight={};self.last_raw_diagnostic=0
        self.last_diagnostic = 0
        self.check_index = 0
        if resume_folder and (self.folder/'candidate.json').exists():
            config=json.loads((self.folder/'candidate.json').read_text())
            if len(self.samples)<24 or not config.get('stereo_pass') or abs(config['square_metres']-self.square)>1e-7:
                raise ValueError('Saved fit is not accepted for this target; retain session for review')
            if (self.folder/'validated.json').exists():
                raise ValueError('This calibration is already complete')
            checks_path=self.folder/'depth_checks.json'
            self.checks=json.loads(checks_path.read_text()) if checks_path.exists() else []
            failed_paths=sorted(self.folder.glob('depth_checks_set_*_FAILED.json'))
            evidence={c['raw_capture_index']:c for c in self.checks}
            for path in failed_paths:
                failed=json.loads(path.read_text())
                if not failed or all(c.get('passed') for c in failed):
                    raise ValueError('Invalid failed-set archive; retain for review')
                evidence.update({c['raw_capture_index']:c for c in failed})
            raw_checks=sorted(self.folder.glob('depth_check_*.npz'))
            if len(raw_checks)!=len(evidence) or set(evidence)!=set(range(len(raw_checks))):
                raise ValueError('Incomplete depth-check write; retain raw evidence for review')
            last_attempt=0
            for index,path in enumerate(raw_checks):
                with np.load(path,allow_pickle=False) as raw:
                    if int(path.stem.split('_')[-1])!=index:
                        raise ValueError('Depth-check sequence mismatch; saved evidence retained')
                    last_attempt=max(last_attempt,int(raw['validation_set']))
            if len(self.checks)>6:raise ValueError('Invalid validation set length; retained for review')
            self.check_index=len(raw_checks);self.guide.validation_attempt=last_attempt
            if failed_paths or any(not c.get('passed') for c in self.checks):
                last=evidence[max(evidence)]
                message=last.get('error') or last.get('failure_reason') or f"Recorded outline p95 {last.get('edge_p95_px',999):.1f} px (old cropped-outline validator). Box/hand edges differ from A4; inspect depth before retrying."
                if last.get('plane_pass') and .95<last.get('surface_to_expected_area_ratio',0)<1.05:
                    message=f"Plane passed; RGB/depth outline alignment needs refinement: median {last['edge_median_px']:.1f} px, p95 {last['edge_p95_px']:.1f} px. Capture the full near/mid/far check set."
                self.guide.block(message)
                audit_path=self.folder/'outline_recheck_v2.json'
                if audit_path.exists():
                    audit=json.loads(audit_path.read_text());entry=next((e for e in audit['checks'] if e['raw_capture_index']==max(evidence)),None)
                    if entry:
                        message=entry.get('error') or entry.get('failure_reason') or 'Recheck retained; six new depth checks still required'
                        self.guide.block(('Plane PASS. ' if entry.get('plane_pass') else '')+message)
                self.retry_available=True
            # Resume an incomplete full-set workflow, including recorded failures.
            # Earlier stop-on-first-failure attempts still open for explicit retry.
            event_path=self.folder/'events.jsonl'
            prior_events=[json.loads(line) for line in event_path.read_text().splitlines() if line.strip()] if event_path.exists() else []
            starts=[e for e in prior_events if e.get('kind')=='manual_validation_retry']
            continuation=starts[-1] if starts else {}
            if continuation.get('workflow')=='complete_six_checks_v1' and continuation.get('validation_set')==last_attempt and 0<len(self.checks)<6:
                for position,check in enumerate(self.checks):
                    with np.load(self.folder/f"depth_check_{check['raw_capture_index']:03}.npz",allow_pickle=False) as raw:
                        if int(raw['validation_set'])!=last_attempt or int(raw['guided_pose'])!=position:
                            raise ValueError('Current check-set sequence mismatch; evidence retained')
                self.guide.phase='ready';self.retry_available=False
            self.config=config;self.projection=Projection(config)
        self.refinement_report = None
        if self.config:
            for report_path in sorted(self.folder.glob('refinement_*/report.json'), reverse=True):
                report=json.loads(report_path.read_text())
                if report.get('source_candidate_sha256')==hashlib.sha256((self.folder/'candidate.json').read_bytes()).hexdigest():
                    self.refinement_report=report
                    break
        if len(self.checks)==6 and not all(c.get('passed') for c in self.checks):
            self.guide.block('Capture complete. Alignment has not passed; calibration is inactive. All results are saved. Put the board down.')
        self.events = []
        self.snapshot = {'title': self.guide.title(), 'feedback': self.guide.feedback,
                         'phase': 'ready', 'progress': 0, 'serial': 0}

    def retry_validation(self,now):
        if not self.retry_available or not self.config:return
        # Keep all previous raw captures and complete attempt results before reset.
        archive=self.folder/f'depth_checks_set_{self.guide.validation_attempt}_FAILED.json'
        if not archive.exists():archive.write_text(json.dumps(self.checks,indent=2))
        self.guide.validation_attempt+=1;self.checks=[]
        (self.folder/'depth_checks.json').write_text('[]')
        self.retry_available=False;self.begin_session(now)
        self.record_event('manual_validation_retry',validation_set=self.guide.validation_attempt,workflow='complete_six_checks_v1')

    def begin_session(self,now):
        if self.config:
            self.guide.solved(True,now)
            self.guide.index=len(self.checks);self.guide.until=now+10
            self.guide.transition_message='Fit retained. Pick up the A4 board for the remaining depth checks'
        else:self.guide.start(now,len(self.samples))

    def start(self):
        self.thread.start()

    def record_event(self, kind, **data):
        event = dict(kind=kind, host_monotonic=time.monotonic(), **data)
        self.events.append(event)
        with (self.folder/'events.jsonl').open('a') as f:
            f.write(json.dumps(event) + '\n')

    def publish(self, left=None, right=None, detail=''):
        g = self.guide
        feedback=g.feedback
        if not self.easy and g.pose and g.cue.get('kind') in ('distance','position','tilt'):
            feedback='Match all four board edges to the coloured shape, then hold still.'
        self.snapshot = dict(title=g.title(), feedback=self.error or feedback,
                             phase=g.phase, progress=g.progress, serial=g.serial,
                             samples=len(self.samples), checks=sum(bool(c.get('passed')) for c in self.checks), checks_recorded=len(self.checks), retry_available=self.retry_available, preview='native_depth', detail=detail, mirror_view=True, cue=g.cue.get('kind'), overlay_opacity=.40, overlay_colors='red-green-blue-white', acquisition_observation=self.observation)
        if self.easy:self.snapshot.update(easy_capture=True,coverage=self.guide.coverage(),next_hint=self.guide.suggestion(),overlay_opacity=0,preflight=self.preflight,dashboard=self.guide.dashboard(time.monotonic()))
        if left is not None:
            self.result = (left, right)

    def save_pose(self, candidate, rgb, sensor):
        for previous in self.samples:
            ids = np.intersect1d(previous['ids'], candidate['ids'])
            if len(ids) >= 18:
                a = np.array([previous['rgb'][np.where(previous['ids']==i)[0][0]] for i in ids])
                b = np.array([candidate['rgb'][np.where(candidate['ids']==i)[0][0]] for i in ids])
                if np.mean(np.linalg.norm(a-b,axis=1)) < 8:
                    self.guide.waiting=False; self.guide.reset_hold()
                    self.guide.feedback='View already saved — change position, distance or tilt' if self.easy else 'Pose too similar — adjust the board inside the blue circle'
                    return
        sample = dict(candidate, rgb_host_arrival_ns=rgb[1], sensor_host_arrival_ns=sensor[1],
                      sensor_timestamp_us=sensor[3], sensor_frame_id=sensor[0],
                      rgb_frame_id=rgb[0], sensor_kind='depth_plane', square_metres=self.square,
                      guided_pose=self.guide.index)
        np.savez(self.folder/f'pose_{len(self.samples):03}.npz', rgb_image=rgb[2], sensor_raw=sensor[2], **sample, **(dict(rgb_native=rgb[3],rgb_profile_json=json.dumps(self.rgb_profile)) if len(rgb)>3 else {}))
        self.samples.append(sample)
        self.record_event('pose_saved', index=len(self.samples))
        if self.guide.captured(time.monotonic()) == 'solve':
            self.publish()
            try:
                config=solve_planes(self.samples,self.cameras.depth_intrinsics,self.square)
                config['session']=str(self.folder)
                self.config=config
                (self.folder/'candidate.json').write_text(json.dumps(config,indent=2))
                self.guide.solved(config['stereo_pass'],time.monotonic(),
                    f"RGB p95 {config['heldout_rgb_reprojection_p95_px']:.2f} px; plane p95 {config['heldout_plane_p95_m']*1000:.1f} mm.")
                if config['stereo_pass']:
                    self.projection=Projection(config)
                self.record_event('geometry_fit',passed=config['stereo_pass'])
            except Exception as e:
                self.record_event('geometry_fit_error',error=str(e))
                self.guide.solved(False,time.monotonic(),str(e))

    def save_check(self, rgb, sensor, detected=None):
        index=self.check_index;self.check_index+=1
        extra={}
        if len(rgb)>3:
            extra=dict(rgb_native=rgb[3],rgb_profile_json=json.dumps(self.rgb_profile),rgb_points=detected[0],rgb_ids=detected[1])
        np.savez(self.folder/f'depth_check_{index:03}.npz',rgb=rgb[2],depth=sensor[2],**extra,
                 rgb_host_arrival_ns=rgb[1],depth_host_arrival_ns=sensor[1],
                 rgb_frame_id=rgb[0],depth_frame_id=sensor[0],depth_sensor_timestamp_us=sensor[3],
                 validation_set=self.guide.validation_attempt,guided_pose=self.guide.index)
        try:
            check=validate_depth(self.config,rgb[2],sensor[2],detected=detected)
        except Exception as e:
            check={'passed':False,'error':str(e)}
        check['raw_capture_index']=index
        self.checks.append(check)
        (self.folder/'depth_checks.json').write_text(json.dumps(self.checks,indent=2))
        self.record_event('depth_check',**check)
        # Record the complete independent set, including failures. A failed
        # metric is evidence for diagnosis, not a reason to loop forever on pose 1.
        failed=any(not c.get('passed') for c in self.checks)
        if failed:
            (self.folder/f'depth_checks_set_{self.guide.validation_attempt}_FAILED.json').write_text(json.dumps(self.checks,indent=2))
        if self.guide.captured(time.monotonic()) == 'validation_done':
            good,reason=depth_validation_summary(self.checks)
            summary=dict(passed=good,reason=reason,recorded=len(self.checks),passed_checks=sum(bool(c.get('passed')) for c in self.checks),checks=self.checks)
            (self.folder/f'validation_summary_set_{self.guide.validation_attempt}.json').write_text(json.dumps(summary,indent=2))
            self.record_event('validation_complete',passed=good,recorded=len(self.checks),passed_checks=summary['passed_checks'])
            if good:self.guide.validated(True,time.monotonic(),reason)
            else:
                self.retry_available=True
                self.guide.block(f"All 6 checks saved — {summary['passed_checks']}/6 passed. Put the board down. Alignment requires review/refinement; calibration not activated.")
        else:
            verdict='PASS' if check.get('passed') else 'alignment needs review'
            self.guide.transition_message=f'CHECK {len(self.checks)} SAVED ({verdict}) — continue to the NEXT pose'
            self.guide.feedback=self.guide.transition_message

    def timing_frame(self, points, ids, aligned, rgb, sensor):
        if len(ids)>=20:
            ok,r,t=cv2.solvePnP(board(self.square).getChessboardCorners()[ids],points,
                              np.array(self.config['K_rgb']),np.array(self.config['dist_rgb']))
            if ok:
                mask=np.zeros((480,640),np.uint8)
                cv2.fillConvexPoly(mask,cv2.convexHull(points).astype(np.int32),1)
                mask=cv2.erode(mask,np.ones((13,13),np.uint8))
                values=aligned[(mask>0)&(aligned>.3)]
                if len(values)>100:
                    self.timing.append(dict(rgb_time=rgb[1]/1e9,depth_time=sensor[1]/1e9,
                                            rgb_range=float(t[2,0]),depth_range=float(np.median(values))))
        if time.monotonic()>=self.guide.deadline:
            (self.folder/'timing_samples.json').write_text(json.dumps(self.timing,indent=2))
            try:
                self.timing_result=estimate_arrival_offset(self.timing)
            except Exception as e:
                self.timing_result={'passed':False,'error':str(e),'exposure_sync_qualified':False}
            self.timing=None
            (self.folder/'timing_result.json').write_text(json.dumps(self.timing_result,indent=2))
            self.record_event('timing_result',**self.timing_result)
            self.activate()

    def activate(self):
        good,reason=depth_validation_summary(self.checks)
        if not self.config or not self.config['stereo_pass'] or not good:
            self.guide.block('Calibration NOT activated: '+reason);return
        if self.rgb_profile:self.config['validation_rgb_profile']=self.rgb_profile
        self.config.update(qualified=True,depth_validation_pass=True,depth_checks=self.checks,
                           calibration_id=self.folder.name)
        if self.timing_result and self.timing_result['passed']:
            self.config['timing_proxy']=self.timing_result
        tmp=ROOT/'active.tmp';tmp.write_text(json.dumps(self.config,indent=2));tmp.replace(ROOT/'active.json')
        (self.folder/'validated.json').write_text(json.dumps(self.config,indent=2))
        self.guide.phase='done';self.guide.serial+=1
        timing='Arrival timing check passed.' if self.timing_result and self.timing_result['passed'] else 'Timing sweep did not pass; no timing correction applied.'
        self.guide.feedback='Geometry validated and activated. '+timing+' Hardware exposure sync remains unqualified.'
        self.record_event('activated',timing_pass=bool(self.timing_result and self.timing_result['passed']))

    def run(self):
        try:
            self.cameras=Cameras('depth',rgb_mode=self.rgb_profile);self.cameras.start()
            last_frame=-1;last_pair=time.monotonic()
            while not self.stop.is_set():
                try:
                    command=self.commands.get_nowait()
                    if command=='start' and self.guide.phase=='ready' and not self.target_layout_pending:
                        self.record_event('start',square_metres=self.square,measured_square_confirmed=True)
                        self.begin_session(time.monotonic())
                    elif command=='retry' and self.guide.phase=='blocked':self.retry_validation(time.monotonic())
                    elif command=='pause':self.guide.pause(time.monotonic())
                except queue.Empty:pass
                if self.cameras.error:raise RuntimeError(self.cameras.error)
                pair=self.cameras.pair()
                if not pair or pair[0][0]==last_frame:
                    if time.monotonic()-last_pair>3:raise RuntimeError('Camera feed stopped. Saved poses retained; close and reopen the wizard.')
                    time.sleep(.02);continue
                last_pair=time.monotonic();rgb,sensor=pair;last_frame=rgb[0]
                if time.monotonic_ns()-min(rgb[1],sensor[1])>2e9:raise RuntimeError('Camera feed stale')
                raw=rgb[2];skew=abs(rgb[1]-sensor[1])/1e6
                native=None
                if self.rgb_profile:
                    native=raw;raw,points,ids=prepare_rgb(native,self.rgb_profile,self.square);rgb=(rgb[0],rgb[1],raw,native)
                else:points,ids=(np.empty((0,2),np.float32),np.empty(0,np.int32)) if self.target_layout_pending else (robust_detect if self.easy else detect)(raw,self.square)
                left=raw.copy();right=depth_preview(sensor[2],max_range=2);candidate=None;outline=None;center=None
                observation={'problem':'Show the whole printed board to the camera'}
                valid_depth=int(np.count_nonzero(sensor[2]>.3))
                detail=f'{len(ids)} RGB corners'+(' • native HD uncompressed' if self.rgb_profile else '')+f' • depth valid {valid_depth/sensor[2].size:.0%} • host arrival skew {skew:.0f} ms • CPU only'
                if len(ids):cv2.aruco.drawDetectedCornersCharuco(left,points[:,None],ids[:,None])
                aligned=None
                if self.projection:
                    aligned=self.projection.align(sensor[2])
                if self.guide.phase not in ('solving','timing','done'):
                    try:
                        center,outline=rgb_outline(points,ids,self.square,min_corners=12 if self.config else 20)
                        if not self.cameras.depth_intrinsics:raise ValueError('Waiting for depth camera intrinsics')
                        plane=auto_board_plane(sensor[2],center,self.cameras.depth_intrinsics,self.seed)
                        self.seed=plane['seed']
                        normal=plane['normal']
                        tilt=np.degrees(np.arctan2(normal[:2],normal[2]))
                        contours,_=cv2.findContours(plane['mask'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(right,contours,-1,(40,255,40),2)
                        cv2.circle(right,tuple(map(int,self.seed)),5,(255,255,255),2)
                        cv2.polylines(left,[outline.astype(np.int32)],True,(80,210,100),2)
                        observation=dict(distance=plane['range_m'],center=center,tilt=tilt,
                                         corners={int(i):p for i,p in zip(ids,points)},skew=skew)
                        candidate=dict(obj=board(self.square).getChessboardCorners()[ids],rgb=points,ids=ids,
                                       plane_normal=normal,plane_offset=plane['offset'],plane_points=plane['points'],
                                       depth_intrinsics=np.array(self.cameras.depth_intrinsics))
                        detail=f'Board {plane["range_m"]:.2f} m • tilt H {tilt[0]:+.0f}° / V {tilt[1]:+.0f}° • '+detail
                    except (ValueError,cv2.error,np.linalg.LinAlgError) as e:
                        observation={'problem':str(e)}
                        self.seed=None
                quality=print_quality(raw,points,ids,self.square) if self.easy else None
                if quality:
                    detail+=f' | print contrast {quality["contrast"]:.0f} / reflection on dark areas {quality["washed_black_percent"]:.0f}%'
                    if quality['contrast']<35:observation={'problem':'Print contrast too low — reduce reflection with a small tilt or softer side light'}
                if self.target_layout_pending:observation={'problem':'50mm square confirmed; waiting for board layout, not testing the old target'}
                elif self.easy and len(ids)<(12 if self.config else 20):observation={'problem':f'Pattern: {len(ids)}/{12 if self.config else 20} corners. At the far pose, aim near 1.05 m; face the print toward the camera and hold still.'}
                if valid_depth<100:
                    observation={'problem':'No usable depth surface — capture is paused; check the depth status below'}
                self.preflight=dict(corners=len(ids),pattern_ok=len(ids)>=(12 if self.config else 20),depth_board_ok=candidate is not None,quality=quality)
                self.observation={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in observation.items() if k!='corners'}
                pose=self.guide.pose
                if pose and not self.easy and not observation.get('problem'):
                    difference=np.array(observation['center'])-pose.center
                    tilt_error=np.array(observation['tilt'])-pose.tilt
                    self.observation['target_distance_m']=pose.distance
                    self.observation['target_tilt_degrees']=list(pose.tilt)
                    self.observation['centre_error_pixels']=difference.tolist()
                    self.observation['tilt_error_degrees']=tilt_error.tolist()
                    detail+=f' | errors: range {abs(observation["distance"]-pose.distance)*100:.1f} cm / XY {difference[0]:+.0f},{difference[1]:+.0f} px / tilt {tilt_error[0]:+.0f},{tilt_error[1]:+.0f} deg'
                if time.monotonic()-self.last_raw_diagnostic>3:
                    self.last_raw_diagnostic=time.monotonic()
                    diagnostic=self.folder/'diagnostics';diagnostic.mkdir(exist_ok=True)
                    # Three rolling local raw pairs retain rejected observations for
                    # diagnosis without growing a continuous private video archive.
                    slot=int(time.monotonic()/3)%3
                    np.savez(diagnostic/f'latest_{slot}.npz',rgb=raw,depth=sensor[2],rgb_points=points,rgb_ids=ids,
                             **(dict(rgb_native=native,rgb_profile_json=json.dumps(self.rgb_profile)) if native is not None else {}),
                             rgb_frame_id=rgb[0],depth_frame_id=sensor[0],rgb_arrival_ns=rgb[1],depth_arrival_ns=sensor[1],
                             observation_json=json.dumps(self.observation))
                action=self.guide.update(observation,time.monotonic())
                if action=='save' and candidate is not None:self.save_pose(candidate,rgb,sensor)
                elif action=='check':self.save_check(rgb,sensor,detected=(points,ids))
                elif action=='timing':self.timing=[]
                if self.timing is not None:
                    self.timing_frame(points,ids,aligned,rgb,sensor)
                # The target stays at the REQUESTED pose, never follows the board.
                pose=self.guide.pose
                if pose is None and self.guide.phase in ('ready','countdown'):
                    from rema.calibration_guide import FIT_POSES
                    pose=FIT_POSES[min(len(self.samples),23)]
                polygon=None
                if pose and not self.easy:
                    K=np.array(self.config['K_rgb']) if self.config and self.config['stereo_pass'] else self.guide_K
                    distortion=self.config['dist_rgb'] if self.config and self.config['stereo_pass'] else None
                    rotation=self.config['R_depth_to_rgb'] if self.config and self.config['stereo_pass'] else None
                    polygon=target_polygon(pose,K,distortion,rotation)
                left=cv2.flip(left,1) if self.easy else filled_overlay(raw,polygon,time.monotonic(),mirror=True)
                if self.config and len(ids)>=12:
                    # Expected paper outline projected into NATIVE depth coordinates.
                    # This is diagnostic only; it does not alter capture or qualification.
                    try:
                        ok,r,t=cv2.solvePnP(board(self.square).getChessboardCorners()[ids],points,np.array(self.config['K_rgb']),np.array(self.config['dist_rgb']))
                        if ok:
                            ox=(.297-10*self.square)/2;oy=(.210-7*self.square)/2
                            sheet=np.float32([[-ox,-oy,0],[.297-ox,-oy,0],[.297-ox,.210-oy,0],[-ox,.210-oy,0]])
                            xyz=sheet@cv2.Rodrigues(r)[0].T+t.ravel()
                            xyz=(xyz-np.array(self.config['t_depth_to_rgb_metres']))@np.array(self.config['R_depth_to_rgb'])
                            if np.all(xyz[:,2]>.1):
                                uv,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),np.array(self.config['K_depth']),np.array(self.config['dist_depth']))
                                cv2.polylines(right,[np.rint(uv).astype(np.int32)],True,(255,255,0),2)
                    except (cv2.error,ValueError):pass
                right=cv2.flip(right,1)
                label(right,f'LIVE DEPTH | frame {sensor[0]} | valid {valid_depth/sensor[2].size:.0%}',(12,26),.6)
                label(right,'Distance colours: 0-2 m | black = no depth',(12,462),.55)
                self.publish(left,right,detail)
                if time.monotonic()-self.last_diagnostic>2:
                    self.last_diagnostic=time.monotonic()
                    (self.folder/'status.json').write_text(json.dumps(dict(self.snapshot,rgb_corners=len(ids),
                         depth_intrinsics=self.cameras.depth_intrinsics,depth_valid_pixels=valid_depth,depth_max_metres=float(sensor[2].max()),depth_frame_id=int(sensor[0]),session=str(self.folder)),indent=2))
                time.sleep(.045)
        except Exception as e:
            self.error=str(e);self.guide.block(self.error);self.record_event('error',error=str(e));self.publish()
        finally:
            if self.cameras:self.cameras.close()

    def close(self):
        self.stop.set();self.thread.join(8)

class OverlayView(QWidget):
    def __init__(self):
        super().__init__();self.frames=None;self.show_depth=True;self.setMinimumSize(640,300)
    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rect(),QColor('#14171b'))
        if not self.frames:return
        frame=self.frames[1] if self.show_depth and self.frames[1] is not None else self.frames[0];w=min(self.width(),int(self.height()*4/3));h=w*3//4
        rect=QRect((self.width()-w)//2,(self.height()-h)//2,w,h)
        img=QImage(frame.data,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_BGR888)
        p.drawImage(rect,img)

class Wizard(QWidget):
    def __init__(self,worker=None,start_camera=True):
        super().__init__()
        self.worker=worker or CalibrationWorker()
        self.setWindowTitle('REMA • Easy automatic calibration' if self.worker.easy else 'REMA • Calibration • Filled target overlay')
        self.resize(1440,1160)
        self.setStyleSheet('QWidget{background:#202328;color:#e4e7eb;font-size:15px} QPushButton,QDoubleSpinBox{background:#343a43;padding:10px;border-radius:4px} QPushButton:disabled{color:#777} QProgressBar{border:1px solid #46505b;height:24px;text-align:center} QProgressBar::chunk{background:#48bc86}')
        layout=QVBoxLayout(self)
        self.heading=QLabel('HANDS-FREE CALIBRATION');self.heading.setWordWrap(True)
        self.heading.setStyleSheet('font-size:28px;font-weight:700;color:#a8d7ff')
        layout.addWidget(self.heading)
        self.start_button=QPushButton('START HANDS-FREE CALIBRATION • 10-second countdown')
        self.start_button.setMinimumHeight(64)
        self.start_button.setStyleSheet('QPushButton{background:#258458;color:white;font-size:23px;font-weight:700;border:2px solid #75dbac} QPushButton:disabled{background:#343a43;color:#a0adb8;border:1px solid #56616e}')
        self.start_button.clicked.connect(self.begin)
        layout.addWidget(self.start_button)
        self.prompt=QLabel('Measure the print, then press Start.');self.prompt.setWordWrap(True)
        self.prompt.setMinimumHeight(48);self.prompt.setStyleSheet('font-size:25px;font-weight:600')
        layout.addWidget(self.prompt)
        self.stage_label=QLabel('1  CAPTURE VIEWS   →   2  FIT   →   3  CHECK DEPTH (6)   →   4  TIMING');self.stage_label.setStyleSheet('font-size:17px;color:#a8d7ff');layout.addWidget(self.stage_label)
        self.capture_count=QLabel('');self.capture_count.setStyleSheet('font-size:27px;font-weight:700');layout.addWidget(self.capture_count)
        self.capture_progress=QProgressBar();self.capture_progress.setRange(0,24);self.capture_progress.setFormat('%v / 24 views saved');layout.addWidget(self.capture_progress)
        self.quality_status=QLabel('');self.quality_status.setWordWrap(True);self.quality_status.setStyleSheet('font-size:19px;color:#ffcb80');layout.addWidget(self.quality_status)
        self.refinement_status=QLabel('');self.refinement_status.setWordWrap(True);self.refinement_status.setStyleSheet('font-size:19px;color:#a8d7ff');layout.addWidget(self.refinement_status)
        self.checklist=QLabel('');self.checklist.setWordWrap(True);self.checklist.setStyleSheet('font-size:20px;color:#70e4a7');layout.addWidget(self.checklist)
        self.next_action=QLabel('');self.next_action.setWordWrap(True);self.next_action.setStyleSheet('font-size:23px;font-weight:600;color:#a8d7ff');layout.addWidget(self.next_action)
        self.bar=QProgressBar();self.bar.setRange(0,100);self.bar.setFormat('Hold steady to capture automatically');layout.addWidget(self.bar)
        self.view_button=QPushButton('SHOWING LIVE DEPTH • click for RGB');self.view_button.clicked.connect(self.toggle_view);layout.addWidget(self.view_button)
        self.depth_legend=QLabel('DEPTH VIEW • Green = detected surface • Cyan = expected A4 outline from saved fit • Mirrored');self.depth_legend.setWordWrap(True);layout.addWidget(self.depth_legend)
        self.views=OverlayView();layout.addWidget(self.views,1)
        self.detail=QLabel('Connecting to Astra RGB + depth…');self.detail.setWordWrap(True);layout.addWidget(self.detail)
        self.preflight_label=QLabel('Expected print: REMA 10 × 7 ChArUco target');self.preflight_label.setStyleSheet('font-size:19px;font-weight:600');self.preflight_label.setWordWrap(True);layout.addWidget(self.preflight_label)
        instructions=QLabel('Match the A4 board to the filled target: red → green → blue → white, 40% opacity. Mirror view. Captures are automatic when the measured pose is steady.\nKeep the rigid A4 board clear of your body. Initial target sizing is provisional; independent depth checks are required before activation.')
        if self.worker.easy:instructions.setText('Move naturally, then HOLD STILL: useful views save automatically with a beep. Green dots/outline show what the camera detected.\nUse a flat rigid A4 backing; hold it clear of your body or table so depth sees its edges. Tilt slightly or soften side lighting if reflections hide the pattern.')
        instructions.setWordWrap(True);layout.addWidget(instructions);self.instructions=instructions
        self.coverage_label=QLabel('');self.coverage_label.setWordWrap(True);self.coverage_label.setVisible(False);layout.addWidget(self.coverage_label)
        row=QHBoxLayout()
        self.measured=QCheckBox('Square size measured');self.measured.setChecked(bool(self.worker.target_settings.get('measurement_confirmed')));row.addWidget(self.measured)
        row.addWidget(QLabel('Square (mm)'))
        self.square=QDoubleSpinBox();self.square.setRange(1,100);self.square.setDecimals(2);self.square.setValue(self.worker.square*1000)
        self.square.valueChanged.connect(lambda v:setattr(self.worker,'square',v/1000));row.addWidget(self.square)
        self.voice=QCheckBox('Spoken cues');self.voice.setChecked(False);row.addWidget(self.voice)
        layout.addLayout(row)
        row=QHBoxLayout()
        self.start_button.setEnabled(True)
        self.start_button.setToolTip('Start remains visible. If setup is incomplete, pressing it explains what is needed.')
        self.retry_button=QPushButton('START NEW 6-CHECK SET • 10-second countdown');self.retry_button.clicked.connect(lambda:self.worker.commands.put('retry'));row.addWidget(self.retry_button)
        self.pause_button=QPushButton('Pause');self.pause_button.clicked.connect(lambda:self.worker.commands.put('pause'));row.addWidget(self.pause_button)
        self.live_button=QPushButton('Return to REMA Live');self.live_button.clicked.connect(self.return_live);row.addWidget(self.live_button)
        close=QPushButton('Close');close.clicked.connect(self.close);row.addWidget(close)
        layout.addLayout(row)
        if self.measured.isChecked() and not self.worker.target_layout_pending and self.worker.guide.phase=='ready':
            self.worker.guide.feedback=f'Square {self.worker.square*1000:.2f} mm • {len(self.worker.samples)} poses saved. Press START, then move and pause to collect varied views.'
            if self.worker.samples:self.start_button.setText(f'RESUME • {len(self.worker.samples)} SAVED • 10-second countdown')
            if self.worker.config:
                self.worker.guide.feedback='Fit passed and saved. Press RESUME DEPTH CHECKS when the A4 board is ready.'
                self.start_button.setText(f'RESUME CHECK {len(self.worker.checks)+1} • {len(self.worker.checks)} / 6 recorded • 10-second countdown')
            if self.worker.samples:self.square.setEnabled(False)
            self.worker.publish()
        if self.worker.samples:self.square.setEnabled(False)
        self.worker.publish()
        self.started=time.monotonic();self.last_disk=0;self.last_serial=0;self.last_speech=0;self.last_prompt='';self.speech_process=None
        if start_camera:self.worker.start()
        self.timer=QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(100)

    def toggle_view(self):
        self.views.show_depth=not self.views.show_depth
        self.view_button.setText('SHOWING LIVE DEPTH • click for RGB' if self.views.show_depth else 'SHOWING RGB • click for live depth')
        self.depth_legend.setVisible(self.views.show_depth);self.views.update()

    def begin(self):
        if self.worker.target_layout_pending:
            self.worker.guide.feedback=f'{self.square.value():.2f} mm confirmed. Before capture: enter the board layout (squares across/down; plain or marker pattern).';self.worker.publish();self.tick();return
        if not self.measured.isChecked():
            self.worker.guide.feedback='Enter the measured square size below and tick Square size measured, then press START.';self.worker.publish();self.tick();return
        self.worker.commands.put('start');self.start_button.setEnabled(False);self.start_button.setText('RUNNING — MOVE AND PAUSE' if self.worker.easy else 'RUNNING — MATCH THE FILLED SHAPE');self.start_button.setVisible(True)
        (ROOT/'target_settings.json').write_text(json.dumps({'square_mm':self.square.value(),'measurement_confirmed':True},indent=2))
        self.measured.setEnabled(False);self.square.setEnabled(False)

    def speak(self,message):
        if not self.voice.isChecked() or not shutil.which('spd-say'):return
        if self.speech_process is not None and self.speech_process.poll() is None:return
        self.speech_process=subprocess.Popen(['spd-say','-N','REMA-calibration','-P','notification','-l','en','-r','15',message],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    def tick(self):
        s=self.worker.snapshot
        self.heading.setText(s['title']);self.prompt.setText(s['feedback'])
        self.retry_button.setVisible(bool(s.get('retry_available')));self.retry_button.setEnabled(s['phase']=='blocked')
        if s['phase']=='blocked':self.start_button.setEnabled(False);self.start_button.setText('CAPTURE COMPLETE • results saved • alignment needs review' if s.get('checks_recorded',0)>=6 else 'SAVED RESULTS • alignment needs review')
        self.prompt.setStyleSheet('font-size:25px;font-weight:600;color:'+('#70e4a7' if s['progress']>0 or s['phase']=='done' else '#ffcb80' if s['phase']=='blocked' else '#f0f3f6'))
        self.bar.setValue(round(100*s['progress']))
        self.bar.setFormat('HOLD STILL — %p%' if s['progress']>0 else ('Move to a new view, then pause' if self.worker.easy else 'Automatic capture waits for the requested pose'))
        if s['phase']=='blocked':self.bar.setFormat('PAUSED — saved results retained')
        elif s['phase']=='validation':self.start_button.setText('DEPTH CHECKS RUNNING — follow NEXT below');self.start_button.setEnabled(False)
        self.quality_status.setVisible(False);self.refinement_status.setVisible(False)
        self.instructions.setVisible(s['phase'] not in ('blocked','done'))
        self.capture_progress.setStyleSheet('QProgressBar::chunk{background:#4989b0}')
        self.detail.setText(s.get('detail','')+f"   |   Saved {s.get('samples',0)} (24 minimum)   •   Depth checks {s.get('checks',0)} / 6")
        if s.get('easy_capture'):
            d=s['dashboard'];self.capture_progress.setValue(min(d['saved'],24))
            self.capture_count.setText(f"{d['saved']} VIEWS SAVED — "+(f"{d['remaining']} MORE TO START FITTING" if d['coverage_complete'] and d['remaining'] else ('COVERAGE COMPLETE' if d['coverage_complete'] else 'FILL THE MISSING CHECKS BELOW')))
            labels={'positions':'Image positions','distances':'Near / far','side_tilt':'Side tilt','vertical_tilt':'Top / bottom tilt','tilt_variety':'Angle variety'}
            self.checklist.setText('   |   '.join(('✓ ' if ok else '○ ')+labels[k] for k,ok in d['checks'].items()))
            phase=s['phase'];next_step=s['next_hint'] if phase in ('capture','ready','countdown','paused') else ('Put the board down while fitting.' if phase=='solving' else ('Follow the prompt for six NEW depth checks. Backing must match the A4 paper outline.' if phase=='validation' else s['feedback']))
            validation_stage=self.worker.config is not None and phase in ('ready','validation','paused','blocked')
            if validation_stage:
                n=s.get('checks',0);recorded=s.get('checks_recorded',0);self.capture_progress.setRange(0,6);self.capture_progress.setValue(recorded);self.capture_progress.setFormat('CAPTURE ONLY: %v / 6 saved')
                self.capture_count.setText(f'{recorded} / 6 CHECKS RECORDED — {n} PASSED — {6-recorded} TO RECORD')
                self.stage_label.setText('✓ VIEWS SAVED   →   ✓ PLANE FIT   →   '+('ALIGNMENT FAILED' if recorded==6 and n<6 else 'CHECK ALIGNMENT')+'   →   TIMING')
                measured=[c for c in self.worker.checks if c.get('plane_pass')]
                errors=' / '.join(f"{c['edge_p95_px']:.1f}" if 'edge_p95_px' in c else 'unmeasured' for c in self.worker.checks)
                self.quality_status.setVisible(True)
                self.quality_status.setText(f'ALIGNMENT: {n} / {recorded} passed • {len(measured)} depth planes passed. Edge p95 by check: {errors or "waiting"} px. Limits: median ≤1 px AND p95 ≤3 px.')
                report=self.worker.refinement_report
                if report and phase=='blocked':
                    comparisons='; '.join(f"{c['before']['edge_p95_px']:.1f} → {c['after']['edge_p95_px']:.1f} px" for c in report['checks'] if c['split']=='evaluation')
                    self.refinement_status.setVisible(True)
                    self.refinement_status.setText(f"OFFLINE REFINEMENT: {report['evaluation_passed']} / {report['evaluation_total']} separate evaluation captures pass. Edge p95: {comparisons}. Candidate saved separately; not activated.")
                self.checklist.setText(f'{len(self.worker.samples)} fitting views retained. '+('Capture finished; saved alignment still needs refinement.' if recorded==6 and n<6 else 'Captures are saved even when alignment fails.'));self.checklist.setStyleSheet('font-size:20px;color:#a8d7ff')
                from rema.calibration_guide import CHECK_POSES
                idx=min(recorded,5);lo,hi=[(.50,.63),(.75,.90),(1.05,1.25)][idx%3]
                tilt=CHECK_POSES[idx].tilt;axis=int(np.argmax(np.abs(tilt)));sign=tilt[axis]>0
                edge=(('LEFT' if sign else 'RIGHT') if axis==0 else ('BOTTOM' if sign else 'TOP'))
                next_step=f'CHECK {idx+1}: {lo:.2f}–{hi:.2f} m from camera. Tip the screen {edge} edge closer, then hold still.'
            elif phase in ('timing_ready','timing','done'):
                self.capture_progress.setStyleSheet('QProgressBar::chunk{background:#48bc86}')
                self.stage_label.setText('✓ CAPTURE SAVED   →   ✓ FIT PASSED   →   ✓ DEPTH CHECKED   →   4 TIMING')
                self.capture_count.setText('6 / 6 DEPTH CHECKS PASSED');self.capture_progress.setRange(0,6);self.capture_progress.setValue(6);self.capture_progress.setFormat('%v / 6 depth checks passed')
            if phase=='blocked':next_step=('Put the board down. Review the saved refinement results; repeating the same capture is not the next step.' if s.get('checks_recorded',0)>=6 else 'Keep the isolated tripod setup. Start a new six-check set; alignment errors will not restart check 1.') if s.get('retry_available') else s['feedback']
            self.next_action.setText('NEXT: '+next_step)
            if d['recent_save']:self.capture_count.setStyleSheet('font-size:27px;font-weight:700;color:#70e4a7')
            else:self.capture_count.setStyleSheet('font-size:27px;font-weight:700;color:#e4e7eb')
            if d['stalled'] and phase=='capture':self.prompt.setText(f"NO CAPTURE FOR {d['seconds_since_save']} s — "+s['feedback'])
            pre=s.get('preflight',{});self.preflight_label.setText(('TARGET LAYOUT NEEDED — square size confirmed' if self.worker.target_layout_pending else ('PATTERN FOUND' if pre.get('pattern_ok') else 'PATTERN NOT SEEN — expected REMA 10 × 7 ChArUco print'))+'  |  '+('DEPTH BOARD FOUND' if pre.get('depth_board_ok') else 'DEPTH BOARD NOT SEPARATED'))
            c=s['coverage'];self.coverage_label.setText(f"Coverage: {c['cells']} image regions | spread {c['span'][0]:.0f} × {c['span'][1]:.0f} px | distance spread {c['range_span']:.2f} m | tilt spread {c['tilt_span'][0]:.0f}° / {c['tilt_span'][1]:.0f}°\nNext: {s['next_hint']}")
        if self.worker.result:self.views.frames=self.worker.result;self.views.update()
        self.pause_button.setText('Resume • 3-second countdown' if s['phase']=='paused' else 'Pause')
        self.pause_button.setEnabled(s['phase'] in ('capture','validation','paused','countdown','timing_ready'))
        if s['serial']!=self.last_serial:
            self.last_serial=s['serial'];QApplication.beep()
            self.speak(s['title'] if s['phase'] in ('capture','validation') else s['feedback'])
            self.last_speech=time.monotonic()
        elif s['phase'] in ('capture','validation','timing') and time.monotonic()-self.last_speech>7:
            # Low-priority cues may be dropped if speech is busy; visual prompts remain authoritative.
            text=s['feedback'].split(' • ')[0]
            if text!=self.last_prompt:
                self.speak(text);self.last_prompt=text
            self.last_speech=time.monotonic()
        now=time.monotonic()
        if now-self.last_disk>10:
            self.last_disk=now
            if any(shutil.disk_usage(v).free<25*1024**3 for v in [str(v) for v in _P.RESERVE_VOLUMES]):self.close()
        if now-self.started>7200:self.close()

    def return_live(self):
        self.close()
        if hasattr(self,'camera_lock'):fcntl.flock(self.camera_lock,fcntl.LOCK_UN)
        subprocess.Popen([sys.executable,str(Path(__file__).with_name('run_gui.py'))],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    def closeEvent(self,event):
        self.timer.stop()
        if self.worker.thread.is_alive():self.worker.close()
        event.accept()

def main():
    lock=Path(str(_P.DATA)+'/locks/p0-cpu.lock').open('a');deadline=time.monotonic()+12
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>deadline:sys.exit('Close REMA Live before opening calibration')
            time.sleep(.2)
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--resume-session',type=Path);parser.add_argument('--auto-resume',action='store_true');parser.add_argument('--easy',action='store_true');parser.add_argument('--new-print',action='store_true');parser.add_argument('--square-mm',type=float);parser.add_argument('--target-layout-pending',action='store_true');args=parser.parse_args()
    if args.square_mm is not None and not 1<=args.square_mm<=100:parser.error('Square size must be1–100mm')
    if args.square_mm is not None and args.square_mm>29.7:args.target_layout_pending=True
    app=QApplication(sys.argv[:1]);w=Wizard(CalibrationWorker(args.resume_session,easy=args.easy,new_print=args.new_print,square_mm=args.square_mm,target_layout_pending=args.target_layout_pending));w.camera_lock=lock;w.show();w.raise_();w.activateWindow()
    if args.auto_resume and w.measured.isChecked():QTimer.singleShot(1000,w.begin)
    signal.signal(signal.SIGTERM,lambda *_:w.close())
    sys.exit(app.exec())
if __name__=='__main__':main()
