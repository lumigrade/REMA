#!/usr/bin/env python3
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',QT_QUICK_BACKEND='software',QT_XCB_GL_INTEGRATION='none',LIBGL_ALWAYS_SOFTWARE='1')
import argparse,fcntl,json,sys,threading,time,shutil,subprocess,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from PySide6.QtCore import Qt,QTimer,QRect
from PySide6.QtGui import QImage,QPainter,QColor,QFont
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QCheckBox,QDoubleSpinBox,QGroupBox
from rema.live_camera import Cameras
from rema.calibration import load_active,Projection
from rema.board_tracking import FloorTargetTracker,tracked_scene
from rema.provisional import load_provisional,align_provisional
from rema.room_tracking import RoomTracker
from rema.camera_controls import Controls
from rema.matting import TemporalMatte,ActorMask,EmptyRoom,VERSION
from rema.live_core import PersonModel,align_depth,select_person,grid,composite,depth_preview
DATA=Path(str(_P.DATA));OUTPUT=Path(str(_P.OUTPUTS))
def label(im,text,pos=(16,30),scale=.7):
 cv2.putText(im,text,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),4,cv2.LINE_AA)
 cv2.putText(im,text,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,(238,238,238),1,cv2.LINE_AA)
class Engine:
 def __init__(self):
  self.cameras=Cameras();self.stop=threading.Event();self.result=None;self.error='';self.options=dict(max_range=4.,tolerance=.9,scale=1.,dx=0.,dy=0.,horizon=.4,mirror=True,overlay=False,refined=True,show_mask=False,tracked_floor=False,room_tracking=False,preview_height=1.2)
  self.background=EmptyRoom();self.background_action=None;self.control_request=None;self.controls=None;self.control_status='Reading RGB controls…';self.active_calibration=load_active();self.projection=Projection(self.active_calibration) if self.active_calibration else None
  self.tracking_calibration=self.active_calibration or load_provisional();self.room_tracker=RoomTracker(self.tracking_calibration) if self.tracking_calibration else None;self.room_action=None;self.room_feedback=''
  self.provisional_projection=Projection(self.tracking_calibration) if self.tracking_calibration and self.tracking_calibration.get('alignment_model')=='rigid_reprojection_v1' else None
  self.tracker=FloorTargetTracker(self.active_calibration) if self.active_calibration else None
  if self.active_calibration and self.active_calibration.get('timing_proxy',{}).get('passed'):self.cameras.pair_offset_ns=int(self.active_calibration['timing_proxy']['rgb_to_depth_arrival_offset_ms']*1e6)
  self.thread=threading.Thread(target=self.run,daemon=True)
 def start(self):self.cameras.start();self.thread.start()
 def run(self):
  try:
   model=PersonModel();temporal=TemporalMatte();actor=ActorMask();last_mode=None;last_alignment=None;last_id=-1;previous=time.monotonic();fps=0;last_log=0;cached_grid=None;grid_horizon=None;last_tracking=None
   for i in range(3):temporal.infer(np.zeros((480,640,3),np.uint8),i/30)
   temporal.reset()
   try:self.controls=Controls();self.control_status=self.controls.status
   except Exception as e:self.control_status='Camera controls unavailable: '+str(e)
   telemetry=(DATA/'logs/gui/frames.jsonl').open('a',buffering=1)
   while not self.stop.is_set():
    if self.cameras.error:raise RuntimeError(self.cameras.error)
    pair=self.cameras.pair()
    if not pair or pair[0][0]==last_id:time.sleep(.003);continue
    rgb,depth=pair;last_id=rgb[0];now=time.monotonic_ns();o=self.options.copy()
    if (now-rgb[1])/1e9>1 or (now-depth[1])/1e9>1:raise RuntimeError('Camera stream stale; restart after checking connection')
    start=time.monotonic()
    if self.control_request:
     request=self.control_request;self.control_request=None
     try:
      if not self.controls:raise RuntimeError('UVC controls unavailable')
      if request=='lock_wb':self.controls.lock_white_balance()
      elif request=='restore':self.controls.restore()
      else:self.controls.set({'power_line_frequency':int(request)});self.controls.status='Anti-flicker setting read back successfully'
      self.control_status=self.controls.status;self.background.clear()
     except Exception as e:self.control_status='Control not applied: '+str(e)
    if last_mode!=o['refined']:temporal.reset();actor.reset();last_mode=o['refined']
    alignment=(o['scale'],o['dx'],o['dy'])
    if last_alignment is not None and last_alignment!=alignment:self.background.clear();actor.reset()
    last_alignment=alignment
    matte_start=time.monotonic()
    prob=temporal.infer(rgb[2],rgb[1]/1e9) if o['refined'] else model.infer(rgb[2])
    matte_ms=(time.monotonic()-matte_start)*1000
    room_tracking=bool(o['room_tracking'] and self.room_tracker)
    aligned=self.projection.align(depth[2]) if self.projection else (align_provisional(depth[2],self.tracking_calibration,self.provisional_projection) if room_tracking else align_depth(depth[2],prob.shape,*alignment))
    tracking=room_tracking or bool(o['tracked_floor'] and self.tracker)
    tracking_mode='room' if room_tracking else ('board' if tracking else 'static')
    if tracking_mode!=last_tracking:
     self.background.clear();actor.reset()
     if self.tracker:self.tracker.reset()
     if self.room_tracker:self.room_tracker.reset()
     last_tracking=tracking_mode
    if self.background_action=='capture' and not tracking:self.background.request(start)
    elif self.background_action=='clear':self.background.clear()
    self.background_action=None
    if not tracking:self.background.update(rgb[2],aligned,prob,start)
    skew=abs(depth[1]-(rgb[1]+self.cameras.pair_offset_ns))/1e6
    if skew<=100:
     if o['refined']:alpha,z=actor.apply(prob,aligned,o['max_range'],o['tolerance'],rgb[1]/1e9,self.background.match(rgb[2],aligned))
     else:alpha,z=select_person(prob,aligned,o['max_range'],o['tolerance'])
    else:alpha=np.zeros_like(prob);z=None
    source=rgb[2].copy()
    if grid_horizon!=o['horizon']:cached_grid=grid(horizon=o['horizon']);grid_horizon=o['horizon']
    pose=None;tracking_start=time.monotonic()
    if tracking:
     if room_tracking:
      self.room_tracker.preview_height=o['preview_height']
      if self.room_action and start>=self.room_action[1] and self.room_action[0]=='reset':self.room_tracker.reset();self.room_action=None;self.room_feedback='Origin reset'
      pose=self.room_tracker.update(rgb[2],aligned,prob,rgb[0],rgb[1]) if skew<=80 else self.room_tracker.fail('RGB/depth arrivals too far apart',rgb[0],rgb[1])
      if self.room_action and start>=self.room_action[1]:
       success=skew<=80 and self.room_tracker.estimate_floor_now(aligned,prob);self.room_action=None;self.room_feedback='Depth floor estimated' if success else 'Floor not found; keep current floor'
     else:pose=self.tracker.update(rgb[2],prob,rgb[0],rgb[1])
     if pose['valid']:
      provider=self.room_tracker if room_tracking else self.tracker
      scene=tracked_scene(pose,provider.K,provider.dist,square=.02317 if room_tracking else self.tracker.square)
      comp=composite(source,alpha,scene)
     else:comp=source.copy() # target loss always reveals current RGB, never a frozen floor
    else:comp=composite(source,alpha,cached_grid)
    tracking_render_ms=(time.monotonic()-tracking_start)*1000
    if o['show_mask']:comp=cv2.cvtColor(np.rint(alpha*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    raw=depth_preview(depth[2])
    if o['overlay']:source=cv2.addWeighted(source,.65,depth_preview(aligned),.35,0)
    if o['mirror']:source=cv2.flip(source,1);comp=cv2.flip(comp,1);raw=cv2.flip(raw,1)
    label(source,'RGB');label(comp,'Mask' if o['show_mask'] else 'Composited')
    inset=cv2.resize(raw,(176,132),interpolation=cv2.INTER_NEAREST)
    source[12:148,452:632]=(210,210,210);source[14:146,454:630]=inset
    label(source,'Raw depth',(460,32),.42)
    label(source,'Black = no depth | 0-5 m',(455,138),.30)
    if tracking:
     label(comp,('ROOM: ' if room_tracking else 'TARGET: ')+pose['reason'],(12,464),.40)
     if pose['valid']:label(comp,('Approx. 25 cm grid / '+pose['floor_mode'] if room_tracking else '25 cm grid / fixed floor target'),(12,60),.45)
     if room_tracking and self.room_tracker.provisional:label(comp,'REFINED / UNQUALIFIED' if self.provisional_projection else 'TEMPORARY CALIBRATION',(12,86),.52)
     if self.room_action:label(comp,self.room_action[0].upper()+' IN '+str(max(0,int(self.room_action[1]-start)+1))+' s',(12,114),.6)
    else:label(comp,'Static grid / '+('measured RGB-depth calibration' if self.projection else 'approximate RGB-depth alignment'),(12,464),.42)
    if z is None:label(comp,'No depth-confirmed person' if skew<=100 else 'RGB/depth arrivals too far apart',(12,60),.5)
    current=time.monotonic();instant=1/max(.001,current-previous);previous=current;fps=instant if fps==0 else fps*.9+instant*.1
    meta={'matte_backend':temporal.backend,'gpu_uuid':os.environ.get('REMA_GPU_UUID'),'matte_ms':round(matte_ms,2),'tracking_render_ms':round(tracking_render_ms,2),'output_interval_ms':round((current-previous_output)*1000,2) if 'previous_output' in locals() else None,'rgb_arrival_sequence':rgb[0],'rgb_host_arrival_ns':rgb[1],'depth_frame_id':depth[0],'depth_host_arrival_ns':depth[1],'depth_sensor_timestamp_us':depth[3],'host_arrival_skew_ms':round(skew,2),'processing_ms':round((current-start)*1000,2),'composite_fps':round(fps,1),'selected_depth_m':z,'alpha_fraction':round(float((alpha>.5).mean()),4),'calibration':self.tracking_calibration['calibration_id'] if room_tracking else (self.active_calibration['calibration_id'] if self.active_calibration else 'approximate_affine_v1'),'sync_qualified':False,'depth_sdk':'OpenNI2_2.3.0.86_depth_1mm','depth_median_m':float(np.median(depth[2][depth[2]>0])),'person_probability_fraction':float((prob>.5).mean()),'matte_version':VERSION if o['refined'] else 'pphumanseg_v1','background_reference':self.background.status,'controls_status':self.control_status,'pair_offset_ms':self.cameras.pair_offset_ns/1e6,'options':o,'camera_pose':pose,'floor_mode':pose['floor_mode'] if room_tracking else ('fixed_target' if tracking else 'illustrative_static'),'room_feedback':self.room_feedback,'tracking_qualified':False}
    roi=rgb[2][alpha>.9] if np.count_nonzero(alpha>.9)>1000 else rgb[2].reshape(-1,3)
    luminance=roi@np.array([.114,.587,.299]);median=float(np.median(luminance));clipped=float(np.mean(np.max(roi,axis=1)>=250)*100)
    meta['lighting']={'median_luma':round(median,1),'clipped_percent':round(clipped,1),'region':'person' if np.count_nonzero(alpha>.9)>1000 else 'scene','advice':'Add diffuse front/side light' if median<55 else ('Reduce harsh highlights/backlight' if clipped>12 else 'Brightness usable; test fast hands for blur')}
    previous_output=current
    self.result=(source,comp,meta,current,(rgb[2],depth[2]))
    if current-last_log>=1:telemetry.write(json.dumps(meta)+'\n');last_log=current
   telemetry.close()
  except Exception as e:self.error=str(e);self.cameras.stop_event.set()
  finally:
   if self.controls and self.controls.changed:
    try:self.controls.restore()
    except Exception as e:print('RGB control restore failed:',e,flush=True)
 def close(self):self.stop.set();self.cameras.close();self.thread.join(3)
class Views(QWidget):
 def __init__(self):super().__init__();self.frames=None;self.message='Starting Astra Pro RGB + depth…';self.setMinimumSize(800,360)
 def paintEvent(self,event):
  p=QPainter(self);p.fillRect(self.rect(),QColor('#191b1e'))
  if not self.frames:
   p.setPen(QColor('#e4e7eb'));p.setFont(QFont('Sans',14));p.drawText(self.rect(),Qt.AlignCenter,self.message);return
  for i,frame in enumerate(self.frames):
   half=self.width()//2;bounds=QRect(i*half,0,half,self.height());w=min(bounds.width()-8,int(bounds.height()*4/3));h=w*3//4
   target=QRect(bounds.x()+(bounds.width()-w)//2,(bounds.height()-h)//2,w,h)
   image=QImage(frame.data,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_BGR888)
   p.drawImage(target,image)
  p.setPen(QColor('#424650'));p.drawLine(self.width()//2,0,self.width()//2,self.height())
class Window(QWidget):
 def __init__(self,args):
  super().__init__();self.args=args;self.engine=Engine();self.recorder=None;self.record_count=0;self.record_due=0;self.record_started=0;self.record_log=None;self.record_path=None;self.started=time.monotonic();self.last_disk=0;self.auto_done=False;self.failed=False;self.last_presented=None;self.present_time=None
  self.compute_label='RTX3090 matting + CPU tracking' if os.environ.get('REMA_MATTE_BACKEND')=='cuda' else 'CPU only'
  self.setWindowTitle('REMA • Astra Pro • '+self.compute_label);self.resize(1440,1000);self.number_controls={}
  self.setStyleSheet('QWidget {background:#202328;color:#e4e7eb;font-size:13px} QPushButton,QDoubleSpinBox {background:#343a43;padding:7px;border-radius:4px} QGroupBox {border:1px solid #474e59;margin-top:8px;padding-top:12px} QGroupBox::title {subcontrol-origin:margin;left:10px} QPushButton:disabled {color:#777}')
  layout=QVBoxLayout(self);head=QHBoxLayout();title=QLabel('REMA  /  LIVE BODY ISOLATION');title.setStyleSheet('font-size:18px;font-weight:600');head.addWidget(title);head.addStretch();head.addWidget(QLabel(self.compute_label+' • ROOM 3D TRACKING'))
  layout.addLayout(head);self.views=Views();layout.addWidget(self.views,1)
  self.status=QLabel('Connecting to the exact Astra Pro endpoints…');layout.addWidget(self.status)
  row=QHBoxLayout()
  self.mirror=QCheckBox('Mirror');self.mirror.setChecked(True);self.mirror.toggled.connect(lambda v:self.set_option('mirror',v));row.addWidget(self.mirror)
  self.add_number(row,'Max distance (m)','max_range',.6,8,.1)
  self.add_number(row,'Body depth ± (m)','tolerance',.25,2.5,.05)
  self.add_number(row,'Grid horizon','horizon',.05,.75,.02)
  row.addStretch();self.record_button=QPushButton('Record 10 seconds');self.record_button.clicked.connect(self.start_record);row.addWidget(self.record_button)
  self.tune=QPushButton('Alignment controls');self.tune.setCheckable(True);row.addWidget(self.tune)
  quit_button=QPushButton('Close');quit_button.clicked.connect(self.close);row.addWidget(quit_button);layout.addLayout(row)
  maskrow=QHBoxLayout();refine=QCheckBox('Temporal edge refinement');refine.setChecked(True);refine.toggled.connect(lambda v:self.set_option('refined',v));maskrow.addWidget(refine)
  showmask=QCheckBox('Show mask');showmask.toggled.connect(lambda v:self.set_option('show_mask',v));maskrow.addWidget(showmask)
  capture=QPushButton('Capture empty room (5s delay)');self.capture_background_button=capture;capture.clicked.connect(lambda:setattr(self.engine,'background_action','capture'));maskrow.addWidget(capture)
  clear=QPushButton('Clear reference');clear.clicked.connect(lambda:setattr(self.engine,'background_action','clear'));maskrow.addWidget(clear)
  self.bg_status=QLabel('No empty-room reference');maskrow.addWidget(self.bg_status,1);layout.addLayout(maskrow)
  setup=QGroupBox('Guided setup: lighting → calibration → empty-room reference');setup_layout=QVBoxLayout(setup)
  self.light_status=QLabel('Lighting check starting…');setup_layout.addWidget(self.light_status)
  self.control_label=QLabel('');setup_layout.addWidget(self.control_label)
  controls=QHBoxLayout()
  for text,request in [('Lock white balance','lock_wb'),('50 Hz anti-flicker',1),('60 Hz anti-flicker',2),('Restore RGB settings','restore')]:
   button=QPushButton(text);button.clicked.connect(lambda checked=False,r=request:setattr(self.engine,'control_request',r));controls.addWidget(button)
  calibration_button=QPushButton('Open calibration wizard');calibration_button.clicked.connect(self.open_calibration);controls.addWidget(calibration_button);setup_layout.addLayout(controls);layout.addWidget(setup)
  box=QGroupBox('Approximate depth alignment — adjust with overlay; measured calibration comes later');tuning=QHBoxLayout(box)
  self.add_number(tuning,'Scale','scale',.5,1.8,.02);self.add_number(tuning,'Horizontal px','dx',-160,160,2);self.add_number(tuning,'Vertical px','dy',-120,120,2)
  overlay=QCheckBox('Depth overlay on RGB');overlay.toggled.connect(lambda v:self.set_option('overlay',v));tuning.addWidget(overlay);box.setVisible(False);box.setEnabled(self.engine.projection is None);self.tune.toggled.connect(box.setVisible);layout.addWidget(box)
  trackingrow=QHBoxLayout()
  self.floor_tracking=QCheckBox('Track floor target — board is fixed FLAT ON THE FLOOR')
  self.floor_tracking.setEnabled(self.engine.tracker is not None)
  self.floor_tracking.setToolTip('Complete measured calibration first. Keep the printed target visible and fixed on the floor; target loss shows RGB.')
  self.floor_tracking.toggled.connect(self.set_floor_tracking);trackingrow.addWidget(self.floor_tracking)
  self.tracking_status=QLabel('Complete calibration to enable floor tracking' if not self.engine.tracker else 'Place the target flat on the floor, then enable tracking')
  trackingrow.addWidget(self.tracking_status,1);layout.addLayout(trackingrow)
  roomrow=QHBoxLayout();self.room_tracking=QCheckBox('Room 3D tracking — no board');self.room_tracking.setEnabled(self.engine.room_tracker is not None);self.room_tracking.toggled.connect(self.set_room_tracking);roomrow.addWidget(self.room_tracking)
  for title,action in [('Reset 3D origin (5s)','reset'),('Estimate floor (5s)','floor')]:
   button=QPushButton(title);button.clicked.connect(lambda checked=False,a=action:self.queue_room_action(a));roomrow.addWidget(button)
  self.add_number(roomrow,'Preview camera height (m)','preview_height',.25,2.5,.05);layout.addLayout(roomrow)
  self.room_tracking.setChecked(bool(getattr(args,'room_tracking',False)))
  foot=QLabel('Temporary calibration: approximate alignment/scale. Keep textured room visible; move slowly. Lost tracking shows live RGB. Floor may be a preview plane.')
  foot.setStyleSheet('color:#a7afb9;font-size:12px');layout.addWidget(foot)
  self.engine.start();self.timer=QTimer(self);self.timer.timeout.connect(self.tick);self.timer.setTimerType(Qt.PreciseTimer);self.timer.start(8)
 def open_calibration(self):
  self.timer.stop();self.finish_record();self.engine.close()
  subprocess.run(['/usr/bin/python3',str(Path(__file__).with_name('run_gui.py')),'--calibrate'],check=True)
  self.close()
 def queue_room_action(self,action):
  if self.engine.options['room_tracking']:self.engine.room_action=(action,time.monotonic()+5);self.engine.room_feedback=action+' scheduled in 5 seconds'
 def update_tracking_controls(self):
  enabled=self.engine.options['tracked_floor'] or self.engine.options['room_tracking']
  self.capture_background_button.setEnabled(not enabled);self.number_controls['horizon'].setEnabled(not enabled)
  for key in ('scale','dx','dy'):self.number_controls[key].setEnabled(not self.engine.options['room_tracking'] and self.engine.projection is None)
 def set_room_tracking(self,enabled):
  self.set_option('room_tracking',enabled);self.engine.room_action=None
  if enabled:self.floor_tracking.setChecked(False)
  self.update_tracking_controls()
 def set_floor_tracking(self,enabled):
  self.set_option('tracked_floor',enabled)
  if enabled and hasattr(self,'room_tracking'):self.room_tracking.setChecked(False)
  self.update_tracking_controls()
 def set_option(self,k,v):self.engine.options[k]=v
 def add_number(self,row,title,key,low,high,step):
  row.addWidget(QLabel(title));spin=QDoubleSpinBox();spin.setRange(low,high);spin.setSingleStep(step);spin.setDecimals(0 if key in ('dx','dy') else 2);spin.setValue(self.engine.options[key]);spin.valueChanged.connect(lambda v:self.set_option(key,v));row.addWidget(spin);self.number_controls[key]=spin
 def start_record(self):
  if self.recorder or not self.engine.result:return
  OUTPUT.mkdir(parents=True,exist_ok=True);self.record_path=OUTPUT/('rema_live_'+time.strftime('%Y%m%d_%H%M%S')+'.mp4')
  self.record_log=self.record_path.with_suffix('.jsonl').open('w')
  self.dataset=DATA/'recordings'/self.record_path.stem;self.dataset.mkdir(exist_ok=False)
  manifest={'matte_version':VERSION,'RVM_source':json.loads((DATA/'models/rvm/source.json').read_text()),'PPHumanSeg_source':json.loads((DATA/'models/pphumanseg/source.json').read_text()),'depth_units':'metres_float32','calibration':self.engine.tracking_calibration if self.engine.options['room_tracking'] else (self.engine.active_calibration or 'approximate_affine_v1'),'sync_qualified':False,'options':self.engine.options.copy(),'raw_data':str(self.dataset)}
  self.record_path.with_suffix('.config.json').write_text(json.dumps(manifest,indent=2))
  if self.engine.background.rgb is not None:np.savez(self.dataset/'empty_room.npz',rgb=self.engine.background.rgb,depth=self.engine.background.depth)
  self.recorder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-threads','2','-filter_threads','2','-f','rawvideo','-pixel_format','bgr24','-video_size','1280x480','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(self.record_path)],stdin=subprocess.PIPE)
  self.record_count=0;self.record_due=time.monotonic();self.record_started=self.record_due;self.record_button.setEnabled(False)
 def finish_record(self):
  if not self.recorder:return
  self.recorder.stdin.close()
  try:code=self.recorder.wait(timeout=10)
  except subprocess.TimeoutExpired:self.recorder.kill();code=self.recorder.wait()
  self.record_log.close();self.recorder=None;self.record_button.setEnabled(True)
  self.record_button.setText('Saved 10s clip' if code==0 else 'Recording failed')
  print(json.dumps({'recording':str(self.record_path),'frames':self.record_count,'exit':code}),flush=True)
 def tick(self):
  now=time.monotonic()
  if now-self.last_disk>5:
   self.last_disk=now
   if any(shutil.disk_usage(v).free<25*1024**3 for v in [str(v) for v in _P.RESERVE_VOLUMES]):self.engine.error='Disk reserve below 25 GiB'
  error=self.engine.error
  result=self.engine.result
  if result and now-result[3]>2:error=error or 'Camera/processing stalled'
  if error and not self.failed:
   self.failed=True;self.views.frames=None;self.views.message='Camera stopped: '+error;self.views.update();self.status.setText('Close and relaunch after resolving the camera issue.');self.record_button.setEnabled(False);self.finish_record();self.engine.close();return
  if self.failed:return
  if result and result[2]['rgb_arrival_sequence']!=self.last_presented:
   self.last_presented=result[2]['rgb_arrival_sequence']
   left,right,meta,_,raw_pair=result;meta=dict(meta,display_interval_ms=round((now-self.present_time)*1000,2) if self.present_time else None);self.present_time=now;self.views.frames=(left,right);self.views.update()
   self.bg_status.setText('Static reference disabled during camera tracking' if meta.get('floor_mode')!='illustrative_static' else meta['background_reference'])
   pose=meta.get('camera_pose')
   if pose:self.tracking_status.setText(pose['reason']+(f" • p95 {pose['reprojection_p95_px']:.2f} px" if pose['valid'] else '')+' • '+meta.get('room_feedback',''))
   light=meta['lighting'];self.light_status.setText(f"Lighting ({light['region']}): median {light['median_luma']}/255, highlights clipped {light['clipped_percent']}% — {light['advice']}")
   self.control_label.setText(meta['controls_status'])
   z=meta['selected_depth_m'];distance=f'{z:.2f} m' if z is not None else 'waiting for a person'
   self.status.setText(f"{self.compute_label} | {meta['composite_fps']:.1f} composite fps  |  {meta['processing_ms']:.0f} ms processing  |  Nearest: {distance}  |  Arrival skew: {meta['host_arrival_skew_ms']:.0f} ms (exposure sync unqualified)")
   if self.args.auto_record and now-self.started>5 and not self.auto_done:self.auto_done=True;self.start_record()
   if self.recorder and now>=self.record_due:
    try:
     self.recorder.stdin.write(np.hstack((left,right)).tobytes());self.record_log.write(json.dumps(dict(meta,preview_present_monotonic=now,raw_sample=self.record_count))+'\n')
     np.savez(self.dataset/(str(self.record_count)+'.npz'),rgb=raw_pair[0],depth=raw_pair[1])
     self.record_count+=1;self.record_due=self.record_started+self.record_count*.1;self.record_button.setText(f'Recording {self.record_count/10:.1f} / 10s')
     if self.record_count>=100:self.finish_record()
    except BrokenPipeError:self.finish_record()
  if now-self.started>self.args.seconds:self.close()
 def closeEvent(self,event):
  self.timer.stop();self.finish_record();self.engine.close();event.accept()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=int,default=7200);parser.add_argument('--auto-record',action='store_true');parser.add_argument('--room-tracking',action='store_true');args=parser.parse_args()
 if not 1<=args.seconds<=7200:parser.error('Runtime must be 1–7200 seconds')
 lock=(DATA/'locks/p0-cpu.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:sys.exit('Another REMA camera/CPU session owns the lock')
 app=QApplication(sys.argv[:1]);window=Window(args);window.show();signal.signal(signal.SIGTERM,lambda *_:window.close());code=app.exec();sys.exit(code or int(window.failed))
if __name__=='__main__':main()
