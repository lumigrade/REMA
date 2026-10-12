import sys,time,json,subprocess
from pathlib import Path
sys.path.insert(0,'/media/pc/WS_DATA/!WORK/REMA')
import numpy as np,cv2
from rema.room_tracking import RoomTracker,inverse
from rema.board_tracking import tracked_scene
K=np.array([[595.9,0,320],[0,595.8,240],[0,0,1.]])
c=dict(qualified=False,provisional=True,user_authorized=True,calibration_id='synthetic_known_geometry',K_rgb=K.tolist(),dist_rgb=[0]*5)
tr=RoomTracker(c);rng=np.random.default_rng(22)
texture=cv2.GaussianBlur(rng.integers(0,256,(480,640,3),dtype=np.uint8),(3,3),0)
for i in range(120):
 x,y=rng.integers([10,10],[630,470]);cv2.circle(texture,(int(x),int(y)),int(rng.integers(3,10)),tuple(map(int,rng.integers(40,230,3))),-1)
y,x=np.indices((480,640));rays=np.stack([(x-320)/K[0,0],(y-240)/K[1,1],np.ones_like(x)],axis=-1)
output=Path('/home/pc/Documents/Codex/2026-10-09/rema/outputs/REMA_room_tracking_synthetic_motion.mp4')
proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-threads','2','-filter_threads','2','-f','rawvideo','-pixel_format','bgr24','-video_size','1280x480','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(output)],stdin=subprocess.PIPE)
records=[]
def label(im,s,y):
 cv2.putText(im,s,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3,cv2.LINE_AA);cv2.putText(im,s,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
for i in range(120):
 angle=.06*np.sin(i*.04);position=np.array([.12*np.sin(i*.04),.025*np.sin(i*.06),.04*np.sin(i*.03)])
 R=cv2.Rodrigues(np.array([.015*np.sin(i*.035),angle,.005*np.sin(i*.03)]))[0];t=-R@position
 H=K@(R+np.outer(t,np.array([0,0,1.]))/2.5)@np.linalg.inv(K)
 rgb=cv2.warpPerspective(texture,H,(640,480));n=R@np.array([0,0,1.]);depth=((2.5+n@t)/(rays@n)).astype(np.float32)
 person=np.zeros((480,640),np.float32);cv2.ellipse(person,(int(320+110*np.sin(i*.08)),330),(45,120),0,0,360,1,-1)
 rgb[person>.5]=(50,150,220)
 lost=55<=i<63
 if lost:rgb[:]=30
 start=time.perf_counter();pose=tr.update(rgb,depth,person,i+1,1_000_000_000+i*100_000_000)
 rendered=tracked_scene(pose,K,np.zeros(5)) if pose['valid'] else rgb.copy()
 rendered[person>.5]=rgb[person>.5]
 elapsed=(time.perf_counter()-start)*1000
 rec=dict(frame=i,valid=pose['valid'],forced_loss=lost,processing_ms=elapsed,reason=pose['reason'])
 if pose['valid']:
  Twc=np.array(pose['world_from_camera']);rec['translation_error_m']=float(np.linalg.norm(Twc[:3,3]-position));rec['rotation_error_deg']=float(np.degrees(np.linalg.norm(cv2.Rodrigues(Twc[:3,:3]@R)[0])))
 records.append(rec)
 label(rgb,'SYNTHETIC RGB-D / known camera motion',25);label(rgb,'Orange region excluded from room tracking',50)
 label(rendered,'BOARD-FREE 3D TRACKING / synthetic test',25);label(rendered,'Preview floor; no physical accuracy claim',50);label(rendered,pose['reason'],455)
 proc.stdin.write(np.hstack([rgb,rendered]).tobytes())
proc.stdin.close();assert proc.wait(timeout=30)==0
valid=[r for r in records if r['valid']];report=dict(scope='Fully synthetic planar textured room, moving excluded foreground; not physical camera qualification',frames=len(records),tracked=len(valid),forced_loss_rejected=all(not r['valid'] for r in records if r['forced_loss']),p95_translation_error_m=float(np.percentile([r['translation_error_m'] for r in valid],95)),p95_rotation_error_deg=float(np.percentile([r['rotation_error_deg'] for r in valid],95)),processing_ms_p50=float(np.median([r['processing_ms'] for r in records])),video=str(output),frames_detail=records)
Path('/home/pc/rema-data/reports/room_tracking_synthetic.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='frames_detail'},indent=2))
assert report['forced_loss_rejected'] and report['tracked']>=105 and report['p95_translation_error_m']<.025
