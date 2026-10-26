"""Measured RGB/IR calibration; explicit independent depth validation before activation.
R,t map IR/depth camera (x right,y down,z forward, metres) into RGB camera.
Native depth is axial z in metres. Infrared-to-depth pixel equivalence is tested,
not silently assumed. Raw RGB stays distorted; projection applies RGB distortion.
"""
from rema import paths as _P
from pathlib import Path
import json,time
import cv2,numpy as np
ROOT=Path(str(_P.DATA)+'/calibration');ROOT.mkdir(exist_ok=True)
SIZE=(640,480)
def board(square=.025):
 return cv2.aruco.CharucoBoard((10,7),float(square),float(square)*.74,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
def detect(image,square=.025):
 # Small far-away markers benefit from denser adaptive thresholds and a scaled
 # detection pass. Return ORIGINAL sensor pixels; no geometry threshold is relaxed.
 params=cv2.aruco.DetectorParameters();params.adaptiveThreshWinSizeStep=2
 detector=cv2.aruco.CharucoDetector(board(square),detectorParams=params)
 corners,ids,_,_=detector.detectBoard(image)
 count=0 if ids is None else len(ids)
 if count<24:
  for scale in (2,3):
   enlarged=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
   more,more_ids,_,_=detector.detectBoard(enlarged)
   if more_ids is not None and len(more_ids)>count:
    corners=(more+.5)/scale-.5;ids=more_ids;count=len(ids)
   if count>=24:break
 if ids is None:return np.empty((0,2),np.float32),np.empty(0,np.int32)
 return corners.reshape(-1,2),ids.reshape(-1)
def detect_robust(image,square=.025):
 points,ids=detect(image,square)
 if len(ids)>=20:return points,ids
 gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
 enhanced=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(gray)
 extra,eids=detect(enhanced,square)
 return (extra,eids) if len(eids)>len(ids) else (points,ids)
def ir_preview(raw,projector=True):
 low,high=np.percentile(raw,[2,99.3]);image=np.clip((raw.astype(np.float32)-low)*255/max(20,high-low),0,255).astype(np.uint8)
 # Suppress projector speckles only for corner detection/display. Preserve raw uint16 in captures.
 return cv2.GaussianBlur(image,(3,3),.65) if projector else image
def common_points(rgb_points,rgb_ids,ir_points,ir_ids,square):
 common=np.intersect1d(rgb_ids,ir_ids)
 obj=board(square).getChessboardCorners()[common].astype(np.float32)
 rgb=np.array([rgb_points[np.where(rgb_ids==i)[0][0]] for i in common],np.float32)
 ir=np.array([ir_points[np.where(ir_ids==i)[0][0]] for i in common],np.float32)
 return obj,ir,rgb,common
def grid_coverage(views,key):
 centers=np.array([np.mean(v[key],axis=0) for v in views]);cells=set(tuple(np.minimum([3,2],np.maximum([0,0],(p/[160,160]).astype(int)))) for p in centers)
 return len(cells),np.ptp(centers,axis=0)
def solve_stereo(samples,square=.025):
 if len(samples)<24:raise ValueError('Collect at least 24 varied stationary RGB/IR pairs (18 fit + 6 held out).')
 train=[s for i,s in enumerate(samples) if i%4!=3];test=[s for i,s in enumerate(samples) if i%4==3]
 coverage,span=grid_coverage(train,'rgb')
 if coverage<4 or min(span)<100:raise ValueError('Target views are clustered. Cover left/right/top/bottom and different distances.')
 objects=[s['obj'].astype(np.float32) for s in train];irs=[s['ir'].astype(np.float32) for s in train];rgbs=[s['rgb'].astype(np.float32) for s in train]
 flags=cv2.CALIB_FIX_K3
 criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,1e-8)
 ir_rms,ki,di,ri,ti=cv2.calibrateCamera(objects,irs,SIZE,None,None,flags=flags,criteria=criteria)
 rgb_rms,kr,dr,rr,tr=cv2.calibrateCamera(objects,rgbs,SIZE,None,None,flags=flags,criteria=criteria)
 normals=np.array([cv2.Rodrigues(r)[0][:,2] for r in rr]);tilt_span=np.degrees(np.ptp(np.arctan2(normals[:,:2],np.abs(normals[:,2,None])),axis=0))
 if min(tilt_span)<18:raise ValueError('More board tilt needed: tip it left/right and up/down by about20–30 degrees.')
 rms,ki,di,kr,dr,R,t,_,_=cv2.stereoCalibrate(objects,irs,rgbs,ki,di,kr,dr,SIZE,flags=cv2.CALIB_FIX_INTRINSIC,criteria=criteria)
 errors=[];independent=[]
 for s in test:
  ok,rvec,tvec=cv2.solvePnP(s['obj'],s['ir'],ki,di)
  if not ok:raise ValueError('Held-out IR pose solve failed')
  points=(R@(cv2.Rodrigues(rvec)[0]@s['obj'].T+tvec)+t).T
  pred,_=cv2.projectPoints(points,np.zeros(3),np.zeros(3),kr,dr)
  e=np.linalg.norm(pred.reshape(-1,2)-s['rgb'],axis=1);errors.extend(e.tolist());independent.append(float(np.median(e)))
 values=np.r_[ki.ravel(),di.ravel(),kr.ravel(),dr.ravel(),R.ravel(),t.ravel()]
 sane=np.isfinite(values).all() and .001<np.linalg.norm(t)<.3 and all(100<k[0,0]<2000 and 100<k[1,1]<2000 and 0<k[0,2]<640 and 0<k[1,2]<480 for k in [ki,kr])
 median,p95=np.percentile(errors,[50,95]);passed=bool(sane and median<=1 and p95<=3)
 return {'schema':'rema_rgb_ir_calibration_v1','created_unix':time.time(),'image_size':list(SIZE),'square_metres':square,'K_depth':ki.tolist(),'dist_depth':di.ravel().tolist(),'K_rgb':kr.tolist(),'dist_rgb':dr.ravel().tolist(),'R_depth_to_rgb':R.tolist(),'t_depth_to_rgb_metres':t.ravel().tolist(),'coordinates':'x right, y down, z forward; axial depth in metres','intrinsic_model':'OpenCV Brown-Conrady k1 k2 p1 p2; k3 fixed0','ir_to_depth_pixel_equivalence':'requires separate depth validation','fit_views':len(train),'heldout_views':len(test),'fit_ir_rms_px':float(ir_rms),'fit_rgb_rms_px':float(rgb_rms),'fit_stereo_rms_px':float(rms),'heldout_rgb_from_ir_median_px':float(median),'heldout_rgb_from_ir_p95_px':float(p95),'heldout_per_view_median_px':independent,'coverage_cells':coverage,'tilt_span_degrees':tilt_span.tolist(),'stereo_pass':passed,'depth_validation_pass':False,'exposure_sync_qualified':False,'qualified':False}
class Projection:
 def __init__(self,config):
  self.config=config;self.K=np.array(config['K_rgb'],float);self.dist=np.array(config['dist_rgb'],float);self.R=np.array(config['R_depth_to_rgb'],float);self.t=np.array(config['t_depth_to_rgb_metres'],float)
  yy,xx=np.indices((480,640));pixels=np.stack((xx,yy),axis=-1).astype(np.float32).reshape(-1,1,2)
  rays=cv2.undistortPoints(pixels,np.array(config['K_depth']),np.array(config['dist_depth'])).reshape(-1,2)
  self.rays=np.c_[rays,np.ones(len(rays))].astype(np.float32)
 def project_points(self,depth):
  z=depth.reshape(-1);valid=np.isfinite(z)&(z>.25)&(z<10)
  if not valid.any():return np.empty((0,2)),np.empty(0),np.empty((0,3))
  xyz=self.rays[valid]*z[valid,None];points=xyz@self.R.T+self.t
  # OpenCV projectPoints also allocates a full parameter Jacobian for every
  # pixel. Use the equivalent five-coefficient projection for dense frames.
  if self.dist.size in (4,5):
   coefficients=np.pad(self.dist.ravel(),(0,5-self.dist.size));k1,k2,p1,p2,k3=coefficients
   inverse=np.divide(1.,points[:,2],out=np.ones(len(points)),where=points[:,2]!=0)
   x=points[:,0]*inverse;y=points[:,1]*inverse;r2=x*x+y*y
   radial=1+r2*(k1+r2*(k2+r2*k3))
   xd=x*radial+2*p1*x*y+p2*(r2+2*x*x)
   yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y
   uv=np.column_stack((self.K[0,0]*xd+self.K[0,2],self.K[1,1]*yd+self.K[1,2]))
  else:
   uv,_=cv2.projectPoints(points,np.zeros(3),np.zeros(3),self.K,self.dist);uv=uv.reshape(-1,2)
  return uv,points[:,2],points
 def align(self,depth):
  from .depth_projection import align_native
  native=align_native(self,depth)
  if native is not None:return native
  uv,z,_=self.project_points(depth);finite=np.isfinite(uv).all(axis=1)&np.isfinite(z)&(z>.1)
  pixels=np.rint(uv[finite]).astype(np.int32);z=z[finite];inside=(pixels[:,0]>=0)&(pixels[:,0]<640)&(pixels[:,1]>=0)&(pixels[:,1]<480)
  indices=pixels[inside,1]*640+pixels[inside,0];result=np.full(640*480,np.inf,np.float32)
  np.minimum.at(result,indices,z[inside]);result[~np.isfinite(result)]=0
  return result.reshape(480,640)
def validate_depth(config,rgb,depth,detected=None):
 corners,ids=detect_robust(rgb,config['square_metres']) if detected is None else detected
 if len(ids)<12:raise ValueError('Need at least12 target corners visible in RGB')
 obj=board(config['square_metres']).getChessboardCorners()[ids]
 kr=np.array(config['K_rgb']);dr=np.array(config['dist_rgb'])
 ok,r,t=cv2.solvePnP(obj,corners,kr,dr)
 if not ok:raise ValueError('Board pose solve failed')
 rotation=cv2.Rodrigues(r)[0];normal=rotation[:,2];projection=Projection(config)
 uv,z,points=projection.project_points(depth)
 hull=cv2.convexHull(corners.astype(np.float32));mask=np.zeros((480,640),np.uint8);cv2.fillConvexPoly(mask,np.rint(hull).astype(np.int32),1);mask=cv2.erode(mask,np.ones((11,11),np.uint8))
 p=np.rint(uv).astype(np.int32);inside=(p[:,0]>=0)&(p[:,0]<640)&(p[:,1]>=0)&(p[:,1]<480);idx=np.where(inside)[0];idx=idx[mask[p[idx,1],p[idx,0]]>0]
 if len(idx)<150:raise ValueError('Too few board depth samples: move closer or reduce tilt')
 signed=(points[idx]-t.ravel())@normal;expected_range=float(t[2,0]);errors=np.abs(signed)
 fraction=float(np.mean(errors<.025));p50,p95=np.percentile(errors,[50,95])
 # Independent depth-edge check requires rigid backing trimmed flush to the A4 page.
 square=config['square_metres'];ox=(.297-10*square)/2;oy=(.210-7*square)/2
 sheet=np.array([[-ox,-oy,0],[.297-ox,-oy,0],[.297-ox,.210-oy,0],[-ox,.210-oy,0]],np.float32)
 outline,_=cv2.projectPoints(sheet,r,t,kr,dr);sheet_mask=np.zeros((480,640),np.uint8);cv2.fillConvexPoly(sheet_mask,np.rint(outline).astype(np.int32),1)
 # Segment the complete measured coplanar surface BEFORE choosing the board.
 # Clipping it to the expected outline manufactures edges at the crop boundary.
 surface=np.zeros_like(sheet_mask)
 all_inside=np.where(inside)[0];plane_error=np.abs((points[all_inside]-t.ravel())@normal);near=all_inside[plane_error<.025]
 surface[p[near,1],p[near,0]]=1;surface=cv2.morphologyEx(surface,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 count,components=cv2.connectedComponents(surface)
 overlap=np.bincount(components[mask>0],minlength=count);overlap[0]=0
 if not overlap.any():raise ValueError('Depth board silhouette missing; hold board away from the wall')
 component=(components==int(np.argmax(overlap))).astype(np.uint8)
 if component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any():
  corners2d=outline.reshape(-1,2)
  board_in_frame=bool(np.all(corners2d>=[2,2]) and np.all(corners2d<=[637,477]))
  reason=('Board is fully in frame, but another connected depth surface reaches the image edge. Support the board from BEHIND; keep fingers/forearms away from its front outline.' if board_in_frame else 'Expected paper outline is cut off: move the whole board into the picture.')
  return {'validator_version':'full_surface_outline_v2','passed':False,'outline_observable':False,'failure_kind':'connected_surface_at_boundary' if board_in_frame else 'board_out_of_frame','failure_reason':reason,'board_in_frame':board_in_frame,'board_range_metres':expected_range,'normal_rgb':normal.tolist(),'samples':len(idx),'plane_median_metres':float(p50),'plane_p95_metres':float(p95),'within25mm_fraction':fraction,'plane_pass':bool(p50<=.015 and p95<=.035 and fraction>=.8),'outline_pass':False,'rgb_corners':corners.tolist()}
 contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 contour=max(contours,key=cv2.contourArea);observed=np.zeros_like(surface);cv2.drawContours(observed,[contour],-1,1,1)
 expected=sheet_mask-cv2.erode(sheet_mask,np.ones((3,3),np.uint8))
 to_observed=cv2.distanceTransform(1-observed,cv2.DIST_L2,3);to_expected=cv2.distanceTransform(1-expected,cv2.DIST_L2,3)
 edge_errors=np.r_[to_observed[expected>0],to_expected[observed>0]];edge50,edge95=np.percentile(edge_errors,[50,95])
 plane_pass=bool(p50<=.015 and p95<=.035 and fraction>=.8);outline_pass=bool(edge50<=1 and edge95<=3)
 area_ratio=float(cv2.contourArea(contour)/max(1,int(sheet_mask.sum())))
 reasons=[]
 if not plane_pass:reasons.append(f'Depth plane failed: median {p50*1000:.1f} mm, p95 {p95*1000:.1f} mm, within25mm {fraction:.0%}')
 if not outline_pass:reasons.append(f'A4 outline mismatch: median {edge50:.1f} px (limit 1), p95 {edge95:.1f} px (limit 3). Recorded for alignment review; continue the complete check set')
 return {'validator_version':'full_surface_outline_v2','board_range_metres':expected_range,'normal_rgb':normal.tolist(),'samples':len(idx),'plane_median_metres':float(p50),'plane_p95_metres':float(p95),'within25mm_fraction':fraction,'plane_pass':plane_pass,'outline_pass':outline_pass,'surface_to_expected_area_ratio':area_ratio,'failure_reason':'; '.join(reasons),'edge_median_px':float(edge50),'edge_p95_px':float(edge95),'passed':bool(plane_pass and outline_pass),'rgb_corners':corners.tolist()}

def depth_validation_summary(checks):
 if len(checks)<6:return False,'Need6 independent board poses, at least3 distances and both tilt axes'
 if not all(c.get('passed') for c in checks):return False,'Recorded set contains failed or unobservable checks; review alignment before activation'
 distances=np.array([c['board_range_metres'] for c in checks]);normals=np.array([c['normal_rgb'] for c in checks]);tilts=np.degrees(np.arctan2(normals[:,:2],np.abs(normals[:,2,None])))
 bins=len(set(np.round(distances/.2).astype(int)))
 good=all(c['passed'] for c in checks) and np.ptp(distances)>=.4 and bins>=3 and min(np.ptp(tilts,axis=0))>=15
 return bool(good),'Pass' if good else 'Need passing held-out poses over >=0.4m range and >=15deg tilt variation in both axes'
def load_active():
 p=ROOT/'active.json'
 if not p.exists():return None
 c=json.loads(p.read_text())
 if not c.get('qualified') or not c.get('stereo_pass') or not c.get('depth_validation_pass') or c.get('image_size')!=[640,480]:return None
 return c

def estimate_arrival_offset(samples):
 """Fit a relative arrival-time correction from independent board range signals.
 Positive delta pairs RGB arrival t with depth arrival t+delta. No exposure timestamp claim.
 """
 if len(samples)<60:raise ValueError('Need at least60 valid target observations during an in/out sweep')
 samples=sorted(samples,key=lambda s:s['rgb_time']);tr=np.array([s['rgb_time'] for s in samples]);td=np.array([s['depth_time'] for s in samples]);zr=np.array([s['rgb_range'] for s in samples]);zd=np.array([s['depth_range'] for s in samples])
 unique=np.r_[True,np.diff(td)>0];td=td[unique];zd=zd[unique]
 if len(td)<40 or np.ptp(zr)<.25:raise ValueError('Move target toward/away by at least25cm repeatedly')
 mid=(tr[0]+tr[-1])/2;valid=(tr>tr[0]+.2)&(tr<tr[-1]-.2);fit=valid&(tr<mid);test=valid&(tr>=mid)
 offsets=np.arange(-.15,.1501,.005);cost=[]
 for delta in offsets:
  errors=zr-np.interp(tr+delta,td,zd);center=np.median(errors[fit]);cost.append(float(np.mean((errors[fit]-center)**2)))
 best=int(np.argmin(cost));delta=float(offsets[best]);error=zr-np.interp(tr+delta,td,zd);bias=float(np.median(error[fit]));test_rmse=float(np.sqrt(np.mean((error[test]-bias)**2)))
 zero=zr-np.interp(tr,td,zd);zero_rmse=float(np.sqrt(np.mean((zero[test]-np.median(zero[fit]))**2)))
 passed=bool(0<best<len(offsets)-1 and test_rmse<.03 and (test_rmse<zero_rmse*.8 or abs(delta)<.015))
 return {'rgb_to_depth_arrival_offset_ms':round(delta*1000,1),'heldout_range_rmse_m':test_rmse,'uncorrected_heldout_rmse_m':zero_rmse,'range_bias_m':bias,'samples':len(samples),'passed':passed,'exposure_sync_qualified':False,'method':'stationary-camera board axial in-out motion, first half fit / second half check; arrival offset proxy'}

def extract_board_plane(depth,seed,intrinsics):
 fx,fy,cx,cy=intrinsics;x,y=map(int,seed);h,w=depth.shape
 patch=depth[max(0,y-2):min(h,y+3),max(0,x-2):min(w,x+3)];values=patch[patch>.3]
 if len(values)<5:raise ValueError('Click a valid interior point on the board depth surface')
 z0=float(np.median(values));yy,xx=np.indices(depth.shape)
 valid=(depth>.3)&(np.abs(depth-z0)<.20)&(np.abs(xx-x)<230)&(np.abs(yy-y)<200)
 n,labels,stats,_=cv2.connectedComponentsWithStats(valid.astype(np.uint8),8)
 label_id=labels[y,x]
 if not label_id or stats[label_id,cv2.CC_STAT_AREA]<400:raise ValueError('Depth board region too small or disconnected')
 use=labels==label_id;v,u=np.where(use);z=depth[v,u];all_points=np.c_[(u-cx)*z/fx,(v-cy)*z/fy,z]
 stride=max(1,len(all_points)//2500);points=all_points[::stride];seed3=np.array([(x-cx)*z0/fx,(y-cy)*z0/fy,z0]);rng=np.random.default_rng(15);best=None;best_count=0
 for _ in range(90):
  tri=points[rng.choice(len(points),3,replace=False)];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);length=np.linalg.norm(normal)
  if length<1e-8:continue
  normal/=length;offset=-normal@tri[0]
  if abs(normal[2])<.45 or abs(normal@seed3+offset)>.015:continue
  inliers=np.abs(points@normal+offset)<.012;count=int(inliers.sum())
  if count>best_count:best=inliers;best_count=count
 if best is None or best_count<250:raise ValueError('No reliable plane around selected board centre')
 selected=points[best]
 for _ in range(2):
  centre=selected.mean(axis=0);_,_,vt=np.linalg.svd(selected-centre,full_matrices=False);normal=vt[-1]
  if normal[2]<0:normal=-normal
  offset=-normal@centre;selected=points[np.abs(points@normal+offset)<.01]
 errors=np.abs(selected@normal+offset)
 plane_mask=np.zeros(depth.shape,np.uint8);inliers=np.abs(all_points@normal+offset)<.015;plane_mask[v[inliers],u[inliers]]=1
 return {'normal':normal,'offset':float(offset),'points':selected[::max(1,len(selected)//800)].astype(np.float32),'mask':plane_mask,'median_error_m':float(np.median(errors)),'seed':np.array(seed),'range_m':z0}
def solve_planes(samples,intrinsics,square=.025):
 if len(samples)<24:raise ValueError('Need24 different still RGB + board-depth poses')
 train=[s for i,s in enumerate(samples) if i%4!=3];test=[s for i,s in enumerate(samples) if i%4==3]
 coverage,span=grid_coverage(train,'rgb')
 if coverage<4 or min(span)<100:raise ValueError('Cover more left/right/top/bottom image positions')
 objects=[s['obj'].astype(np.float32) for s in train];pixels=[s['rgb'].astype(np.float32) for s in train]
 rms,K,dist,rvecs,tvecs=cv2.calibrateCamera(objects,pixels,SIZE,None,None,flags=cv2.CALIB_FIX_K3,criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,1e-8))
 nr=[];dr=[];nd=[];dd=[]
 for sample,r,t in zip(train,rvecs,tvecs):
  n=cv2.Rodrigues(r)[0][:,2]
  if n[2]<0:n=-n
  nr.append(n);dr.append(-n@t.ravel());nd.append(sample['plane_normal']);dd.append(sample['plane_offset'])
 nr=np.array(nr);nd=np.array(nd);dr=np.array(dr);dd=np.array(dd)
 tilt=np.degrees(np.ptp(np.arctan2(nr[:,:2],np.abs(nr[:,2,None])),axis=0))
 if min(tilt)<25 or np.linalg.cond(nr)>15:raise ValueError('Need stronger varied board tilts around both axes (about±25degrees)')
 u,_,vt=np.linalg.svd(nd.T@nr);R=vt.T@u.T
 if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
 t=np.linalg.lstsq(nr,dd-dr,rcond=None)[0]
 # Refine six rigid parameters with robust plane residuals. Depth scale/rays remain explicit SDK values.
 parameters=np.r_[cv2.Rodrigues(R)[0].ravel(),t]
 pointsets=[s['plane_points'][::max(1,len(s['plane_points'])//300)] for s in train]
 def residual(p):
  rotation=cv2.Rodrigues(p[:3])[0];translation=p[3:]
  return np.concatenate([(points@rotation.T+translation)@normal+offset for points,normal,offset in zip(pointsets,nr,dr)])
 for _ in range(25):
  error=residual(parameters);eps=1e-5;jac=np.column_stack([(residual(parameters+np.eye(6)[i]*eps)-error)/eps for i in range(6)])
  weights=np.minimum(1,.008/np.maximum(np.abs(error),1e-8));J=jac*np.sqrt(weights[:,None]);e=error*np.sqrt(weights)
  step=np.linalg.lstsq(J,-e,rcond=None)[0]
  if np.linalg.norm(step)>1:raise ValueError('Plane fit diverged; verify selected depth patches')
  parameters+=step
  if np.linalg.norm(step)<1e-7:break
 R=cv2.Rodrigues(parameters[:3])[0];t=parameters[3:];pixel_errors=[];plane_errors=[]
 for sample in test:
  ok,r,p=cv2.solvePnP(sample['obj'],sample['rgb'],K,dist)
  if not ok:raise ValueError('Held-out board pose failure')
  pred,_=cv2.projectPoints(sample['obj'],r,p,K,dist);pixel_errors.extend(np.linalg.norm(pred.reshape(-1,2)-sample['rgb'],axis=1))
  n=cv2.Rodrigues(r)[0][:,2];plane_errors.extend(np.abs((sample['plane_points']@R.T+t-p.ravel())@n))
 pixel50,pixel95=np.percentile(pixel_errors,[50,95]);plane50,plane95=np.percentile(plane_errors,[50,95]);fx,fy,cx,cy=intrinsics
 sane=np.isfinite(np.r_[K.ravel(),dist.ravel(),R.ravel(),t]).all() and 100<K[0,0]<2000 and 100<K[1,1]<2000 and np.linalg.norm(t)<.3 and np.linalg.norm(parameters[:3])<.4
 passed=bool(sane and pixel50<=1 and pixel95<=3 and plane50<=.015 and plane95<=.035)
 return {'schema':'rema_rgb_depth_planes_v1','created_unix':time.time(),'image_size':list(SIZE),'square_metres':square,'K_depth':[[fx,0,cx],[0,fy,cy],[0,0,1]],'dist_depth':[0]*5,'depth_intrinsics_source':'OpenNI CoordinateConverter, zero additional distortion assumed; held-out depth outline validation REQUIRED','K_rgb':K.tolist(),'dist_rgb':dist.ravel().tolist(),'R_depth_to_rgb':R.tolist(),'t_depth_to_rgb_metres':t.tolist(),'coordinates':'x right,y down,z forward; axial depth metres; transform depth into RGB','fit_views':len(train),'heldout_views':len(test),'fit_rgb_rms_px':float(rms),'heldout_rgb_from_ir_median_px':float(pixel50),'heldout_rgb_from_ir_p95_px':float(pixel95),'heldout_rgb_reprojection_median_px':float(pixel50),'heldout_rgb_reprojection_p95_px':float(pixel95),'heldout_plane_median_m':float(plane50),'heldout_plane_p95_m':float(plane95),'metric_note':'RGB heldout pose reprojection is not cross-sensor edge alignment; separate depth checks are mandatory','coverage_cells':coverage,'tilt_span_degrees':tilt.tolist(),'stereo_pass':passed,'depth_validation_pass':False,'exposure_sync_qualified':False,'qualified':False}
