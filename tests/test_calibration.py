import unittest
from unittest.mock import patch
import numpy as np,cv2
from rema.calibration import board,detect,solve_stereo,Projection,validate_depth,depth_validation_summary,extract_board_plane,solve_planes,estimate_arrival_offset
from rema.live_camera import Cameras
class Calibration(unittest.TestCase):
 def config(self):
  k=[[500.,0,320],[0,500.,240],[0,0,1]]
  return dict(K_depth=k,K_rgb=k,dist_depth=[0]*5,dist_rgb=[0]*5,R_depth_to_rgb=np.eye(3).tolist(),t_depth_to_rgb_metres=[0,0,0],square_metres=.025)
 def test_printed_target_detects_all_54_corners(self):
  image=board().generateImage((1000,700));image=cv2.copyMakeBorder(image,80,80,80,80,cv2.BORDER_CONSTANT,value=255)
  points,ids=detect(image);self.assertEqual(len(ids),54);self.assertEqual(len(set(ids)),54)
 def test_empty_depth_projects_to_empty(self):
  self.assertFalse(Projection(self.config()).align(np.zeros((480,640),np.float32)).any())
 def test_projection_preserves_axial_metres_and_identity(self):
  depth=np.full((480,640),1.7,np.float32);aligned=Projection(self.config()).align(depth);np.testing.assert_allclose(aligned,depth,atol=1e-6)
 def test_depth_outline_rejects_pixel_misalignment(self):
  c=self.config();obj=board().getChessboardCorners();r=np.zeros(3);t=np.array([-.125,-.0875,1.]);points,_=cv2.projectPoints(obj,r,t,np.array(c['K_rgb']),np.zeros(5));ids=np.arange(54)
  depth=np.full((480,640),3.,np.float32);depth[187:294,246:395]=1.
  with patch('rema.calibration.detect',return_value=(points.reshape(-1,2),ids)):
   good=validate_depth(c,np.zeros((480,640,3),np.uint8),depth);self.assertTrue(good['passed'],good)
   wrong=dict(c,t_depth_to_rgb_metres=[.04,0,0]);bad=validate_depth(wrong,np.zeros((480,640,3),np.uint8),depth);self.assertFalse(bad['passed']);self.assertGreater(bad['edge_p95_px'],3)
 def test_depth_scale_error_rejected(self):
  c=self.config();obj=board().getChessboardCorners();points,_=cv2.projectPoints(obj,np.zeros(3),np.array([-.125,-.0875,1.]),np.array(c['K_rgb']),np.zeros(5));depth=np.full((480,640),3.,np.float32);depth[187:294,246:395]=1.08
  with patch('rema.calibration.detect',return_value=(points.reshape(-1,2),np.arange(54))):
   with self.assertRaises(ValueError):validate_depth(c,np.zeros((480,640,3),np.uint8),depth)
 def test_coplanar_repeated_depth_checks_not_qualified(self):
  checks=[dict(passed=True,board_range_metres=1.,normal_rgb=[0,0,1]) for _ in range(6)]
  passed,_=depth_validation_summary(checks);self.assertFalse(passed)
 def test_timing_offset_sign_and_heldout_check(self):
  samples=[dict(rgb_time=float(t),depth_time=float(t),rgb_range=1+.25*np.sin(3*(t-.08)),depth_range=1+.25*np.sin(3*t)) for t in np.arange(0,12,.08)]
  result=estimate_arrival_offset(samples);self.assertTrue(result['passed'],result);self.assertAlmostEqual(result['rgb_to_depth_arrival_offset_ms'],-80,delta=5)
 def test_corrected_pair_selects_history_not_newest(self):
  c=Cameras();c.rgb=(3,300,None);c.rgb_history.extend([(1,100,None),(2,200,None),(3,300,None)]);c.depth.extend([(1,150,None),(2,250,None),(3,350,None)]);c.pair_offset_ns=-50
  self.assertEqual(c.pair()[1][1],250);c.pair_offset_ns=100;self.assertEqual(c.pair()[0][1],200)
 def test_stereo_recovers_known_transform_on_heldout_views(self):
  rng=np.random.default_rng(6);obj=board().getChessboardCorners();ki=np.array([[580.,0,320],[0,578.,240],[0,0,1]]);kr=np.array([[610.,0,317],[0,607.,242],[0,0,1]])
  R=cv2.Rodrigues(np.array([.005,.015,-.008]))[0];t=np.array([.025,.002,.001]);samples=[]
  for i in range(28):
   r=rng.uniform(-.4,.4,3);r[2]*=.3;tr=np.array([rng.uniform(-.26,.08),rng.uniform(-.2,.07),rng.uniform(.65,1.15)]);points=(cv2.Rodrigues(r)[0]@obj.T).T+tr
   ir,_=cv2.projectPoints(points,np.zeros(3),np.zeros(3),ki,np.zeros(5));rgb,_=cv2.projectPoints(points@R.T+t,np.zeros(3),np.zeros(3),kr,np.zeros(5))
   samples.append(dict(obj=obj,ir=(ir.reshape(-1,2)+rng.normal(0,.025,(54,2))).astype(np.float32),rgb=(rgb.reshape(-1,2)+rng.normal(0,.025,(54,2))).astype(np.float32)))
  result=solve_stereo(samples);self.assertTrue(result['stereo_pass'],result);self.assertFalse(result['qualified']);np.testing.assert_allclose(result['t_depth_to_rgb_metres'],t,atol=.002)

 def test_depth_plane_selection_rejects_background(self):
  yy,xx=np.indices((480,640));depth=np.full((480,640),2.5,np.float32);surface=1/(1-.25*(xx-320)/580+.10*(yy-240)/578)
  depth[180:301,240:401]=surface[180:301,240:401]
  result=extract_board_plane(depth,(320,240),[580,578,320,240]);normal=np.array([-.25,.1,1]);normal/=np.linalg.norm(normal)
  np.testing.assert_allclose(result['normal'],normal,atol=.001);self.assertEqual(result['mask'][20,20],0);self.assertEqual(result['mask'][240,320],1)
 def test_plane_route_recovers_known_rigid_calibration(self):
  rng=np.random.default_rng(12);obj=board().getChessboardCorners();K=np.array([[605.,0,318],[0,607.,241],[0,0,1]])
  R=cv2.Rodrigues(np.array([.006,-.012,.009]))[0];translation=np.array([.027,.002,-.001]);samples=[]
  for i in range(32):
   r=rng.uniform(-.50,.50,3);r[2]*=.3;rot=cv2.Rodrigues(r)[0];t=np.array([rng.uniform(-.28,.10),rng.uniform(-.20,.08),rng.uniform(.65,1.1)])
   points=obj@rot.T+t;rgb,_=cv2.projectPoints(points@R.T+translation,np.zeros(3),np.zeros(3),K,np.zeros(5));normal=rot[:,2]
   samples.append(dict(obj=obj,rgb=(rgb.reshape(-1,2)+rng.normal(0,.02,(54,2))).astype(np.float32),plane_normal=normal,plane_offset=-normal@t,plane_points=(points+rng.normal(0,.0003,points.shape)).astype(np.float32)))
  result=solve_planes(samples,[580,578,320,240]);self.assertTrue(result['stereo_pass'],result);self.assertFalse(result['qualified']);np.testing.assert_allclose(result['t_depth_to_rgb_metres'],translation,atol=.003)

 def test_outline_uses_full_surface_not_expected_crop(self):
  c=self.config();obj=board().getChessboardCorners();points,_=cv2.projectPoints(obj,np.zeros(3),np.array([-.125,-.0875,1.]),np.array(c['K_rgb']),np.zeros(5))
  depth=np.full((480,640),3.,np.float32);depth[150:330,210:431]=1.
  with patch('rema.calibration.detect',return_value=(points.reshape(-1,2),np.arange(54))):
   result=validate_depth(c,np.zeros((480,640,3),np.uint8),depth)
  self.assertTrue(result['plane_pass']);self.assertFalse(result['outline_pass'])
  self.assertGreater(result['edge_p95_px'],30);self.assertGreater(result['surface_to_expected_area_ratio'],2)
 def test_disconnected_coplanar_object_does_not_replace_board(self):
  c=self.config();obj=board().getChessboardCorners();points,_=cv2.projectPoints(obj,np.zeros(3),np.array([-.125,-.0875,1.]),np.array(c['K_rgb']),np.zeros(5))
  depth=np.full((480,640),3.,np.float32);depth[187:294,246:395]=1.;depth[20:450,10:200]=1.
  with patch('rema.calibration.detect',return_value=(points.reshape(-1,2),np.arange(54))):
   result=validate_depth(c,np.zeros((480,640,3),np.uint8),depth)
  self.assertTrue(result['passed'],result)

 def test_connected_hand_does_not_report_board_out_of_frame(self):
  c=self.config();obj=board().getChessboardCorners();points,_=cv2.projectPoints(obj,np.zeros(3),np.array([-.125,-.0875,1.]),np.array(c['K_rgb']),np.zeros(5))
  depth=np.full((480,640),3.,np.float32);depth[187:294,246:395]=1.;depth[290:480,260:275]=1.
  with patch('rema.calibration.detect',return_value=(points.reshape(-1,2),np.arange(54))):
   result=validate_depth(c,np.zeros((480,640,3),np.uint8),depth)
  self.assertFalse(result['passed']);self.assertTrue(result['plane_pass']);self.assertTrue(result['board_in_frame']);self.assertFalse(result['outline_observable'])
  self.assertEqual(result['failure_kind'],'connected_surface_at_boundary')
