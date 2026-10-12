import unittest,tempfile,json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np,cv2
from rema.calibration_easy import EasyGuide,robust_detect,print_quality
from rema.calibration_guide import FIT_POSES
from rema.calibration import board
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import calibrate_app

def obs(p):
 points={i:np.array(p.center,float)+[(i%6-2.5)*12,(i//6-1.5)*12] for i in range(24)}
 return dict(distance=p.distance,center=np.array(p.center,float),tilt=p.tilt,corners=points,skew=10)
class EasyCalibrationTests(unittest.TestCase):
 def test_free_pose_can_capture_without_matching_shape(self):
  g=EasyGuide();g.phase='capture';o=obs(FIT_POSES[0]);o['distance']=.67
  self.assertIsNone(g.update(o,1));self.assertEqual(g.update(o,2.5),'save');g.captured(2.5)
  self.assertEqual(g.index,1);self.assertIsNone(g.update(o,5));self.assertIn('Already saved',g.feedback)
 def test_varied_views_reach_solver_with_holdout_coverage(self):
  g=EasyGuide();g.phase='capture'
  for i,p in enumerate(FIT_POSES):
   t=1+i*5;o=obs(p);g.update(o,t);self.assertEqual(g.update(o,t+1.5),'save');action=g.captured(t+1.5)
  self.assertEqual(action,'solve');self.assertTrue(g.coverage()['ready'])
 def test_weak_coverage_does_not_finish_at24(self):
  g=EasyGuide();v=dict(center=[320,240],distance=.8,tilt=[0,0],normal=[0,0,1]);g.views=[v]*24
  self.assertFalse(g.coverage()['ready'])
 def test_motion_bad_pair_and_missing_depth_prevent_capture(self):
  g=EasyGuide();g.phase='capture';o=obs(FIT_POSES[0]);g.update(o,1)
  moved=dict(o,corners={i:p+10 for i,p in o['corners'].items()});self.assertIsNone(g.update(moved,2.5));self.assertEqual(g.progress,0)
  self.assertIsNone(g.update(dict(o,skew=100),4));self.assertIsNone(g.update({'problem':'No depth'},6));self.assertEqual(g.progress,0)
 def test_white_paper_not_mistaken_for_glare(self):
  image=cv2.cvtColor(board(.025).generateImage((500,350)),cv2.COLOR_GRAY2BGR);p,ids=robust_detect(image,.025);self.assertGreaterEqual(len(ids),40)
  q=print_quality(image,p,ids,.025);self.assertGreater(q['contrast'],200);self.assertLess(q['washed_black_percent'],1)
  self.assertEqual(len(robust_detect(np.full((480,640,3),255,np.uint8),.025)[1]),0)
 def test_new_print_does_not_inherit_old_measurement(self):
  with tempfile.TemporaryDirectory() as d,patch.object(calibrate_app,'ROOT',Path(d)):
   (Path(d)/'target_settings.json').write_text(json.dumps(dict(square_mm=23.17,measurement_confirmed=True)))
   w=calibrate_app.CalibrationWorker(easy=True,new_print=True)
   self.assertEqual(w.square,.025);self.assertFalse(w.target_settings['measurement_confirmed']);self.assertEqual(w.samples,[])
   self.assertEqual(json.loads((Path(d)/'target_settings.json').read_text())['square_mm'],23.17)

 def test_far_validation_uses_twelve_spread_corners_without_lowering_fit_gate(self):
  from rema.calibration_guide import rgb_outline
  ids=np.array([0,4,8,18,22,26,36,40,44,45,49,53])
  points,_=cv2.projectPoints(board(.025).getChessboardCorners()[ids],np.zeros(3),np.array([-.125,-.0875,1.05]),np.array([[587.,0,320],[0,587.,240],[0,0,1]]),np.zeros(5))
  points=points.reshape(-1,2)
  center,outline=rgb_outline(points,ids,.025,min_corners=12)
  self.assertEqual(outline.shape,(4,2))
  with self.assertRaises(ValueError):rgb_outline(points,ids,.025)
  g=EasyGuide();g.phase='validation';g.index=2
  o=dict(distance=1.1,center=center,tilt=[0,20],corners={int(i):p for i,p in zip(ids,points)},skew=10)
  g.update(o,1);self.assertEqual(g.update(o,2.5),'check')
 def test_zero_native_detection_still_attempts_scaled_search(self):
  from unittest.mock import MagicMock
  from rema.calibration import detect
  detector=MagicMock();ids=np.arange(24,dtype=np.int32)[:,None];pts=np.full((24,1,2),40.,np.float32)
  detector.detectBoard.side_effect=[(None,None,None,None),(pts,ids,None,None)]
  with patch('rema.calibration.cv2.aruco.CharucoDetector',return_value=detector):
   found,found_ids=detect(np.zeros((480,640),np.uint8))
  self.assertEqual(len(found_ids),24);np.testing.assert_allclose(found,19.75)
