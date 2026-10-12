import sys,os,json,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from rema.calibration_easy import EasyGuide
import calibrate_app as c
class ProgressResumeTests(unittest.TestCase):
 def test_saved_count_is_separate_from_hold_and_stall(self):
  g=EasyGuide();g.restore([dict(rgb=np.array([[100+(i%2)*200,100+(i%3)*100]]),plane_normal=np.array([(-1)**i*.4,(-1)**(i//2)*.4,1.]),plane_points=np.array([[0,0,.5+(i%3)*.25]])) for i in range(23)])
  g.start(0,23);g.phase='capture';d=g.dashboard(40)
  self.assertEqual(d['remaining'],1);self.assertEqual(d['saved'],23);self.assertTrue(d['stalled']);self.assertEqual(g.progress,0)
  g.progress=.5;self.assertEqual(g.dashboard(41)['saved'],23)
 def test_easy_resume_restores_existing_file_without_rewriting(self):
  with tempfile.TemporaryDirectory() as d,patch.object(c,'ROOT',Path(d)):
   folder=Path(d)/'session_test_easy';folder.mkdir()
   path=folder/'pose_000.npz'
   np.savez(path,rgb=np.array([[200.,200.],[210.,210.]]),ids=np.array([1,2]),obj=np.zeros((2,3)),plane_normal=np.array([0.,0.,1.]),plane_points=np.array([[0.,0.,.6]]),square_metres=.025,guided_pose=0,depth_intrinsics=np.array([570.,570.,320.,240.]))
   before=path.read_bytes()
   with patch.object(c,'estimate_guide_K',return_value=np.eye(3)):
    w=c.CalibrationWorker(folder,easy=True,square_mm=25)
   self.assertEqual(len(w.samples),1);self.assertEqual(w.guide.index,1);self.assertEqual(len(w.guide.views),1);self.assertEqual(path.read_bytes(),before)
 def test_resume_rejects_different_square(self):
  with tempfile.TemporaryDirectory() as d,patch.object(c,'ROOT',Path(d)):
   folder=Path(d)/'session_test_easy';folder.mkdir();np.savez(folder/'pose_000.npz',square_metres=.02317,guided_pose=0)
   with self.assertRaises(ValueError):c.CalibrationWorker(folder,easy=True,square_mm=25)

 def test_fit_resume_retains_views_and_continues_depth_checks(self):
  with tempfile.TemporaryDirectory() as d,patch.object(c,'ROOT',Path(d)):
   folder=Path(d)/'session_test_easy';folder.mkdir()
   for i in range(24):
    np.savez(folder/f'pose_{i:03}.npz',rgb=np.array([[200.,200.]]),plane_normal=np.array([0.,0.,1.]),plane_points=np.array([[0.,0.,.6]]),square_metres=.025,guided_pose=i)
   candidate=folder/'candidate.json';candidate.write_text(json.dumps(dict(stereo_pass=True,square_metres=.025)))
   before=candidate.read_bytes()
   with patch.object(c,'estimate_guide_K',return_value=np.eye(3)),patch.object(c,'Projection'):
    w=c.CalibrationWorker(folder,easy=True,square_mm=25)
   w.begin_session(100)
   self.assertEqual(w.guide.phase,'validation');self.assertEqual(w.guide.index,0);self.assertEqual(w.guide.until,110)
   self.assertEqual(len(w.samples),24);self.assertEqual(candidate.read_bytes(),before)
   # An orphan raw check must never be overwritten or silently ignored.
   np.savez(folder/'depth_check_000.npz',guided_pose=0,validation_set=0)
   with patch.object(c,'estimate_guide_K',return_value=np.eye(3)),self.assertRaises(ValueError):
    c.CalibrationWorker(folder,easy=True,square_mm=25)
   (folder/'depth_check_000.npz').unlink()
   (folder/'depth_checks_set_0_FAILED.json').write_text('[]')
   with patch.object(c,'estimate_guide_K',return_value=np.eye(3)),self.assertRaises(ValueError):
    c.CalibrationWorker(folder,easy=True,square_mm=25)

 def test_failed_checks_not_counted_and_retry_preserves_evidence(self):
  with tempfile.TemporaryDirectory() as d,patch.object(c,'ROOT',Path(d)):
   folder=Path(d)/'session_test_easy';folder.mkdir()
   for i in range(24):
    np.savez(folder/f'pose_{i:03}.npz',rgb=np.array([[200.,200.]]),plane_normal=np.array([0.,0.,1.]),plane_points=np.array([[0.,0.,.6]]),square_metres=.025,guided_pose=i)
   (folder/'candidate.json').write_text(json.dumps(dict(stereo_pass=True,square_metres=.025)))
   check=dict(passed=False,raw_capture_index=0,edge_p95_px=13.4)
   (folder/'depth_checks.json').write_text(json.dumps([check]))
   archive=folder/'depth_checks_set_0_FAILED.json';archive.write_text(json.dumps([check]))
   raw=folder/'depth_check_000.npz';np.savez(raw,guided_pose=0,validation_set=0)
   before=(archive.read_bytes(),raw.read_bytes())
   with patch.object(c,'estimate_guide_K',return_value=np.eye(3)),patch.object(c,'Projection'):
    w=c.CalibrationWorker(folder,easy=True,square_mm=25)
   w.publish();self.assertEqual(w.snapshot['checks'],0);self.assertEqual(w.snapshot['checks_recorded'],1)
   self.assertEqual(w.guide.phase,'blocked');self.assertTrue(w.retry_available)
   w.retry_validation(100)
   self.assertEqual(w.guide.phase,'validation');self.assertEqual(w.guide.index,0)
   self.assertEqual(w.check_index,1);self.assertEqual(w.guide.validation_attempt,1)
   self.assertEqual((archive.read_bytes(),raw.read_bytes()),before)
   self.assertEqual(json.loads((folder/'depth_checks.json').read_text()),[])
