import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import dataclasses, hashlib, json, pathlib, tempfile, unittest
import numpy as np
from rema.contracts import Frame, pair_frames, project_depth, validate_calibration, load_replay
class Contracts(unittest.TestCase):
 def frame(self,i,t,stream='rgb'):
  return Frame(stream,i,t,t+5000000,t,'mapped_device','synthetic-v1','none','x','')
 def calibration(self):
  sensor={'K':[[500,0,320],[0,500,240],[0,0,1]],'distortion':[0,0,0,0,0],'distortion_model':'brown_conrady','size':[640,480]}
  return {'units':'metres','transform_direction':'depth_to_rgb','T_rgb_from_depth':[[1,0,0,.05],[0,1,0,0],[0,0,1,0],[0,0,0,1]],'depth_scale_metres_per_unit':.001,'rgb':sensor,'depth':sensor}
 def test_metric_baseline_and_direction(self):
  uv=project_depth([[0,0,1],[0,0,2]],self.calibration()); np.testing.assert_allclose(uv,[[345,240],[332.5,240]])
 def test_distortion_is_applied(self):
  c=self.calibration(); c['rgb']['distortion']=[.1,0,0,0,0]; uv=project_depth([[.45,0,1]],c); np.testing.assert_allclose(uv,[[576.25,240]])
 def test_reject_reflection_and_units(self):
  c=self.calibration(); c['T_rgb_from_depth'][0][0]=-1
  with self.assertRaises(ValueError): validate_calibration(c)
  c=self.calibration(); c['units']='millimetres'
  with self.assertRaises(ValueError): validate_calibration(c)
 def test_pair_skew_drop_and_unique(self):
  r=[self.frame(i,i*33333333) for i in range(4)]; d=[self.frame(i,i*33333333+2000000,'depth') for i in [0,2,3]]
  pairs=pair_frames(r,d); self.assertEqual([a.frame_id for a,b in pairs],[0,2,3]); self.assertEqual(len({b.frame_id for a,b in pairs}),3)
 def test_arrival_is_not_exposure(self):
  with self.assertRaises(ValueError): pair_frames([dataclasses.replace(self.frame(0,0),exposure_host_ns=None)],[self.frame(0,0,'depth')])
 def test_nonmonotonic_rejected(self):
  with self.assertRaises(ValueError): pair_frames([self.frame(1,2),self.frame(0,1)],[self.frame(0,0,'depth')])
 def test_lossless_depth_replay_and_integrity(self):
  with tempfile.TemporaryDirectory(dir=str(_P.DATA)+'/work') as tmp:
   p=pathlib.Path(tmp); depth=np.array([[0,1000],[2000,65535]],np.uint16); np.save(p/'depth.npy',depth)
   f=dataclasses.replace(self.frame(0,0,'depth'),payload='depth.npy',sha256=hashlib.sha256((p/'depth.npy').read_bytes()).hexdigest())
   (p/'frames.jsonl').write_text(json.dumps(dataclasses.asdict(f))+'\n'); self.assertEqual(load_replay(p),load_replay(p)); np.testing.assert_array_equal(np.load(p/'depth.npy'),depth)
   (p/'depth.npy').write_bytes(b'corrupt')
   with self.assertRaises(ValueError): load_replay(p)
if __name__=='__main__': unittest.main()
