import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,cv2
from rema.room_tracking import RoomTracker,floor_from_depth,inverse
from rema import provisional

def config():
 return dict(qualified=False,provisional=True,user_authorized=True,calibration_id='test',K_rgb=[[570.,0,320],[0,570.,240],[0,0,1]],K_depth=[[570.,0,320],[0,570.,240],[0,0,1]],dist_rgb=[0]*5)
class RoomTests(unittest.TestCase):
 def test_permission(self):
  c=config();c['user_authorized']=False
  with self.assertRaises(ValueError):RoomTracker(c)
 def test_provisional_isolation(self):
  with tempfile.TemporaryDirectory() as d,patch.object(provisional,'ROOT',Path(d)):
   active=Path(d)/'active.json';active.write_text('preserve')
   c=provisional.create_provisional();self.assertFalse(c['qualified']);self.assertFalse(c['stereo_pass']);self.assertEqual(active.read_text(),'preserve');self.assertTrue(provisional.load_provisional()['provisional'])
 def test_alignment(self):
  depth=np.ones((480,640),np.float32)*2
  np.testing.assert_array_equal(provisional.align_provisional(depth,config()),depth)
 def test_pose_and_bad_depth(self):
  tr=RoomTracker(config());rng=np.random.default_rng(1)
  world=np.c_[rng.uniform(-1,1,200),rng.uniform(-.7,.7,200),rng.uniform(2,4,200)].astype(np.float32)
  r=np.array([.01,.04,-.01]);t=np.array([.05,-.01,.03]);R=cv2.Rodrigues(r)[0]
  pixels=cv2.projectPoints(world,r,t,tr.K,tr.dist)[0].reshape(-1,2);current=world@R.T+t
  result=tr.solve_matches(world,pixels,current);self.assertIsNotNone(result)
  T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
  np.testing.assert_allclose(result['Tworld_camera'],inverse(T),atol=1e-5)
  self.assertIsNone(tr.solve_matches(world,pixels,current+.3))
 def test_loss_recovery_map_and_actor_exclusion(self):
  rng=np.random.default_rng(5);gray=rng.integers(0,255,(480,640),dtype=np.uint8);rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);depth=np.ones(gray.shape,np.float32)*2
  tr=RoomTracker(config());self.assertFalse(tr.update(rgb,depth,None,1,1000000000)['valid'])
  for i in range(2,203):result=tr.update(rgb,depth,None,i,1000000000+i*50000000)
  self.assertTrue(result['valid']);self.assertLessEqual(len(tr.keyframes),8);self.assertEqual(tr.keyframes[0]['frame_id'],1)
  self.assertEqual(len(tr.extract(rgb,depth,np.ones_like(depth))['pixels']),0)
  self.assertFalse(tr.update(rgb*0,depth,None,203,12000000000)['valid']);self.assertEqual(tr.keyframes[0]['frame_id'],1)
  self.assertFalse(tr.update(rgb,depth,None,204,12050000000)['valid'])
  recovered=tr.update(rgb,depth,None,205,12100000000);self.assertTrue(recovered['valid']);np.testing.assert_allclose(recovered['world_from_camera'],np.eye(4),atol=1e-4)
  self.assertFalse(tr.update(rgb,depth,None,205,12100000000)['valid'])
 def test_flow_uses_current_depth_and_excludes_actor(self):
  rng=np.random.default_rng(8);rgb=rng.integers(0,256,(480,640,3),dtype=np.uint8);d=np.ones((480,640),np.float32)*2;tr=RoomTracker(config())
  for i in range(1,3):tr.update(rgb,d,None,i,i*100000000)
  pose=tr.update(rgb,d,None,3,300000000);self.assertTrue(pose['valid']);self.assertEqual(pose['method'],'optical_flow_rgbd');self.assertEqual(pose['frame_id'],3)
  self.assertIsNone(tr.follow_features(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),d,np.ones_like(d)))
  self.assertFalse(tr.update(rgb,d+1,None,4,400000000)['valid'])
 def test_optical_flow_cannot_hide_descriptor_loss_indefinitely(self):
  rng=np.random.default_rng(9);rgb=rng.integers(0,256,(480,640,3),dtype=np.uint8);d=np.ones((480,640),np.float32)*2;tr=RoomTracker(config())
  for i in range(1,3):tr.update(rgb,d,None,i,i*100000000)
  empty=dict(pixels=np.empty((0,2),np.float32),points=np.empty((0,3),np.float32),descriptors=None)
  with patch.object(tr,'extract',return_value=empty):
   for i in range(3,11):self.assertTrue(tr.update(rgb,d,None,i,i*100000000)['valid'])
   self.assertFalse(tr.update(rgb,d,None,11,1100000000)['valid'])
 def test_floor_rejects_wall(self):
  K=np.array(config()['K_rgb']);y,x=np.indices((480,640));depth=np.zeros((480,640),np.float32)
  np.divide(.5,(y-240)/570,out=depth,where=y>240);depth[(depth>4)|(depth<.35)]=0
  floor=floor_from_depth(depth,K);self.assertIsNotNone(floor);self.assertAlmostEqual(floor[1,3],.5,places=3)
  self.assertIsNone(floor_from_depth(np.ones((480,640),np.float32)*2,K))
if __name__=='__main__':unittest.main()
