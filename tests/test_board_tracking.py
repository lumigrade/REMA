import unittest
from unittest.mock import patch
import cv2,numpy as np
from rema.board_tracking import FloorTargetTracker,tracked_scene
from rema.calibration import board
from rema.live_core import composite

class FloorTrackingTests(unittest.TestCase):
    def setUp(self):
        self.config=dict(qualified=True,stereo_pass=True,depth_validation_pass=True,
             K_rgb=[[610.,0,320],[0,608.,240],[0,0,1]],dist_rgb=[0,0,0,0,0],square_metres=.02317,
             calibration_id='synthetic-test-only')
        self.tracker=FloorTargetTracker(self.config);self.obj=board(.02317).getChessboardCorners();self.ids=np.arange(len(self.obj))
        self.r=np.array([.55,-.17,.06]);self.t=np.array([-.10,-.08,.8])
    def points(self,r=None,t=None):
        return cv2.projectPoints(self.obj,self.r if r is None else r,self.t if t is None else t,self.tracker.K,self.tracker.dist)[0].reshape(-1,2)
    def lock(self):
        self.tracker.update_points(self.points(),self.ids,1,1_000_000_000)
        return self.tracker.update_points(self.points(),self.ids,2,1_033_000_000)
    def test_requires_qualified_calibration(self):
        bad=dict(self.config,qualified=False)
        with self.assertRaises(ValueError):FloorTargetTracker(bad)
    def test_recovers_world_camera_transform_and_same_frame_timestamp(self):
        pose=self.lock();self.assertTrue(pose['valid'],pose)
        np.testing.assert_allclose(pose['tvec_metres'],self.t,atol=1e-5)
        R=cv2.Rodrigues(self.r)[0];np.testing.assert_allclose(np.array(pose['world_from_camera'])[:3,3],-R.T@self.t,atol=1e-5)
        self.assertEqual(pose['frame_id'],2);self.assertEqual(pose['rgb_host_arrival_ns'],1_033_000_000)
    def test_target_loss_and_duplicate_frames_never_reuse_pose(self):
        self.assertTrue(self.lock()['valid'])
        lost=self.tracker.update_points(np.empty((0,2)),np.array([],int),3,1_066_000_000)
        self.assertFalse(lost['valid']);self.assertNotIn('rvec',lost)
        reacquire=self.tracker.update_points(self.points(),self.ids,4,1_099_000_000);self.assertFalse(reacquire['valid'])
        self.assertTrue(self.tracker.update_points(self.points(),self.ids,5,1_132_000_000)['valid'])
        repeated=self.tracker.update_points(self.points(),self.ids,5,1_132_000_000)
        self.assertFalse(repeated['valid']);self.assertNotIn('rvec',repeated)
    def test_actor_mask_excludes_target_corners(self):
        with patch('rema.board_tracking.detect',return_value=(self.points(),self.ids)):
            result=self.tracker.update(np.zeros((480,640,3),np.uint8),np.ones((480,640),np.float32),1,1_000_000_000)
        self.assertFalse(result['valid']);self.assertEqual(result['corners'],0)
    def test_sudden_jump_is_rejected_and_bounded_reacquisition(self):
        self.lock();moved=self.points(t=np.array([.18,-.08,.8]))
        result=self.tracker.update_points(moved,self.ids,3,1_066_000_000)
        self.assertFalse(result['valid']);self.assertIn('jump',result['reason'])
        self.assertFalse(self.tracker.update_points(moved,self.ids,4,1_099_000_000)['valid'])
        self.assertTrue(self.tracker.update_points(moved,self.ids,5,1_132_000_000)['valid'])
    def test_corrupted_corner_geometry_rejected(self):
        rng=np.random.default_rng(10);points=self.points()+rng.normal(0,12,(54,2)).astype(np.float32)
        result=self.tracker.update_points(points,self.ids,1,1_000_000_000)
        self.assertFalse(result['valid'])
    def test_rendering_moves_with_pose_and_keeps_opaque_actor_pixels(self):
        pose=self.lock();scene=tracked_scene(pose,self.tracker.K,self.tracker.dist)
        second=dict(pose,tvec_metres=[-.05,-.08,.8]);moved=tracked_scene(second,self.tracker.K,self.tracker.dist)
        self.assertGreater(np.mean(scene!=moved),.02)
        raw=np.full((480,640,3),[30,90,180],np.uint8);alpha=np.zeros((480,640),np.float32);alpha[100:300,200:400]=1
        output=composite(raw,alpha,scene);np.testing.assert_array_equal(output[150:250,250:350],raw[150:250,250:350])
        with self.assertRaises(ValueError):tracked_scene({'valid':False},self.tracker.K,self.tracker.dist)
    def test_synthetic_motion_has_no_accumulated_pose_drift(self):
        rng=np.random.default_rng(18);errors=[]
        for i in range(121):
            t=self.t+np.array([.08*np.sin(i*.05),.03*np.sin(i*.07),.04*np.cos(i*.04)])
            points=self.points(t=t)+rng.normal(0,.08,(54,2)).astype(np.float32)
            result=self.tracker.update_points(points,self.ids,i+1,int(1e9+i*100_000_000))
            if result['valid']:errors.append(np.linalg.norm(np.array(result['tvec_metres'])-t))
        self.assertGreater(len(errors),110);self.assertLess(max(errors),.004)
