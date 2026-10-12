import unittest
import cv2,numpy as np
from rema.calibration_overlay import filled_overlay,target_polygon,estimate_guide_K
from rema.calibration_guide import Pose
from rema.calibration import board

class FilledOverlayTests(unittest.TestCase):
    def test_uniform_opacity_and_cycle_without_raw_mutation(self):
        raw=np.full((480,640,3),100,np.uint8);saved=raw.copy()
        polygon=np.float32([[100,100],[200,100],[200,200],[100,200]])
        expected=[[60,60,162],[60,162,60],[162,60,60],[162,162,162]]
        for phase,want in enumerate(expected):
            out=filled_overlay(raw,polygon,phase,mirror=False)
            np.testing.assert_array_equal(out[100,100],want)
            np.testing.assert_array_equal(out[50,50],raw[50,50])
            np.testing.assert_array_equal(out,filled_overlay(raw,polygon,phase+4,mirror=False))
        np.testing.assert_array_equal(raw,saved)
    def test_native_polygon_is_mirrored_only_at_display(self):
        raw=np.zeros((480,640,3),np.uint8);p=np.float32([[10,100],[50,100],[50,150],[10,150]])
        image=filled_overlay(raw,p,0)
        self.assertTrue(np.any(image[120,610]));self.assertFalse(np.any(image[120,30]))
        self.assertEqual(p[0,0],10)
    def test_target_uses_requested_pose_not_observed_board(self):
        K=np.array([[600.,0,320],[0,600.,240],[0,0,1]])
        near=target_polygon(Pose(.6,(320,240),(0,0)),K)
        far=target_polygon(Pose(1.2,(320,240),(0,0)),K)
        np.testing.assert_allclose(np.ptp(near,axis=0),2*np.ptp(far,axis=0),atol=.001)
        angled=target_polygon(Pose(.6,(320,240),(28,0)),K)
        self.assertGreater(np.linalg.norm(angled[2]-angled[1]),np.linalg.norm(angled[3]-angled[0]))
    def test_provisional_focal_estimate_uses_metric_homography(self):
        obj=board(.02317).getChessboardCorners();K=np.array([[610.,0,320],[0,590.,240],[0,0,1]])
        rot=cv2.Rodrigues(np.array([.13,.25,0.]))[0]
        center=np.array([5*.02317,3.5*.02317,0.]);t=np.array([0.,0.,.8])-rot@center
        rgb,_=cv2.projectPoints(obj,cv2.Rodrigues(rot)[0],t,K,np.zeros(5))
        points=obj@rot.T+t
        # Symmetric corners have median z equal to the physical centre z.
        sample=dict(obj=obj,rgb=rgb.reshape(-1,2),square_metres=.02317,plane_points=points)
        estimated=estimate_guide_K([sample]);np.testing.assert_allclose(estimated,K,atol=.1)
