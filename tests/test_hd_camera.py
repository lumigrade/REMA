import unittest,cv2,numpy as np
from unittest.mock import patch
from rema.calibration_rgb import prepare_rgb
from rema.live_camera import Cameras

class HDCameraTests(unittest.TestCase):
 def test_live_default_stays_native_640(self):
  self.assertEqual(Cameras().rgb_mode,dict(width=640,height=480,fourcc='MJPG',fps=30))
 def test_hd_points_map_to_calibration_coordinates_and_preserve_source(self):
  im=np.full((720,1280,3),77,np.uint8);before=im.copy()
  profile=dict(width=1280,height=720,H_native_to_calibration=[[.6,0,-64],[0,.6,24],[0,0,1]])
  points=np.array([[640,360],[0,0],[1200,700]],np.float32);ids=np.array([1,2,3])
  with patch('rema.calibration_rgb.detect_robust',return_value=(points,ids)):
   virtual,p,found=prepare_rgb(im,profile,.025)
  self.assertEqual(virtual.shape,(480,640,3));np.testing.assert_array_equal(im,before)
  np.testing.assert_allclose(p,[[320,240]],atol=1e-5);np.testing.assert_array_equal(found,[1])
 def test_wrong_native_resolution_rejected(self):
  with self.assertRaises(ValueError):prepare_rgb(np.zeros((480,640,3),np.uint8),dict(width=1280,height=720),.025)
