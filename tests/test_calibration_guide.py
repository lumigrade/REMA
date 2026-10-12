import unittest
import numpy as np
import cv2
from rema.calibration_guide import Guide,FIT_POSES,CHECK_POSES,rgb_outline,auto_board_plane
from rema.calibration import board,solve_planes,depth_validation_summary

class GuideTests(unittest.TestCase):
    def observation(self,pose):
        return dict(distance=pose.distance,center=np.array(pose.center),tilt=pose.tilt,skew=10,
                    corners={i:np.array([float(i%6),float(i//6)]) for i in range(24)})

    def ready(self):
        g=Guide();g.start(0);g.update({},10);return g

    def test_countdown_and_no_capture_before_start(self):
        g=Guide();self.assertIsNone(g.update({},100));self.assertEqual(g.phase,'ready')
        g.start(100);g.update({},109);self.assertEqual(g.phase,'countdown')
        g.update({},110);self.assertEqual(g.phase,'capture')

    def test_wrong_pose_loss_and_motion_reset_hold(self):
        g=self.ready();obs=self.observation(g.pose)
        self.assertIsNone(g.update(obs,11));self.assertIsNone(g.update(obs,12))
        g.update({'problem':'Board lost'},12.1);self.assertEqual(g.progress,0)
        g.update(obs,13);moved=dict(obs,corners={i:p+3 for i,p in obs['corners'].items()})
        self.assertIsNone(g.update(moved,14.4));self.assertEqual(g.progress,0)
        wrong=dict(moved,distance=2)
        self.assertIsNone(g.update(wrong,16));self.assertIn('CLOSER',g.feedback)
        self.assertIsNone(g.update(dict(obs,skew=81),20));self.assertEqual(g.progress,0)
        self.assertIsNone(g.update(dict(obs,tilt=(-28,0)),21));self.assertEqual(g.progress,0)

    def test_capture_fires_once_and_pause_resets_hold(self):
        g=self.ready();obs=self.observation(g.pose)
        g.update(obs,11);self.assertEqual(g.update(obs,12.5),'save')
        self.assertIsNone(g.update(obs,14));g.captured(14)
        obs=self.observation(g.pose);g.update(obs,17);g.pause(18)
        self.assertEqual(g.phase,'paused');self.assertEqual(g.progress,0)
        g.update(obs,20);self.assertEqual(g.phase,'paused');g.pause(21)
        self.assertIsNone(g.update(obs,22));self.assertIsNone(g.update(obs,24))
        self.assertEqual(g.update(obs,25.5),'save')

    def test_complete_automatic_sequence_and_rejection(self):
        g=self.ready();now=20
        for i in range(24):
            self.assertEqual(g.index,i);obs=self.observation(g.pose)
            g.update(obs,now);self.assertEqual(g.update(obs,now+1.5),'save')
            action=g.captured(now+1.5);now+=5
        self.assertEqual(action,'solve');self.assertEqual(g.phase,'solving')
        g.solved(True,now);now+=7
        for i in range(6):
            obs=self.observation(g.pose);g.update(obs,now)
            self.assertEqual(g.update(obs,now+1.5),'check')
            action=g.captured(now+1.5);now+=5
        self.assertEqual(action,'validation_done');g.validated(True,now)
        self.assertIsNone(g.update({},now+7));self.assertEqual(g.update({},now+8),'timing')
        self.assertEqual(g.deadline,now+20)
        bad=Guide();bad.solved(False,0,'bad fit');self.assertEqual(bad.phase,'blocked')
        bad.validated(False,0,'bad outline');self.assertEqual(bad.phase,'blocked')

    def test_auto_depth_seed_finds_board_and_rejects_wall(self):
        depth=np.full((480,640),2.,np.float32)
        depth[179:301,234:406]=1.
        result=auto_board_plane(depth,np.array([300,240]),[580,578,320,240])
        self.assertAlmostEqual(result['range_m'],1,places=3)
        with self.assertRaises(ValueError):
            auto_board_plane(np.ones((480,640),np.float32),np.array([320,240]),[580,578,320,240])

    def test_guided_pose_schedule_passes_geometry_and_validation_variety(self):
        obj=board().getChessboardCorners();K=np.array([[605.,0,320],[0,607.,240],[0,0,1]])
        samples=[];checks=[];translation=np.array([.025,.002,.001])
        for pose in FIT_POSES:
            tx,ty=pose.tilt;rotation=cv2.Rodrigues(np.array([-np.radians(ty),np.radians(tx),0.]))[0]
            center=np.array([(pose.center[0]-320)*pose.distance/605,(pose.center[1]-240)*pose.distance/607,pose.distance])
            t=center-rotation@np.array([.125,.0875,0]);points=obj@rotation.T+t
            rgb,_=cv2.projectPoints(points+translation,np.zeros(3),np.zeros(3),K,np.zeros(5))
            samples.append(dict(obj=obj,rgb=rgb.reshape(-1,2),plane_normal=rotation[:,2],plane_offset=-rotation[:,2]@t,plane_points=points.astype(np.float32)))
            center2,outline=rgb_outline(rgb.reshape(-1,2),np.arange(54),.025)
            self.assertTrue(np.isfinite(center2).all())
        result=solve_planes(samples,[580,578,320,240]);self.assertTrue(result['stereo_pass'],result)
        self.assertGreaterEqual(result['coverage_cells'],4)
        np.testing.assert_allclose(result['t_depth_to_rgb_metres'],translation,atol=.002)
        for pose in CHECK_POSES:
            tx,ty=pose.tilt;n=cv2.Rodrigues(np.array([-np.radians(ty),np.radians(tx),0.]))[0][:,2]
            checks.append(dict(passed=True,board_range_metres=pose.distance,normal_rgb=n))
        self.assertTrue(depth_validation_summary(checks)[0])
