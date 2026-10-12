import unittest
import numpy as np
from rema.calibration_guide import Pose, movement_cue,mirror_points,Guide
from rema.calibration_coach import selected_edge,mirror_preview,board_points,EDGE_IDS,coach_panel,motion_points

class MirrorGuidanceTests(unittest.TestCase):
    def obs(self,center=(320,240),tilt=(0,0),distance=.8):
        return dict(center=center,tilt=tilt,distance=distance,skew=10)
    def test_screen_movement_directions_follow_mirror(self):
        pose=Pose(.8,(320,240),(0,0))
        for center,expected in [((360,240),'RIGHT'),((280,240),'LEFT'),((320,270),'UP'),((320,210),'DOWN')]:
            cue=movement_cue(pose,self.obs(center=center))
            self.assertEqual(cue['direction'],expected)
    def test_tilt_edges_push_pull_and_overshoot(self):
        for tilt,edge in [((28,0),'LEFT'),((-28,0),'RIGHT'),((0,28),'BOTTOM'),((0,-28),'TOP')]:
            pose=Pose(.8,(320,240),tilt)
            cue=movement_cue(pose,self.obs());self.assertEqual(cue['edge'],edge);self.assertTrue(cue['away'])
            overshoot=np.array(tilt)*2
            cue=movement_cue(pose,self.obs(tilt=overshoot));self.assertEqual(cue['edge'],edge);self.assertFalse(cue['away'])
    def test_only_one_correction_and_cross_axis_leveling(self):
        pose=Pose(.8,(320,240),(28,0))
        self.assertEqual(movement_cue(pose,self.obs(center=(400,400),distance=1))['kind'],'distance')
        self.assertEqual(movement_cue(pose,self.obs(center=(400,400)))['kind'],'position')
        cue=movement_cue(pose,self.obs(tilt=(28,20)))
        self.assertEqual(cue['edge'],'BOTTOM');self.assertFalse(cue['away'])
    def test_display_flip_does_not_modify_raw_or_outline(self):
        raw=np.zeros((480,640,3),np.uint8);raw[250:280,20:30]=[10,20,30];before=raw.copy()
        outline=np.array([[100,100],[300,100],[300,300],[100,300]],float);original=outline.copy()
        output=mirror_preview(raw,None,None,None,{'kind':'wait'})
        np.testing.assert_array_equal(output[250:280,610:620],raw[250:280,20:30][:,::-1])
        np.testing.assert_array_equal(raw,before)
        left=selected_edge(outline,'LEFT');self.assertEqual(left[0][0],339);self.assertEqual(left[1][0],339)
        np.testing.assert_array_equal(outline,original)
        np.testing.assert_allclose(mirror_points(mirror_points(outline)),outline)
    def test_shaded_animation_moves_selected_edge_toward_camera(self):
        for tilt,edge in [((28,0),'LEFT'),((-28,0),'RIGHT'),((0,28),'BOTTOM'),((0,-28),'TOP')]:
            ids=EDGE_IDS[edge];before=board_points((0,0))[list(ids),2].mean();after=board_points(tilt)[list(ids),2].mean()
            self.assertGreater(after,before) # scene z means away from user, toward camera
    def test_resume_countdown_keeps_saved_pose_index(self):
        guide=Guide();guide.start(0,index=1);guide.update({},10)
        self.assertEqual(guide.index,1);self.assertEqual(guide.phase,'capture')
    def test_all_cue_panels_render_at_native_aspect(self):
        pose=Pose(.8,(320,240),(28,0))
        for obs in [self.obs(distance=1),self.obs(center=(400,240)),self.obs(),self.obs(tilt=(28,0)),{'problem':'Target lost'}]:
            cue=movement_cue(pose,obs);im=coach_panel(pose,cue,obs,.6)
            self.assertEqual(im.shape,(480,640,3));self.assertGreater(im.std(),20)

    def test_motion_example_holds_opposite_edge_fixed(self):
        before=board_points((28,0));after=motion_points((-28,0),(28,0),'RIGHT')
        np.testing.assert_allclose(before[list(EDGE_IDS['LEFT'])].mean(0),after[list(EDGE_IDS['LEFT'])].mean(0))
        self.assertGreater(after[list(EDGE_IDS['RIGHT']),2].mean(),before[list(EDGE_IDS['RIGHT']),2].mean())
