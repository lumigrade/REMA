import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import calibrate_app as app
from rema.calibration_guide import CHECK_POSES
import cv2

class WizardPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.root_patch=patch.object(app,'ROOT',self.root);self.root_patch.start()
        self.worker=app.CalibrationWorker()
        self.worker.config={'stereo_pass':True,'qualified':False}
        self.worker.guide.phase='validation'
        self.rgb=(1,1000000,np.zeros((480,640,3),np.uint8))
        self.depth=(2,1000000,np.ones((480,640),np.float32),123)
    def tearDown(self):
        self.root_patch.stop();self.temp.cleanup()
    def test_failed_validation_collects_six_without_restarting_or_activation(self):
        with patch.object(app,'validate_depth',return_value={'passed':False,'edge_p95_px':9.}):
            for i in range(6):
                self.worker.save_check(self.rgb,self.depth)
                self.assertEqual(self.worker.guide.validation_attempt,0)
                self.assertEqual(len(self.worker.checks),i+1)
                if i<5:self.assertEqual(self.worker.guide.phase,'validation')
            self.assertEqual(self.worker.guide.phase,'blocked')
            record=json.loads((self.worker.folder/'depth_checks_set_0_FAILED.json').read_text())
            self.assertEqual(len(record),6);self.assertEqual(len(list(self.worker.folder.glob('depth_check_*.npz'))),6)
            summary=json.loads((self.worker.folder/'validation_summary_set_0.json').read_text())
            self.assertEqual(summary['passed_checks'],0);self.assertFalse(summary['passed'])
            self.assertFalse((self.root/'active.json').exists())
    def test_depth_check_exception_keeps_raw_and_reason(self):
        with patch.object(app,'validate_depth',side_effect=ValueError('No silhouette')):
            self.worker.save_check(self.rgb,self.depth)
        record=json.loads((self.worker.folder/'depth_checks_set_0_FAILED.json').read_text())
        self.assertEqual(record[0]['error'],'No silhouette')
        self.assertTrue((self.worker.folder/'depth_check_000.npz').exists())
    def test_qualified_geometry_can_activate_without_failed_timing_correction(self):
        checks=[]
        for pose in CHECK_POSES:
            tx,ty=pose.tilt;n=cv2.Rodrigues(np.array([-np.radians(ty),np.radians(tx),0.]))[0][:,2]
            checks.append(dict(passed=True,board_range_metres=pose.distance,normal_rgb=n.tolist()))
        self.worker.checks=checks;self.worker.timing_result={'passed':False}
        self.worker.activate()
        result=json.loads((self.root/'active.json').read_text())
        self.assertTrue(result['qualified']);self.assertNotIn('timing_proxy',result)
        self.assertEqual(self.worker.guide.phase,'done')
        self.assertIn('no timing correction',self.worker.guide.feedback)
    def test_incomplete_or_failed_geometry_cannot_activate(self):
        self.worker.activate();self.assertEqual(self.worker.guide.phase,'blocked')
        self.assertFalse((self.root/'active.json').exists())

    def test_hd_raw_and_detection_provenance_saved(self):
        self.worker.rgb_profile=dict(profile_id='test',width=1280,height=720)
        native=np.zeros((720,1280,3),np.uint8);points=np.zeros((12,2),np.float32);ids=np.arange(12)
        rgb=self.rgb+(native,)
        with patch.object(app,'validate_depth',return_value=dict(passed=False,edge_p95_px=5.)) as validator:
            self.worker.save_check(rgb,self.depth,detected=(points,ids))
            self.assertIs(validator.call_args.kwargs['detected'][0],points)
        with np.load(self.worker.folder/'depth_check_000.npz') as raw:
            self.assertEqual(raw['rgb_native'].shape,(720,1280,3))
            self.assertEqual(raw['rgb'].shape,(480,640,3))
            self.assertEqual(json.loads(str(raw['rgb_profile_json']))['profile_id'],'test')
            np.testing.assert_array_equal(raw['rgb_ids'],ids)
