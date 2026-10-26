import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import importlib.util,json,pathlib,tempfile,unittest
import cv2,numpy as np
spec=importlib.util.spec_from_file_location('analysis','tools/analyze_capture.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class Recording(unittest.TestCase):
 def fixture(self,p):
  (p/'capture.json').write_text(json.dumps({'width':4,'height':2,'calibration_version':'synthetic','model_version':'none','evidence_origin':'synthetic'}));(p/'result.json').write_text('{"success":true}')
  for stream in ['rgb','depth']:
   rows=[]
   for i in [0,1,3]:
    name=f'{stream}{i}.bin' if stream=='depth' else f'{stream}{i}.jpg'
    if stream=='depth':(p/name).write_bytes(np.full((2,4),1000,dtype='<u2').tobytes())
    else:cv2.imwrite(str(p/name),np.zeros((2,4,3),np.uint8))
    r={'frame_id':i,'sensor_timestamp_ns':i*33333333 if stream=='depth' else None,'host_arrival_ns':1000000000+i*33333333,'payload':name,'stride_bytes':8,'height':2,'width':4,'v4l2_flags':0};rows.append(r)
   (p/(stream+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
 def test_integrity_gap_and_unqualified_sync(self):
  with tempfile.TemporaryDirectory(dir=str(_P.DATA)+'/work') as t:
   p=pathlib.Path(t);self.fixture(p);r=mod.analyze(p);self.assertEqual(r['evidence_origin'],'synthetic');self.assertEqual(r['streams']['depth']['sequence_gaps'],1);self.assertFalse(r['sync_qualified']);self.assertIsNone(r['exposure_pair_skew_ns']);self.assertTrue(r['deterministic_replay_passed'])
 def test_truncated_depth_fails(self):
  with tempfile.TemporaryDirectory(dir=str(_P.DATA)+'/work') as t:
   p=pathlib.Path(t);self.fixture(p);(p/'depth1.bin').write_bytes(b'x')
   with self.assertRaises(ValueError):mod.analyze(p)
 def test_failed_capture_remains_failed(self):
  with tempfile.TemporaryDirectory(dir=str(_P.DATA)+'/work') as t:
   p=pathlib.Path(t);self.fixture(p);(p/'result.json').write_text('{"success":false}')
   with self.assertRaises(ValueError):mod.analyze(p)
