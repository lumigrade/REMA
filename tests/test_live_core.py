import unittest
import numpy as np
from rema.live_core import select_person,composite,align_depth
class LiveSelection(unittest.TestCase):
 def scene(self):
  p=np.zeros((120,180),np.float32);p[10:110,10:65]=1;p[10:110,110:165]=1
  d=np.full_like(p,4);d[10:110,10:65]=2.5;d[10:110,110:165]=1.2
  return p,d
 def test_nearest_person_excludes_far_person_and_furniture(self):
  p,d=self.scene();d[:,75:100]=.5;a,z=select_person(p,d)
  self.assertAlmostEqual(z,1.2,5);self.assertEqual(a[60,130],1);self.assertEqual(a[60,30],0);self.assertEqual(a[60,85],0)
 def test_no_depth_cannot_select_person(self):
  p,d=self.scene();a,z=select_person(p,np.zeros_like(d));self.assertIsNone(z);self.assertFalse(a.any())
 def test_missing_depth_inside_selected_body_is_preserved(self):
  p,d=self.scene();d[50:65,125:145]=0;a,z=select_person(p,d);self.assertEqual(a[55,130],1)
 def test_foreground_pixels_preserved_exactly(self):
  rng=np.random.default_rng(4);rgb=rng.integers(0,256,(120,180,3),dtype=np.uint8);bg=np.zeros_like(rgb);p,d=self.scene();a,z=select_person(p,d);out=composite(rgb,a,bg)
  np.testing.assert_array_equal(out[a==1],rgb[a==1]);np.testing.assert_array_equal(out[a==0],bg[a==0])
 def test_alignment_keeps_depth_units_and_invalid_edges(self):
  d=np.full((10,20),1.234,np.float32);out=align_depth(d,(10,20),1,3,0)
  self.assertTrue((out[:,:3]==0).all());np.testing.assert_array_equal(out[:,3:],d[:,3:])
