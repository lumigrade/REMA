import unittest
import numpy as np
from rema.matting import ActorMask,EmptyRoom
class Matting(unittest.TestCase):
 def scene(self):
  a=np.zeros((140,220),np.float32);d=np.full_like(a,4.)
  a[20:130,55:115]=1;d[20:130,55:115]=1.5
  return a,d
 def test_bad_depth_does_not_cut_confident_connected_arm(self):
  a,d=self.scene();a[25:42,110:190]=1
  out,z=ActorMask().apply(a,d,timestamp=1)
  self.assertEqual(out[30,170],1)
 def test_disconnected_hand_retained(self):
  a,d=self.scene();a[40:56,125:144]=.97;d[40:56,125:144]=1.3
  out,z=ActorMask().apply(a,d,timestamp=1)
  self.assertGreater(out[45,130],.9)
 def test_near_hand_does_not_become_whole_actor(self):
  a,d=self.scene();a[40:56,125:144]=.99;d[40:56,125:144]=.6
  out,z=ActorMask().apply(a,d,timestamp=1)
  self.assertEqual(out[60,80],1);self.assertAlmostEqual(z,1.5)
 def test_isolated_background_speck_is_removed(self):
  a,d=self.scene();a[4:6,200:202]=1;d[4:6,200:202]=1.5
  out,z=ActorMask().apply(a,d,timestamp=1);self.assertEqual(out[4,200],0)
 def test_depth_holes_have_finite_grace(self):
  a,d=self.scene();actor=ActorMask();actor.apply(a,d,timestamp=1)
  out,z=actor.apply(a,np.zeros_like(d),timestamp=1.1);self.assertIsNotNone(z)
  out,z=actor.apply(a,np.zeros_like(d),timestamp=1.5);self.assertIsNone(z);self.assertFalse(out.any())
 def test_actor_exit_has_no_stale_mask(self):
  a,d=self.scene();actor=ActorMask();actor.apply(a,d,timestamp=1)
  out,z=actor.apply(np.zeros_like(a),d,timestamp=1.06);self.assertIsNone(z);self.assertFalse(out.any())
 def test_further_person_excluded(self):
  a,d=self.scene();a[20:130,155:215]=1;d[20:130,155:215]=2.8
  out,z=ActorMask().apply(a,d,timestamp=1);self.assertEqual(out[60,80],1);self.assertEqual(out[60,180],0)
 def test_background_agreement_only_removes_uncertain_pixels(self):
  a,d=self.scene();a[35:40,70:75]=.6
  out,z=ActorMask().apply(a,d,timestamp=1,background_match=np.ones_like(a,dtype=bool))
  self.assertEqual(out[37,72],0);self.assertEqual(out[60,80],1)
 def test_empty_reference_refuses_person_and_stays_frozen(self):
  a,d=self.scene();rgb=np.full((140,220,3),100,np.uint8);room=EmptyRoom();room.request(0)
  room.update(rgb,d,a,6);self.assertIsNone(room.rgb)
  for i in range(12):room.update(rgb,d,np.zeros_like(a),7+i*.06)
  self.assertIsNotNone(room.rgb);saved=room.rgb.copy()
  room.update(np.zeros_like(rgb),d,a,20);np.testing.assert_array_equal(room.rgb,saved)
 def test_background_depth_disagreement_preserves_new_surface(self):
  a,d=self.scene();rgb=np.full((140,220,3),100,np.uint8);room=EmptyRoom();room.rgb=rgb.copy();room.depth=d+1
  self.assertFalse(room.match(rgb,d).any())
