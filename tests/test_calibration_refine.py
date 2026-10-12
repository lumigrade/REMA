import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
cv2.setNumThreads(2)
from rema.calibration_refine import fit_rigid_edges
from rema.calibration import Projection
from rema.provisional import align_provisional,load_provisional

class RefinementTests(unittest.TestCase):
 def config(self):
  K=[[580.,0,320],[0,580.,240],[0,0,1]]
  return dict(K_rgb=K,K_depth=K,dist_rgb=[0.]*5,dist_depth=[0.]*5,R_depth_to_rgb=np.eye(3).tolist(),t_depth_to_rgb_metres=[0.,0.,0.],provisional=True,qualified=False,user_authorized=True,schema='rema_provisional_rgbd_v1')
 def test_recovers_known_rigid_transform_without_qualification(self):
  c=self.config();truthR=cv2.Rodrigues(np.array([.006,-.009,.003]))[0];truthT=np.array([-.024,.002,.001]);observations=[];planes=[]
  polygon=np.array([[-.15,-.105,0],[.15,-.105,0],[.15,.105,0],[-.15,.105,0.]])
  for i in range(6):
   r=cv2.Rodrigues(np.array([(-1)**i*.3,(-1)**(i//2)*.25,.04*i]))[0];t=np.array([-.08+.025*i,.03,.6+.1*i]);poly=polygon@r.T+t
   edges=np.concatenate([poly[j]+np.linspace(0,1,80)[:,None]*(poly[(j+1)%4]-poly[j]) for j in range(4)])
   rgbpoly=poly@truthR.T+truthT;uv,_=cv2.projectPoints(rgbpoly,np.zeros(3),np.zeros(3),np.array(c['K_rgb']),np.zeros(5))
   observations.append(dict(edge_points=edges,polygon=uv.reshape(4,2)))
   planes.append((poly,r[:,2]@truthR.T,t@truthR.T+truthT))
  original=copy.deepcopy(c);out,metrics=fit_rigid_edges(c,observations,planes)
  self.assertEqual(c,original);self.assertFalse(out['qualified']);self.assertFalse(out['depth_validation_pass'])
  np.testing.assert_allclose(out['t_depth_to_rgb_metres'],truthT,atol=.0002)
  np.testing.assert_allclose(out['R_depth_to_rgb'],truthR,atol=.0002)
  self.assertLess(metrics['after_cost'],metrics['before_cost']*.001)
 def test_refined_provisional_projects_depth_dependent_parallax(self):
  c=self.config();c.update(alignment_model='rigid_reprojection_v1',t_depth_to_rgb_metres=[.03,0,0])
  depth=np.zeros((480,640),np.float32);depth[220,300]=1;depth[260,300]=2
  result=align_provisional(depth,c,Projection(c))
  self.assertAlmostEqual(result[220,317],1);self.assertAlmostEqual(result[260,309],2)
  self.assertEqual(result[220,300],0)
  legacy=self.config();np.testing.assert_array_equal(align_provisional(depth,legacy),depth)
 def test_nonidentity_geometry_cannot_use_homography_fallback(self):
  c=self.config();c['t_depth_to_rgb_metres']=[.03,0,0]
  with self.assertRaises(ValueError):align_provisional(np.ones((480,640),np.float32),c)
 def test_invalid_rigid_transform_rejected(self):
  c=self.config();c.update(alignment_model='rigid_reprojection_v1',R_depth_to_rgb=np.zeros((3,3)).tolist())
  with tempfile.TemporaryDirectory() as folder:
   (Path(folder)/'provisional.json').write_text(json.dumps(c))
   with patch('rema.provisional.ROOT',Path(folder)):
    with self.assertRaises(ValueError):load_provisional()

 def test_dense_projection_matches_opencv_with_distortion(self):
  c=self.config();c['dist_rgb']=[.03,-.02,.001,-.002,.006];c['R_depth_to_rgb']=cv2.Rodrigues(np.array([.02,-.01,.03]))[0].tolist();c['t_depth_to_rgb_metres']=[-.025,.003,.001]
  d=np.zeros((480,640),np.float32);rng=np.random.default_rng(9);d[::13,::11]=rng.uniform(.4,4,d[::13,::11].shape)
  uv,z,points=Projection(c).project_points(d)
  expected,_=cv2.projectPoints(points,np.zeros(3),np.zeros(3),np.array(c['K_rgb']),np.array(c['dist_rgb']))
  np.testing.assert_allclose(uv,expected.reshape(-1,2),atol=1e-9)
 def test_hole_fill_preserves_observations_and_depth_edges(self):
  from rema.provisional import fill_small_depth_holes
  d=np.ones((20,20),np.float32);d[:,10:]=2;d[5,5]=0;d[5,9]=0;d[12:17,2:7]=0
  out=fill_small_depth_holes(d)
  self.assertEqual(out[5,5],1);self.assertEqual(out[5,9],0);self.assertEqual(out[14,4],0)
  np.testing.assert_array_equal(out[d>0],d[d>0])

 def test_native_registration_matches_reference_including_invalid_depth(self):
  import rema.depth_projection as native
  if native._library is None:self.skipTest('Optional CPU library not built')
  c=self.config();c['dist_rgb']=[.03,-.02,.001,-.002,.006];c['R_depth_to_rgb']=cv2.Rodrigues(np.array([.02,-.01,.03]))[0].tolist();c['t_depth_to_rgb_metres']=[-.025,.003,.001]
  rng=np.random.default_rng(7);d=rng.uniform(.2,4,(480,640)).astype(np.float32);d[::7]=0;d[2,2]=np.nan;d[3,3]=np.inf
  projection=Projection(c);actual=projection.align(d)
  with patch.object(native,'_library',None):reference=projection.align(d)
  np.testing.assert_array_equal(actual>0,reference>0)
  np.testing.assert_allclose(actual,reference,atol=1e-6,rtol=0)
