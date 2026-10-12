// CPU-only depth registration. No interpolation or qualification logic.
#include <cmath>
#include <algorithm>
extern "C" void rema_project_depth(const float* depth, const float* rays,
 const double* rotation, const double* translation, const double* camera,
 const double* distortion, float* output) {
 constexpr int width=640, height=480, count=width*height;
 std::fill(output,output+count,0.0f);
 for (int i=0;i<count;++i) {
  const float z=depth[i];
  if (!std::isfinite(z)||z<=.25f||z>=10.f) continue;
  // Match float32 ray*depth arithmetic in the Python reference.
  const float source[3]={rays[3*i]*z,rays[3*i+1]*z,z};
  double p[3];
  for (int j=0;j<3;++j) p[j]=rotation[3*j]*source[0]+rotation[3*j+1]*source[1]+rotation[3*j+2]*source[2]+translation[j];
  if (!std::isfinite(p[2])||p[2]<=.1) continue;
  const double x=p[0]/p[2],y=p[1]/p[2],r2=x*x+y*y;
  const double radial=1+r2*(distortion[0]+r2*(distortion[1]+r2*distortion[4]));
  const double xd=x*radial+2*distortion[2]*x*y+distortion[3]*(r2+2*x*x);
  const double yd=y*radial+distortion[2]*(r2+2*y*y)+2*distortion[3]*x*y;
  const double u=camera[0]*xd+camera[2],v=camera[4]*yd+camera[5];
  if (!std::isfinite(u)||!std::isfinite(v)||u<-.5||u>width-.5||v<-.5||v>height-.5) continue;
  const int px=static_cast<int>(std::nearbyint(u)),py=static_cast<int>(std::nearbyint(v));
  if (px<0||px>=width||py<0||py>=height) continue;
  float& previous=output[py*width+px];
  if (previous==0||p[2]<previous) previous=static_cast<float>(p[2]);
 }
}
