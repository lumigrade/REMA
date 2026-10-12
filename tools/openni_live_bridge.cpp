#include <OpenNI.h>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
void check(openni::Status result){if(result!=openni::STATUS_OK)throw std::runtime_error(openni::OpenNI::getExtendedError());}
struct Camera {
 openni::Device device;openni::VideoStream stream;bool infrared;
 Camera(bool ir=false):infrared(ir){
  openni::OpenNI::setLogMinSeverity(3);check(openni::OpenNI::initialize());
  openni::Array<openni::DeviceInfo> list;openni::OpenNI::enumerateDevices(&list);std::string uri;
  for(int i=0;i<list.getSize();i++)if(list[i].getUsbVendorId()==0x2bc5&&list[i].getUsbProductId()==0x0403){if(!uri.empty())throw std::runtime_error("Multiple Astra devices");uri=list[i].getUri();}
  if(uri.empty())throw std::runtime_error("Astra Pro depth 2bc5:0403 not found");
  check(device.open(uri.c_str()));check(stream.create(device,ir?openni::SENSOR_IR:openni::SENSOR_DEPTH));
  openni::VideoMode mode;mode.setResolution(640,480);mode.setFps(30);mode.setPixelFormat(ir?openni::PIXEL_FORMAT_GRAY16:openni::PIXEL_FORMAT_DEPTH_1_MM);
  check(stream.setVideoMode(mode));check(stream.setMirroringEnabled(false));check(stream.start());
  // The depth stream owns emitter startup. On this Astra firmware, writing
  // OBEXTENSION_ID_LASER_EN while depth streams returns success but zeros depth.
  // Never issue that control in the normal metric-depth path.
 }
 ~Camera(){if(infrared)device.setProperty(openni::OBEXTENSION_ID_LASER_EN,true);stream.stop();stream.destroy();device.close();openni::OpenNI::shutdown();}
};
extern "C" {
void* rema_open(char* error,int size){try{return new Camera();}catch(std::exception&e){snprintf(error,size,"%s",e.what());return nullptr;}}
int rema_read(void* ptr,uint16_t* output,uint64_t* id,uint64_t* timestamp,float* scale,char* error,int size){try{
 auto camera=static_cast<Camera*>(ptr);openni::VideoStream* streams[]={&camera->stream};int index;
 auto status=openni::OpenNI::waitForAnyStream(streams,1,&index,1000);if(status==openni::STATUS_TIME_OUT)return 0;check(status);
 openni::VideoFrameRef frame;check(camera->stream.readFrame(&frame));
 if(!frame.isValid()||frame.getWidth()!=640||frame.getHeight()!=480||frame.getVideoMode().getPixelFormat()!=(camera->infrared?openni::PIXEL_FORMAT_GRAY16:openni::PIXEL_FORMAT_DEPTH_1_MM))throw std::runtime_error("Unexpected depth layout");
 for(int y=0;y<480;y++)memcpy(output+y*640,static_cast<const char*>(frame.getData())+y*frame.getStrideInBytes(),640*2);
 *id=frame.getFrameIndex();*timestamp=frame.getTimestamp();*scale=1;return 1;
}catch(std::exception&e){snprintf(error,size,"%s",e.what());return -1;}}
void* rema_ir_open(char* error,int size){try{return new Camera(true);}catch(std::exception&e){snprintf(error,size,"%s",e.what());return nullptr;}}
int rema_projector(void* ptr,int enabled){auto camera=static_cast<Camera*>(ptr);if(!camera->infrared)return openni::STATUS_NOT_SUPPORTED;return camera->device.setProperty(openni::OBEXTENSION_ID_LASER_EN,bool(enabled));}
int rema_intrinsics(void* ptr,float* output){
 auto c=static_cast<Camera*>(ptr);if(c->infrared)return 4;
 float x0,y0,z0,x1,y1,z1;
 auto a=openni::CoordinateConverter::convertDepthToWorld(c->stream,0,0,1000,&x0,&y0,&z0);
 auto b=openni::CoordinateConverter::convertDepthToWorld(c->stream,639,479,1000,&x1,&y1,&z1);
 if(a!=openni::STATUS_OK||b!=openni::STATUS_OK)return 1;
 output[0]=639000.f/(x1-x0);output[1]=-479000.f/(y1-y0);
 output[2]=-x0*output[0]/1000;output[3]=y0*output[1]/1000;return 0;
}
void rema_close(void* ptr){delete static_cast<Camera*>(ptr);}
}
