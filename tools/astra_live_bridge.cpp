#include <libobsensor/ObSensor.hpp>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <cstdio>
struct Camera {
 ob::Context context;
 std::shared_ptr<ob::Device> device;
 std::unique_ptr<ob::Pipeline> pipeline;
 Camera(){
  context.enableNetDeviceEnumeration(false);
  auto list=context.queryDeviceList();
  for(unsigned i=0;i<list->deviceCount();i++) if(list->vid(i)==0x2bc5&&list->pid(i)==0x0403) device=list->getDevice(i);
  if(!device)throw std::runtime_error("Astra Pro depth 2bc5:0403 not found");
  pipeline.reset(new ob::Pipeline(device));
  auto profile=pipeline->getStreamProfileList(OB_SENSOR_DEPTH)->getVideoStreamProfile(640,480,OB_FORMAT_Y11,30);
  auto config=std::make_shared<ob::Config>(); config->enableStream(profile);pipeline->start(config);
 }
 ~Camera(){try{pipeline->stop();}catch(...) {}}
};
extern "C" {
void* rema_open(char* error,int size){try{
 ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_ERROR);
 ob::Context::setLoggerToFile(OB_LOG_SEVERITY_WARN,"/home/pc/rema-data/logs/gui");
 return new Camera();
}catch(ob::Error&e){snprintf(error,size,"%s",e.getMessage());}catch(std::exception&e){snprintf(error,size,"%s",e.what());}return nullptr;}
int rema_read(void* ptr,uint16_t* output,uint64_t* id,uint64_t* timestamp,float* scale,char* error,int size){try{
 auto fs=static_cast<Camera*>(ptr)->pipeline->waitForFrames(1000);if(!fs)return 0;
 auto f=fs->depthFrame();if(!f)return 0;
 if(f->width()!=640||f->height()!=480||f->dataSize()!=640*480*2)throw std::runtime_error("Unexpected depth layout");
 memcpy(output,f->data(),640*480*2);*id=f->index();*timestamp=f->timeStampUs();*scale=f->getValueScale();return 1;
}catch(ob::Error&e){snprintf(error,size,"%s",e.getMessage());}catch(std::exception&e){snprintf(error,size,"%s",e.what());}return -1;}
void rema_close(void* ptr){delete static_cast<Camera*>(ptr);}
}
