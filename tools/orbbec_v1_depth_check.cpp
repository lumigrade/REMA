#include <libobsensor/ObSensor.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cstdint>
int main(){try{
 ob::Context::setLoggerToFile(OB_LOG_SEVERITY_DEBUG,"/home/pc/rema-data/logs/orbbec_v1");
 ob::Context ctx;ctx.enableNetDeviceEnumeration(false);auto list=ctx.queryDeviceList();std::shared_ptr<ob::Device> device;
 for(unsigned i=0;i<list->deviceCount();i++)if(list->vid(i)==0x2bc5&&list->pid(i)==0x0403)device=list->getDevice(i);
 if(!device){std::cerr<<"Exact Astra Pro not found\n";return 3;}
 auto info=device->getDeviceInfo();std::cout<<"Firmware="<<info->firmwareVersion()<<"\n";
 ob::Pipeline pipe(device);auto profiles=pipe.getStreamProfileList(OB_SENSOR_DEPTH);for(unsigned i=0;i<profiles->count();i++){auto p=profiles->getProfile(i)->as<ob::VideoStreamProfile>();std::cout<<"PROFILE "<<p->width()<<"x"<<p->height()<<" fps="<<p->fps()<<" format="<<p->format()<<"\n";} auto profile=profiles->getVideoStreamProfile(640,480,OB_FORMAT_Y11,30);
 auto config=std::make_shared<ob::Config>();config->enableStream(profile);pipe.start(config);
 std::ofstream meta("/home/pc/rema-data/reports/orbbec_v1_depth_frames.csv");meta<<"frame_id,timestamp_us,width,height,scale_mm,nonzero_fraction\n";
 auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);unsigned count=0;
 while(std::chrono::steady_clock::now()<end){auto fs=pipe.waitForFrames(1000);if(!fs)throw std::runtime_error("frames timeout");auto f=fs->depthFrame();if(!f)continue;
 auto data=(uint16_t*)f->data();auto n=f->dataSize()/2;size_t nonzero=0;for(size_t j=0;j<n;j++)nonzero+=(data[j]!=0);
 meta<<f->index()<<","<<f->timeStampUs()<<","<<f->width()<<","<<f->height()<<","<<f->getValueScale()<<","<<double(nonzero)/n<<"\n";count++;
 }
 pipe.stop();std::cout<<"Depth frames="<<count<<"\n";return count?0:4;
}catch(ob::Error&e){std::cerr<<e.getMessage()<<"\n";return 2;}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}
