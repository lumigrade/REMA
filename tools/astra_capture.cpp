#include <OpenNI.h>
#include <PS1080.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <stdexcept>
using namespace std;
namespace fs=std::filesystem;
uint64_t now_ns(){return chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now().time_since_epoch()).count();}
void ck(bool b,const string& s){if(!b) throw runtime_error(s+": "+openni::OpenNI::getExtendedError());}
void ioc(int fd,unsigned long cmd,void* arg){if(ioctl(fd,cmd,arg)<0)throw runtime_error("V4L2 ioctl failed: "+string(strerror(errno)));}
void save(const fs::path& p,const void* v,size_t n){if(fs::exists(p))throw runtime_error("refusing duplicate payload: "+p.string());ofstream f(p,ios::binary);f.exceptions(ios::badbit|ios::failbit);f.write((const char*)v,n);}
void floats(ostream& o,const float* p,int n){o<<"[";for(int i=0;i<n;i++){if(i)o<<",";if(std::isfinite(p[i]))o<<p[i];else o<<"null";}o<<"]";}
int main(int argc,char** argv){
 fs::path out;bool created_out=false;
 try{
  if(argc<3)throw runtime_error("usage: astra_capture OUTPUT_DIR SECONDS [RGB_DEVICE] ; 0 seconds = inventory only");
  out=argv[1]; int seconds=stoi(argv[2]);ck(seconds>=0&&seconds<=300,"duration range 0..300");
  ck(!fs::exists(out),"refusing existing output");fs::create_directories(out);created_out=true;
  openni::OpenNI::setLogOutputFolder("/home/pc/rema-data/logs");openni::OpenNI::setLogMinSeverity(0);openni::OpenNI::setLogFileOutput(true);ck(openni::OpenNI::initialize()==openni::STATUS_OK,"OpenNI initialize");
  openni::Array<openni::DeviceInfo> list;openni::OpenNI::enumerateDevices(&list);string uri;
  for(int i=0;i<list.getSize();i++)if(list[i].getUsbVendorId()==0x2bc5&&list[i].getUsbProductId()==0x0403){ck(uri.empty(),"ambiguous multiple Astra devices");uri=list[i].getUri();}
  ck(!uri.empty(),"Astra Pro depth not found");openni::Device dev;ck(dev.open(uri.c_str())==openni::STATUS_OK,"Astra depth open");
  ofstream inv(out/"device.json");inv<<"{\n\"usb_id\":\"2bc5:0403\",\"firmware\":";
  XnVersions version={};int size=sizeof(version);auto rc=dev.getProperty(XN_MODULE_PROPERTY_VERSION,&version,&size);
  if(rc==openni::STATUS_OK) inv<<"\""<<int(version.nMajor)<<"."<<int(version.nMinor)<<"."<<version.nBuild<<"\"";else inv<<"null";
  char serial[256]={};size=sizeof(serial);rc=dev.getProperty(ONI_DEVICE_PROPERTY_SERIAL_NUMBER,serial,&size);if(rc==openni::STATUS_OK)save(out/"serial.local.txt",serial,strnlen(serial,sizeof(serial)));
  inv<<",\n\"sensor_modes\":[";bool comma=false;
  for(auto type:{openni::SENSOR_DEPTH,openni::SENSOR_IR,openni::SENSOR_COLOR}){auto info=dev.getSensorInfo(type);if(!info)continue;auto& modes=info->getSupportedVideoModes();for(int i=0;i<modes.getSize();i++){auto& m=modes[i];if(comma)inv<<",";comma=true;inv<<"{\"sensor\":"<<int(type)<<",\"width\":"<<m.getResolutionX()<<",\"height\":"<<m.getResolutionY()<<",\"fps\":"<<m.getFps()<<",\"pixel_format\":"<<int(m.getPixelFormat())<<"}";}}
  OBCameraParams params={};size=sizeof(params);rc=dev.getProperty(openni::OBEXTENSION_ID_CAM_PARAMS,&params,&size);
  inv<<"],\n\"factory_params_status\":"<<int(rc)<<",\"factory_params_raw\":";
  if(rc==openni::STATUS_OK){save(out/"factory_params.bin",&params,sizeof(params));inv<<"{\"l_intr_p\":";floats(inv,params.l_intr_p,4);inv<<",\"r_intr_p\":";floats(inv,params.r_intr_p,4);inv<<",\"r2l_r\":";floats(inv,params.r2l_r,9);inv<<",\"r2l_t\":";floats(inv,params.r2l_t,3);inv<<",\"l_k\":";floats(inv,params.l_k,5);inv<<",\"r_k\":";floats(inv,params.r_k,5);inv<<"}";}else inv<<"null";
  inv<<",\n\"factory_calibration_qualified\":false,\"factory_translation_units_verified\":false,\"exposure_alignment_qualified\":false}\n";inv.close();
  if(!seconds){dev.close();openni::OpenNI::shutdown();cout<<"Inventory saved; no stream started\n";return 0;}
  openni::VideoStream ds;ck(ds.create(dev,openni::SENSOR_DEPTH)==openni::STATUS_OK,"create depth");bool found=false;auto& modes=ds.getSensorInfo().getSupportedVideoModes();for(int i=0;i<modes.getSize();i++){auto m=modes[i];if(m.getResolutionX()==640&&m.getResolutionY()==480&&m.getFps()==30&&m.getPixelFormat()==openni::PIXEL_FORMAT_DEPTH_1_MM){ck(ds.setVideoMode(m)==openni::STATUS_OK,"depth mode");found=true;break;}}ck(found,"no advertised 640x480/30 1mm depth mode");ck(ds.setMirroringEnabled(false)==openni::STATUS_OK,"depth unmirror");
  string video=argc>3?argv[3]:"/dev/video2";fs::path sys=fs::path("/sys/class/video4linux")/fs::path(video).filename()/"device";auto parent=fs::canonical(sys);bool astra=false;while(parent!=parent.root_path()){if(fs::exists(parent/"idVendor")){string v,p;ifstream(parent/"idVendor")>>v;ifstream(parent/"idProduct")>>p;astra=v=="2bc5"&&p=="0501";break;}parent=parent.parent_path();}ck(astra,"RGB device is not Astra 2bc5:0501");
  int fd=open(video.c_str(),O_RDWR|O_NONBLOCK);ck(fd>=0,"RGB open");v4l2_format fmt={};fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;fmt.fmt.pix.width=640;fmt.fmt.pix.height=480;fmt.fmt.pix.pixelformat=V4L2_PIX_FMT_MJPEG;fmt.fmt.pix.field=V4L2_FIELD_ANY;ioc(fd,VIDIOC_S_FMT,&fmt);ck(fmt.fmt.pix.width==640&&fmt.fmt.pix.height==480&&fmt.fmt.pix.pixelformat==V4L2_PIX_FMT_MJPEG,"RGB negotiated mode mismatch");
  v4l2_streamparm sp={};sp.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;sp.parm.capture.timeperframe={1,30};ioc(fd,VIDIOC_S_PARM,&sp);ck(sp.parm.capture.timeperframe.denominator==30*sp.parm.capture.timeperframe.numerator,"RGB FPS mismatch");
  v4l2_requestbuffers req={};req.count=4;req.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;req.memory=V4L2_MEMORY_MMAP;ioc(fd,VIDIOC_REQBUFS,&req);vector<void*> ptr(req.count);vector<size_t> length(req.count);
  for(unsigned i=0;i<req.count;i++){v4l2_buffer b={};b.index=i;b.type=req.type;b.memory=req.memory;ioc(fd,VIDIOC_QUERYBUF,&b);ptr[i]=mmap(nullptr,b.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,b.m.offset);ck(ptr[i]!=MAP_FAILED,"mmap RGB");length[i]=b.length;ioc(fd,VIDIOC_QBUF,&b);}
  fs::create_directory(out/"depth");fs::create_directory(out/"rgb");
  ofstream config(out/"capture.json");config<<"{\"evidence_origin\":\"live_astra\",\"usb_compatibility\":\"language_descriptor_full_read_v1\",\"width\":640,\"height\":480,\"fps_requested\":30,\"depth_pixel_format\":100,\"depth_scale_metres_per_unit\":0.001,\"depth_storage\":\"little_endian_uint16_rows\",\"RGB_format\":\"MJPG\",\"RGB_device\":\""<<video<<"\",\"RGB_colorspace\":"<<fmt.fmt.pix.colorspace<<",\"RGB_ycbcr_enc\":"<<fmt.fmt.pix.ycbcr_enc<<",\"RGB_quantization\":"<<fmt.fmt.pix.quantization<<",\"RGB_xfer_func\":"<<fmt.fmt.pix.xfer_func<<",\"depth_invalid_policy\":\"zero_invalid_other_codes_unqualified\",\"sync_qualified\":false,\"calibration_version\":\"factory_raw_unqualified\",\"model_version\":\"none\",\"duration_seconds\":"<<seconds<<"}\n";config.close();
  ck(ds.start()==openni::STATUS_OK,"depth start");auto type=req.type;ioc(fd,VIDIOC_STREAMON,&type);
  atomic<bool> failed(false);uint64_t end=now_ns()+uint64_t(seconds)*1000000000ULL;string de,re;
  thread dt([&]{try{ofstream meta(out/"depth.jsonl");meta.exceptions(ios::badbit|ios::failbit);while(now_ns()<end&&!failed){openni::VideoStream* streams[]={&ds};int idx;auto status=openni::OpenNI::waitForAnyStream(streams,1,&idx,1000);if(status==openni::STATUS_TIME_OUT)throw runtime_error("depth timeout");ck(status==openni::STATUS_OK,"depth wait");openni::VideoFrameRef frame;ck(ds.readFrame(&frame)==openni::STATUS_OK&&frame.isValid(),"depth frame");auto arrival=now_ns();string name="depth/"+to_string(frame.getFrameIndex())+".bin";save(out/name,frame.getData(),frame.getDataSize());meta<<"{\"frame_id\":"<<frame.getFrameIndex()<<",\"sensor_timestamp_ns\":"<<frame.getTimestamp()*1000ULL<<",\"host_arrival_ns\":"<<arrival<<",\"exposure_host_ns\":null,\"stride_bytes\":"<<frame.getStrideInBytes()<<",\"width\":"<<frame.getWidth()<<",\"height\":"<<frame.getHeight()<<",\"payload\":\""<<name<<"\"}\n";}}catch(exception& e){de=e.what();failed=true;}});
  thread rt([&]{try{ofstream meta(out/"rgb.jsonl");meta.exceptions(ios::badbit|ios::failbit);while(now_ns()<end&&!failed){pollfd p={fd,POLLIN,0};int ready=poll(&p,1,1000);if(ready<=0)throw runtime_error("RGB poll timeout/error");v4l2_buffer b={};b.type=req.type;b.memory=req.memory;if(ioctl(fd,VIDIOC_DQBUF,&b)<0){if(errno==EAGAIN)continue;throw runtime_error("RGB dequeue failed");}auto arrival=now_ns();ck(b.index<ptr.size()&&b.bytesused<=length[b.index],"RGB bounds");string name="rgb/"+to_string(b.sequence)+".jpg";save(out/name,ptr[b.index],b.bytesused);meta<<"{\"frame_id\":"<<b.sequence<<",\"sensor_timestamp_ns\":null,\"v4l2_timestamp_ns\":"<<(uint64_t(b.timestamp.tv_sec)*1000000000ULL+uint64_t(b.timestamp.tv_usec)*1000ULL)<<",\"v4l2_flags\":"<<b.flags<<",\"host_arrival_ns\":"<<arrival<<",\"exposure_host_ns\":null,\"payload\":\""<<name<<"\"}\n";ioc(fd,VIDIOC_QBUF,&b);}}catch(exception& e){re=e.what();failed=true;}});
  dt.join();rt.join();ds.stop();ds.destroy();ioctl(fd,VIDIOC_STREAMOFF,&type);for(unsigned i=0;i<req.count;i++)munmap(ptr[i],length[i]);close(fd);dev.close();openni::OpenNI::shutdown();
  ofstream result(out/"result.json");result<<"{\"success\":"<<(failed?"false":"true")<<"}\n";
  if(failed){cerr<<"depth: "<<de<<" RGB: "<<re<<"\n";return 4;}cout<<"RGB/depth recording complete; sync and calibration remain unqualified\n";return 0;
 }catch(exception& e){if(created_out){ofstream(out/"failure.log")<<e.what()<<"\n";ofstream(out/"result.json")<<"{\"success\":false}\n";}cerr<<e.what()<<"\n";return 2;}
}
