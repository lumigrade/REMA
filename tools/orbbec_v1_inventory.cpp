#include <libobsensor/ObSensor.hpp>
#include <iostream>
int main(){try{
 ob::Context::setLoggerToFile(OB_LOG_SEVERITY_DEBUG,"/home/pc/rema-data/logs/orbbec_v1");
 ob::Context context;context.enableNetDeviceEnumeration(false);
 auto list=context.queryDeviceList();std::cout<<"Device count: "<<list->deviceCount()<<"\n";
 for(unsigned i=0;i<list->deviceCount();i++)std::cout<<"vid="<<std::hex<<list->vid(i)<<" pid="<<list->pid(i)<<"\n";
 return 0;
}catch(ob::Error&e){std::cerr<<e.getMessage()<<"\n";return 2;}}
