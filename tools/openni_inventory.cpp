#include <OpenNI.h>
#include <iostream>
int main() {
 auto status=openni::OpenNI::initialize();
 if(status!=openni::STATUS_OK){std::cerr<<openni::OpenNI::getExtendedError()<<"\n";return 2;}
 openni::Array<openni::DeviceInfo> devices; openni::OpenNI::enumerateDevices(&devices);
 std::cout<<"OpenNI enumerated devices: "<<devices.getSize()<<"\n";
 for(int i=0;i<devices.getSize();++i) std::cout<<devices[i].getVendor()<<" "<<devices[i].getName()<<" vid="<<devices[i].getUsbVendorId()<<" pid="<<devices[i].getUsbProductId()<<"\n";
 std::cout<<openni::OpenNI::getExtendedError()<<"\n";
 openni::OpenNI::shutdown(); return devices.getSize()?0:3;
}
