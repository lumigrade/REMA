// Astra Pro 2bc5:0403 legacy OpenNI language-descriptor compatibility.
// Read the complete language table, return only the byte requested by OpenNI.
// Never alter vendor commands, response IDs, depth data, or other USB devices.
#define _GNU_SOURCE
#include <libusb.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
static pthread_once_t once=PTHREAD_ONCE_INIT;
static int (*control)(libusb_device_handle*,uint8_t,uint8_t,uint16_t,uint16_t,unsigned char*,uint16_t,unsigned int);
static libusb_device*(*get_device)(libusb_device_handle*);
static int(*get_descriptor)(libusb_device*,struct libusb_device_descriptor*);
static void init(void){
 void*lib=dlopen("libusb-1.0.so.0",RTLD_NOW|RTLD_LOCAL);
 if(lib){*(void**)(&control)=dlsym(lib,"libusb_control_transfer");*(void**)(&get_device)=dlsym(lib,"libusb_get_device");*(void**)(&get_descriptor)=dlsym(lib,"libusb_get_device_descriptor");}
}
int libusb_control_transfer(libusb_device_handle*h,uint8_t type,uint8_t request,uint16_t value,uint16_t index,unsigned char*data,uint16_t length,unsigned int timeout){
 pthread_once(&once,init);if(!control||!get_device||!get_descriptor)return LIBUSB_ERROR_OTHER;
 if(type==0x80&&request==LIBUSB_REQUEST_GET_DESCRIPTOR&&value==0x0300&&index==0x0409&&length==1&&data){
  struct libusb_device_descriptor device;
  if(get_descriptor(get_device(h),&device)==0&&device.idVendor==0x2bc5&&device.idProduct==0x0403){
   unsigned char descriptor[256]={};int n=control(h,type,request,value,index,descriptor,sizeof(descriptor),timeout);
   fprintf(stderr,"REMA Astra descriptor workaround: read %d bytes instead of 1\n",n);
   if(n>=1){data[0]=descriptor[0];return 1;}return n;
  }
 }
 return control(h,type,request,value,index,data,length,timeout);
}
