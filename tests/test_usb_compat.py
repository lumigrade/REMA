"""Exercise the interposer with a fake libusb; no real device is touched."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import os,pathlib,subprocess,tempfile,unittest
class USBCompatibility(unittest.TestCase):
 def test_scope_bounds_and_errors(self):
  with tempfile.TemporaryDirectory(dir=str(_P.DATA)+'/work') as tmp:
   root=pathlib.Path(tmp);include=str(_P.DATA)+'/deps/libusb-legacy-src/libusb-1.0.9/libusb'
   (root/'fake.c').write_text(r'''#include <libusb.h>
#include <string.h>
int mode=0,seen=0;
libusb_device *libusb_get_device(libusb_device_handle*h){return (libusb_device*)h;}
int libusb_get_device_descriptor(libusb_device*d,struct libusb_device_descriptor*p){memset(p,0,sizeof(*p));p->idVendor=mode==1?0x1234:0x2bc5;p->idProduct=0x0403;return 0;}
int libusb_control_transfer(libusb_device_handle*h,uint8_t type,uint8_t req,uint16_t v,uint16_t idx,unsigned char*data,uint16_t n,unsigned int timeout){seen=n;if(mode==3)return -7;if(mode==4)return 0;if(type==0x80){unsigned char reply[]={4,3,9,4};int size=n<4?n:4;memcpy(data,reply,size);return size;}return n;}
''')
   (root/'check.c').write_text(r'''#include <libusb.h>
#include <assert.h>
extern int mode,seen;
int main(){libusb_device_handle*h=(void*)1;unsigned char b[]={0xaa,0,0xbb};int r;
mode=0;r=libusb_control_transfer(h,0x80,6,0x300,0x409,b+1,1,1000);assert(r==1&&seen==256&&b[1]==4&&b[0]==0xaa&&b[2]==0xbb);
mode=1;r=libusb_control_transfer(h,0x80,6,0x300,0x409,b+1,1,1000);assert(r==1&&seen==1);
mode=0;r=libusb_control_transfer(h,0x40,0,0,0,b,3,1000);assert(r==3&&seen==3&&b[0]==0xaa&&b[2]==0xbb);
mode=3;r=libusb_control_transfer(h,0x80,6,0x300,0x409,b+1,1,1000);assert(r==-7);
mode=4;r=libusb_control_transfer(h,0x80,6,0x300,0x409,b+1,1,1000);assert(r==0);
return 0;}
''')
   subprocess.run(['gcc','-shared','-fPIC','-I'+include,str(root/'fake.c'),'-o',str(root/'libusb-1.0.so.0')],check=True)
   subprocess.run(['gcc','-I'+include,str(root/'check.c'),'-L'+tmp,'-Wl,-rpath,'+tmp,'-l:libusb-1.0.so.0','-o',str(root/'check')],check=True)
   env=dict(os.environ,LD_LIBRARY_PATH=tmp,LD_PRELOAD=str(_P.DATA)+'/work/astra_usb_compat.so')
   result=subprocess.run([str(root/'check')],env=env,capture_output=True,text=True,timeout=5)
   self.assertEqual(result.returncode,0,result.stderr)
