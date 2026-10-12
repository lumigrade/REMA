from pathlib import Path
import subprocess
base=Path('/home/pc/rema-data')
sdk=next((base/'deps/openni').glob('OpenNI*'))/'sdk'
subprocess.run(['g++','-std=c++17','-O2','-pthread','tools/astra_capture.cpp','-I'+str(sdk/'Include'),'-L'+str(sdk/'libs'),'-Wl,-rpath,'+str(sdk/'libs'),'-lOpenNI2','-o',str(base/'work/astra_capture')],check=True)

headers=base/"deps/libusb-legacy-src/libusb-1.0.9/libusb"
subprocess.run(["gcc","-Wall","-Wextra","-shared","-fPIC","-pthread","tools/astra_usb_compat.c","-I"+str(headers),"-ldl","-o",str(base/"work/astra_usb_compat.so")],check=True)
