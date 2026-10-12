from pathlib import Path
import subprocess
base=Path('/home/pc/rema-data');sdk=base/'deps/orbbec_sdk_v1/OrbbecSDK_v1.10.37/SDK'
for name in ['orbbec_v1_inventory','orbbec_v1_depth_check']:
 subprocess.run(['g++','-std=c++11','tools/'+name+'.cpp','-I'+str(sdk/'include'),'-L'+str(sdk/'lib'),'-Wl,-rpath,'+str(sdk/'lib'),'-lOrbbecSDK','-o',str(base/'work'/name)],check=True)
