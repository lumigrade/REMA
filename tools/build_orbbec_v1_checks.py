import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
from pathlib import Path
import subprocess
base=Path(str(_P.DATA));sdk=base/'deps/orbbec_sdk_v1/OrbbecSDK_v1.10.37/SDK'
for name in ['orbbec_v1_inventory','orbbec_v1_depth_check']:
 subprocess.run(['g++','-std=c++11','tools/'+name+'.cpp','-I'+str(sdk/'include'),'-L'+str(sdk/'lib'),'-Wl,-rpath,'+str(sdk/'lib'),'-lOrbbecSDK','-o',str(base/'work'/name)],check=True)
