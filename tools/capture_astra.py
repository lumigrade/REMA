"""CPU-only bounded capture using Astra-specific local USB compatibility library."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import argparse,datetime,pathlib,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=5);p.add_argument('--device',default='Astra');a=p.parse_args()
from rema.sdi_capture import resolve_device;a.device=resolve_device(a.device)
if not 0<=a.seconds<=60:p.error('duration must be 0..60 seconds')
base=pathlib.Path(str(_P.DATA));stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out=base/('recordings' if a.seconds else 'reports')/('astra_'+stamp)
cmd=['python3','tools/launch.py','--seconds',str(a.seconds+30),'--','/usr/bin/env','LD_PRELOAD='+str(base/'work/astra_usb_compat.so'),str(base/'work/astra_capture'),str(out),str(a.seconds),a.device]
r=subprocess.run(cmd);print('Dataset:',out,flush=True);sys.exit(r.returncode)
