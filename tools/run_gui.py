#!/usr/bin/env python3
"""Launch one CPU-only interactive app, 4 GiB RAM, 4 CPU cores quota, 2-hour limit."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import os,subprocess,sys,shutil,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];data=Path(str(_P.DATA));sdk=next((data/'deps/openni').glob('OpenNI*'))/'sdk'
calibrating='--calibrate' in sys.argv
preferences=data/'config/live_runtime.json'
default_gpu=preferences.exists() and json.loads(preferences.read_text()).get('matte_backend')=='cuda'
gpu=('--gpu' in sys.argv or default_gpu) and '--cpu' not in sys.argv and not calibrating
unit='rema-calibration-gui' if calibrating else 'rema-live-gui'
forward=[a for a in sys.argv[1:] if a not in ('--calibrate','--gpu','--cpu')]
for v in [Path(v) for v in _P.RESERVE_VOLUMES]:
 if shutil.disk_usage(v).free<25*1024**3:sys.exit('25 GiB disk reserve failed')
if subprocess.run(['systemctl','--user','is-active','--quiet',unit+'.service']).returncode==0:
 print('REMA is already running; select its window in the desktop.');sys.exit(0)
projection_source=root/'tools/depth_projection.cpp';projection_binary=data/'work/depth_projection.so'
if not projection_binary.exists() or projection_source.stat().st_mtime>projection_binary.stat().st_mtime:
 subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(projection_source),'-o',str(projection_binary)+'.tmp'],check=True)
 Path(str(projection_binary)+'.tmp').replace(projection_binary)
source=root/'tools/openni_live_bridge.cpp';binary=data/'work/openni_live_bridge.so'
if not binary.exists() or source.stat().st_mtime>binary.stat().st_mtime:
 subprocess.run(['g++','-std=c++11','-shared','-fPIC',str(source),'-I'+str(sdk/'Include'),'-L'+str(sdk/'libs'),'-Wl,-rpath,'+str(sdk/'libs'),'-lOpenNI2','-o',str(binary)+'.tmp'],check=True)
 Path(str(binary)+'.tmp').replace(binary)
subprocess.run(['systemctl','--user','reset-failed',unit+'.service'],stderr=subprocess.DEVNULL)
args=['systemd-run','--user','--collect','--unit='+unit,'--property=MemoryMax=4G','--property=MemorySwapMax=0','--property=TasksMax=64','--property=CPUQuota=400%','--property=OOMPolicy=stop','--property=RuntimeMaxSec=7220','--property=TimeoutStopSec=10','--property=StandardOutput=append:'+str(data/'logs/gui/app.log'),'--property=StandardError=append:'+str(data/'logs/gui/app.log'),'--working-directory='+str(root)]
for k,v in {'LD_PRELOAD':str(data/'work/astra_usb_compat.so'),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','QT_QUICK_BACKEND':'software','QT_XCB_GL_INTEGRATION':'none','LIBGL_ALWAYS_SOFTWARE':'1','DISPLAY':os.environ.get('DISPLAY',':1'),'XAUTHORITY':os.environ.get('XAUTHORITY','/run/user/1000/gdm/Xauthority')}.items():args+=['--setenv='+k+'='+v]
python=str(data/('envs/live-gpu/bin/python' if gpu else 'envs/p0/bin/python'))
command=[python,str(root/('tools/calibrate_app.py' if calibrating else 'tools/live_app.py'))]+forward
if gpu:command=[python,str(root/'tools/gpu_session.py'),'--']+command
args+=['--']+command
subprocess.run(args,check=True)
print('REMA window launched. '+('RTX3090 GPU matting; CPU tracking/rendering.' if gpu else 'CPU only.')+' Closes after two hours or with Close.')
