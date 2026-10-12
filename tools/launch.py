#!/usr/bin/env python3
"""P0 CPU launcher: cgroup memory/runtime caps, one owner, disk reserve. GPU closed."""
import argparse, fcntl, os, pathlib, shutil, subprocess, sys, time, uuid
p=argparse.ArgumentParser(); p.add_argument('--seconds',type=int,default=120); p.add_argument('--memory-mib',type=int,default=4096); p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
if not 1 <= a.seconds <= 600 or not 256 <= a.memory_mib <= 4096: p.error('P0 limits: 1–600 seconds, 256–4096 MiB')
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if not cmd: p.error('command required')
for volume in ['/home/pc/rema-data','/media/pc/WS_DATA']:
 if shutil.disk_usage(volume).free < 25*1024**3: sys.exit('25 GiB free-space reserve failed: '+volume)
locks=pathlib.Path('/home/pc/rema-data/locks'); locks.mkdir(exist_ok=True)
with (locks/'p0-cpu.lock').open('a') as lock:
 try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: sys.exit('REMA P0 owner already running')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 unit='rema-p0-'+uuid.uuid4().hex
 args=['systemd-run','--unit='+unit,'--user','--wait','--pipe','--collect','--property=MemoryMax='+str(a.memory_mib)+'M','--property=MemorySwapMax=0','--property=TasksMax=64','--property=OOMPolicy=stop','--property=RuntimeMaxSec='+str(a.seconds),'--working-directory=/media/pc/WS_DATA/!WORK/REMA','--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=2','--setenv=OPENBLAS_NUM_THREADS=2','--']+cmd
 proc=subprocess.Popen(args,env=env)
 deadline=time.monotonic()+a.seconds+15
 try:
  while proc.poll() is None:
   if time.monotonic()>deadline or any(shutil.disk_usage(v).free<25*1024**3 for v in ['/home/pc/rema-data','/media/pc/WS_DATA']):
    subprocess.run(['systemctl','--user','stop',unit],check=False); sys.exit('runtime/disk reserve failed; REMA unit stopped')
   time.sleep(.25)
  sys.exit(proc.returncode)
 except KeyboardInterrupt:
  subprocess.run(['systemctl','--user','stop',unit],check=False); sys.exit(130)
