#!/usr/bin/env python3
"""One GPU owner, shared UUID locks, 4 GiB process VRAM watchdog, finite runtime.
Run inside the existing RAM/CPU-capped systemd launcher. Never kills other jobs.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import argparse,os,sys,subprocess,time,shutil,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rema.gpu_guard import reserve_gpu
UUID=_P.host_config().get('live_gpu_uuid') or 'GPU-not-inventoried'
def descendants(pid):
 """pid plus all its descendant processes (from /proc), e.g. REMA's own ffmpeg NVENC encoder."""
 children={}
 for d in Path('/proc').iterdir():
  if d.name.isdigit():
   try:ppid=int((d/'stat').read_text().rsplit(')',1)[1].split()[1])
   except (OSError,IndexError,ValueError):continue
   children.setdefault(ppid,[]).append(int(d.name))
 out,todo={pid},[pid]
 while todo:
  for c in children.get(todo.pop(),[]):
   if c not in out:out.add(c);todo.append(c)
 return out

def violation(output,uuid,pid,cap=4096):
 """pid: the session child pid or a set of allowed pids (the child and its descendants)."""
 allowed=pid if isinstance(pid,(set,frozenset)) else {pid}
 total=0
 for line in output.splitlines():
  fields=[s.strip() for s in line.split(',')]
  if len(fields)!=3 or fields[0]!=uuid:continue
  if int(fields[1]) not in allowed:return 'Another compute owner appeared; stopping only REMA'
  total+=int(fields[2])
 return 'REMA process exceeded VRAM budget' if total>cap else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=7200);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
 if not cmd or not 1<=a.seconds<=7200:p.error('command and finite runtime1–7200 required')
 with reserve_gpu(UUID):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,REMA_MATTE_BACKEND='cuda',REMA_GPU_UUID=UUID)
  child=subprocess.Popen(cmd,env=env);deadline=time.monotonic()+a.seconds
  def stop(*_):
   if child.poll() is None:child.terminate()
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  try:
   while child.poll() is None:
    if time.monotonic()>deadline:raise RuntimeError('GPU session runtime limit')
    if any(shutil.disk_usage(v).free<25*1024**3 for v in [str(v) for v in _P.RESERVE_VOLUMES]):raise RuntimeError('Disk reserve below25GiB')
    q=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
    error=violation(q.stdout,UUID,descendants(child.pid))
    if error:raise RuntimeError(error)
    time.sleep(1)
   return child.returncode
  finally:
   stop()
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:child.kill();child.wait()
if __name__=='__main__':sys.exit(main())
