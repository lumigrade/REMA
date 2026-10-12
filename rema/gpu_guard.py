"""GPU admission guard: physical UUID ownership, legacy shared locks, no contention.
CPU fallback remains available. GPU sessions also enforce a runtime VRAM watchdog.
"""
import contextlib, fcntl, pathlib, subprocess
GPU_LOCKS={
 'GPU-653546c9-e49d-2a25-bd0e-1c8da819d062':['gpu0.lock'],
 'GPU-8ddc2281-7575-4d2a-962a-538850e1b60f':['gpu1.lock','gpu1_eval.lock'],
}
@contextlib.contextmanager
def reserve_gpu(uuid):
 if uuid not in GPU_LOCKS: raise ValueError('physical GPU UUID not inventoried')
 with contextlib.ExitStack() as stack:
  files=[pathlib.Path('/home/pc/rema-data/locks')/(uuid+'.lock')]
  files.extend(pathlib.Path('/home/pc/vto-data/locks')/n for n in GPU_LOCKS[uuid])
  for i,p in enumerate(files):
   f=stack.enter_context(p.open('a' if i==0 else 'r'))
   try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError: raise RuntimeError('GPU owner lock busy: '+str(p))
  query=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
  if any(line.split(',')[0].strip()==uuid for line in query.stdout.splitlines()): raise RuntimeError('GPU has active compute owner')
  yield
