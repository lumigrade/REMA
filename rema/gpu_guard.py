"""GPU admission guard: physical UUID ownership, optional shared locks, no contention.
Inventoried UUIDs come from $REMA_DATA/config/host.json ("gpus": {uuid: {"shared_locks": [...]}}).
Other projects' shared lock files are respected whenever their directory exists on this host.
CPU fallback remains available. GPU sessions also enforce a runtime VRAM watchdog.
"""
from rema import paths as _P
import contextlib, fcntl, pathlib, subprocess
def gpu_locks():
 return {u:list(v.get('shared_locks',[])) for u,v in _P.host_config().get('gpus',{}).items()}
@contextlib.contextmanager
def reserve_gpu(uuid):
 locks=gpu_locks()
 if uuid not in locks: raise ValueError('physical GPU UUID not inventoried in '+str(_P.HOST_CONFIG))
 with contextlib.ExitStack() as stack:
  _P.LOCKS.mkdir(parents=True,exist_ok=True)
  files=[_P.LOCKS/(uuid+'.lock')]
  if _P.SHARED_LOCKS.is_dir(): files.extend(_P.SHARED_LOCKS/n for n in locks[uuid])
  for i,p in enumerate(files):
   f=stack.enter_context(p.open('a' if i==0 else 'r'))
   try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError: raise RuntimeError('GPU owner lock busy: '+str(p))
  query=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
  if any(line.split(',')[0].strip()==uuid for line in query.stdout.splitlines()): raise RuntimeError('GPU has active compute owner')
  yield
