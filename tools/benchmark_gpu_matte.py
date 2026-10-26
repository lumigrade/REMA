import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import os,sys,time,json,numpy as np
from pathlib import Path
sys.path.insert(0,str(_P.ROOT))
from rema.matting import TemporalMatte
frames=[np.load(str(_P.DATA)+'/recordings/rema_live_20261009_230803/'+str(i)+'.npz')['rgb'] for i in range(50)]
report={};reference=[]
for backend in ['cpu','cuda']:
 os.environ['REMA_MATTE_BACKEND']=backend;m=TemporalMatte()
 for i in range(4):m.infer(frames[0],i/30)
 m.reset();times=[];errors=[]
 for i,f in enumerate(frames):
  t=time.perf_counter();alpha=m.infer(f,i*.1);times.append((time.perf_counter()-t)*1000)
  if backend=='cpu':reference.append(alpha.copy())
  else:errors.append(float(np.mean(np.abs(alpha-reference[i]))))
 report[backend]=dict(providers=m.session.get_providers(),p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),max_ms=max(times),mean_alpha_error=float(np.mean(errors)) if errors else 0)
 print(backend,report[backend],flush=True)
 del m
Path(str(_P.DATA)+'/reports/gpu_matte_benchmark.json').write_text(json.dumps(report,indent=2));assert report['cuda']['mean_alpha_error']<.002
