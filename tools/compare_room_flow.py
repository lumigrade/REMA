import sys,importlib.util,time,json,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,'/media/pc/WS_DATA/!WORK/REMA')
from rema.room_tracking import RoomTracker
from rema.provisional import load_provisional,align_provisional
from rema.matting import TemporalMatte
spec=importlib.util.spec_from_file_location('old_room','/home/pc/rema-data/work/stutter_backup/room_tracking.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
root=Path('/home/pc/rema-data/recordings/rema_live_20261009_233833');c=load_provisional();m=TemporalMatte();frames=[]
for i in range(100):
 f=np.load(root/(str(i)+'.npz'));frames.append((f['rgb'],align_provisional(f['depth'],c),m.infer(f['rgb'],i*.1)))
report={}
for name,cls in [('descriptor_only',old.RoomTracker),('optical_flow',RoomTracker)]:
 tr=cls(c);times=[];poses=[]
 for i,(rgb,d,prob) in enumerate(frames):
  t=time.perf_counter();p=tr.update(rgb,d,prob,i+1,1_000_000_000+i*100_000_000);times.append((time.perf_counter()-t)*1000);poses.append(p)
 report[name]=dict(valid=sum(p['valid'] for p in poses),frames=100,p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),reasons=dict(collections.Counter(p['reason'] for p in poses)),methods=dict(collections.Counter(p.get('method','descriptor') for p in poses)))
print(json.dumps(report,indent=2));Path('/home/pc/rema-data/reports/room_flow_replay.json').write_text(json.dumps(report,indent=2))
