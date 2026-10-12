"""P0 recording integrity/arrival diagnostics. No exposure synchronization claim."""
import argparse,dataclasses,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from rema.contracts import Frame,load_replay

def analyze(root):
 root=Path(root);config=json.loads((root/'capture.json').read_text());result=json.loads((root/'result.json').read_text())
 if not result['success']:raise ValueError('capture failed; partial evidence must be retained')
 report={'stream_integrity_passed':True,'evidence_origin':config.get('evidence_origin','unqualified'),'sync_qualified':False,'calibration_qualified':False,'exposure_pair_skew_ns':None,'streams':{}};allframes=[];streams={}
 for stream in ['rgb','depth']:
  rows=[json.loads(x) for x in (root/(stream+'.jsonl')).read_text().splitlines()]
  if len(rows)<2:raise ValueError('not enough '+stream+' frames')
  ids=np.array([r['frame_id'] for r in rows],dtype=np.int64);arrival=np.array([r['host_arrival_ns'] for r in rows],dtype=np.int64)
  if np.any(np.diff(ids)<=0) or np.any(np.diff(arrival)<=0):raise ValueError('non-monotonic IDs/arrival in '+stream)
  device_time=[r['sensor_timestamp_ns'] for r in rows]
  if stream=='depth' and any(b<=a for a,b in zip(device_time,device_time[1:])):raise ValueError('non-monotonic depth sensor timestamps')
  valid=[]
  for row in rows:
   p=(root/row['payload']).resolve()
   if not p.is_relative_to(root.resolve()):raise ValueError('payload path outside recording')
   data=p.read_bytes()
   if stream=='depth':
    if len(data)!=row['stride_bytes']*row['height'] or row['stride_bytes']<row['width']*2:raise ValueError('depth stride/size invalid')
    depth=np.frombuffer(data,dtype='<u2').reshape(row['height'],row['stride_bytes']//2)[:,:row['width']]
    valid.append(float(np.mean(depth!=0)))
   else:
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.shape[:2]!=(config['height'],config['width']):raise ValueError('RGB decode/mode failure')
   allframes.append(Frame(stream,row['frame_id'],row['sensor_timestamp_ns'],row['host_arrival_ns'],None,'openni_depth_device' if stream=='depth' else 'v4l2_kernel_unqualified',config['calibration_version'],config['model_version'],row['payload'],hashlib.sha256(data).hexdigest()))
  stat={'frames':len(rows),'sequence_gaps':int(np.maximum(0,np.diff(ids)-1).sum()),'arrival_fps':float((len(rows)-1)*1e9/(arrival[-1]-arrival[0])),'arrival_interval_ms_p50_p95_max':[float(v) for v in np.percentile(np.diff(arrival)/1e6,[50,95,100])],'device_timestamp_available':stream=='depth'}
  if valid:stat['nonzero_depth_fraction_p05_p50_p95']=[float(v) for v in np.percentile(valid,[5,50,95])]
  else:stat['v4l2_flags']=sorted({r['v4l2_flags'] for r in rows});stat['timestamp_semantics']='kernel buffer timestamp retained; camera exposure time unqualified'
  report['streams'][stream]=stat;streams[stream]=arrival
 # A diagnostic only: nearest arrival from separate capture threads can hide real exposure skew.
 r,d=streams['rgb'],streams['depth'];idx=np.searchsorted(d,r);idx=np.clip(idx,0,len(d)-1);prior=np.clip(idx-1,0,len(d)-1);skew=np.minimum(np.abs(d[idx]-r),np.abs(d[prior]-r))/1e6
 report['nearest_arrival_difference_ms_p50_p95_max']=[float(v) for v in np.percentile(skew,[50,95,100])]
 report['nearest_arrival_difference_warning']='not exposure skew; not a sync qualification metric'
 allframes.sort(key=lambda f:(f.host_arrival_ns,f.stream));(root/'frames.jsonl').write_text(''.join(json.dumps(dataclasses.asdict(f))+'\n' for f in allframes));
 # Compare full ordered metadata twice; integrity checks visit every payload each time.
 if load_replay(root)!=load_replay(root):raise ValueError('deterministic replay mismatch')
 report['deterministic_replay_passed']=True;(root/'analysis.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('recording');a=p.parse_args();print(json.dumps(analyze(a.recording),indent=2))
