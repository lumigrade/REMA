"""Only enumerated exact-Astra UVC controls, checked readback and restoration."""
import subprocess,re,json
from pathlib import Path
ROOT=Path('/home/pc/rema-data/calibration')
def astra_device():
 for p in sorted(Path('/sys/class/video4linux').glob('video*')):
  for parent in (p/'device').resolve().parents:
   if (parent/'idVendor').exists():
    if (parent/'idVendor').read_text().strip()=='2bc5' and (parent/'idProduct').read_text().strip()=='0501':
     node='/dev/'+p.name
     result=subprocess.run(['v4l2-ctl','-d',node,'--list-ctrls-menus'],capture_output=True,text=True,timeout=3)
     if result.returncode==0 and 'white_balance_automatic' in result.stdout:return node
    break
 raise RuntimeError('Exact Astra RGB control endpoint unavailable')
def read_controls():
 device=astra_device();result=subprocess.run(['v4l2-ctl','-d',device,'--list-ctrls-menus'],capture_output=True,text=True,check=True,timeout=3)
 values={}
 for line in result.stdout.splitlines():
  m=re.search(r'^\s*(\w+)\s+0x.*value=(-?\d+)',line)
  if m:values[m[1]]=int(m[2])
 return device,values,result.stdout
class Controls:
 def __init__(self):
  self.device,self.original,self.raw=read_controls();self.status='RGB exposure/shutter is not exposed by this driver';self.changed=False
  ROOT.mkdir(exist_ok=True);(ROOT/'uvc_controls_before.json').write_text(json.dumps({'device':self.device,'controls':self.original,'raw':self.raw},indent=2))
 def set(self,values):
  if any(k not in self.original for k in values):raise ValueError('Control not advertised by camera')
  result=subprocess.run(['v4l2-ctl','-d',self.device,'--set-ctrl',','.join(k+'='+str(v) for k,v in values.items())],capture_output=True,text=True,timeout=3)
  if result.returncode:raise RuntimeError(result.stderr.strip())
  _,actual,_=read_controls()
  if any(actual.get(k)!=v for k,v in values.items()):raise RuntimeError('Camera control readback mismatch')
  self.changed=True;return actual
 def lock_white_balance(self):
  _,current,_=read_controls();temperature=current.get('white_balance_temperature',4600)
  actual=self.set({'white_balance_automatic':0});actual=self.set({'white_balance_temperature':temperature})
  self.status=f'White balance locked at {temperature}K; exposure remains automatic';return actual
 def restore(self):
  for key in ['power_line_frequency','white_balance_temperature','white_balance_automatic']:
   if key in self.original:
    try:self.set({key:self.original[key]})
    except RuntimeError:
     if key!='white_balance_temperature':raise
  self.status='Original RGB controls restored';self.changed=False
