import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import sys,os,json
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(_P.ROOT))
from unittest.mock import patch
from types import SimpleNamespace
from PySide6.QtWidgets import QApplication
from tools.live_app import Engine,Window
app=QApplication([])
with patch.object(Engine,'start'),patch.object(Engine,'close'):
 w=Window(SimpleNamespace(room_tracking=True,auto_record=False,seconds=7200))
 assert w.room_tracking.isChecked() and not w.floor_tracking.isEnabled()
 assert not w.capture_background_button.isEnabled() and not w.number_controls['horizon'].isEnabled()
 assert not w.number_controls['scale'].isEnabled()
 w.queue_room_action('reset');assert w.engine.room_action[0]=='reset'
 w.room_tracking.setChecked(False);assert w.engine.room_action is None
 assert w.capture_background_button.isEnabled() and w.number_controls['scale'].isEnabled()
 w.room_tracking.setChecked(True);w.queue_room_action('floor');assert w.engine.room_action[0]=='floor'
 w.close()
report=dict(room_tracking_enabled_without_board=True,measured_floor_target_still_disabled=True,static_reference_and_manual_alignment_disabled_while_tracking=True,timed_reset_and_floor_actions=True)
open(str(_P.DATA)+'/reports/room_gui_integration.json','w').write(json.dumps(report,indent=2));print(report)
