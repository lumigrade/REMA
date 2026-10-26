#!/usr/bin/env python3
"""REMA desktop app launcher (used by the Dash icon).

Pre-flight checks, then starts the fullscreen operator view through the GPU guard:
  - not already running (a second click just reports it),
  - Osprey present and receiving a picture (colour bars = no SDI input),
  - Antilatency radio socket on USB (warns about bus-powered hubs),
  - Astra Pro present (optional: depth picture-in-picture).
Problems are shown as desktop notifications; the session log goes to $REMA_DATA/logs/rema_app.log.
Settings (profile, scene, alignment) can be overridden in $REMA_DATA/config/rema_app.json.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from rema import paths as _P
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LOG = _P.DATA / 'logs' / 'rema_app.log'
DEFAULTS = {
    'alt_serial': '4C0046001951383039383438',
    'profile': str(_P.DATA / 'calibration/rig/RED_VRaptor_Optimo2876_ALT6_rema.json'),
    'background': 'splat',
    'scene': str(_P.DATA / 'scenes/samples/train.splat'),
    'scene_align': str(_P.DATA / 'scenes/samples/train_align.json'),
    'seconds': 7200,
}


def notify(title, body, urgent=False):
    try:
        subprocess.run(['notify-send', '-a', 'REMA', '-i', str(Path.home() / '.local/share/icons/hicolor/256x256/apps/rema.png'),
                        '-u', 'critical' if urgent else 'normal', title, body], timeout=5)
    except Exception:
        pass
    print(f'{title}: {body}', flush=True)


def usb_present(vendor, product=None):
    for d in Path('/sys/bus/usb/devices').glob('*'):
        try:
            if (d / 'idVendor').read_text().strip() == vendor and (product is None or (d / 'idProduct').read_text().strip() == product):
                return d
        except OSError:
            continue
    return None


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULTS)
    cfg_path = _P.CONFIG / 'rema_app.json'
    if cfg_path.exists():
        cfg.update(json.loads(cfg_path.read_text()))

    if subprocess.run(['pgrep', '-f', 'tools/rig_live.py'], capture_output=True).returncode == 0:
        notify('REMA is already running', 'Switch to the REMA window (Alt+Tab). Press Q in it to quit.')
        return 0

    # Osprey + signal
    from rema.sdi_capture import SignalMonitor, V4L2Capture, find_video_device, picture_inset
    try:
        dev = find_video_device('Osprey')
    except RuntimeError:
        notify('REMA: Osprey not found', 'Connect the Osprey SDI bridge (USB 3 port) and try again.', urgent=True)
        return 1
    try:
        with V4L2Capture(dev) as cap:
            mon, states, insets, t0 = SignalMonitor(), [], [], time.monotonic()
            while time.monotonic() - t0 < 1.5:
                f = cap.read(timeout=1.0)
                if f is not None:
                    states.append(mon.update(f.image_bgr))
                    insets.append(picture_inset(f.image_bgr))
    except OSError as e:
        notify('REMA: Osprey busy or unavailable', f'{dev}: {e}. Close other programs using the camera.', urgent=True)
        return 1
    if insets and sum(i is not None for i in insets) > len(insets) // 2:
        notify('REMA: camera SDI overlays are on', 'The RED is sending its monitoring overlays on SDI, which shrinks the picture '
               'into a window. On the camera: Monitoring > Monitors > SDI > overlays Clean. Starting anyway.', urgent=True)
    if states and states.count('colour_bars') > len(states) // 2:
        notify('REMA: no camera signal', 'The Osprey shows colour bars: check the RED SDI output and cable. Starting anyway.')

    # Antilatency socket
    sock = usb_present('3237')
    if sock is None:
        notify('REMA: Antilatency socket not found', 'Plug the radio socket into a direct USB port. Starting without tracking (camera picture only).', urgent=True)
    elif sock.name.count('.') >= 1:
        notify('REMA: tracking socket is on a hub', 'For reliable tracking use a direct motherboard USB port (the socket needs 500 mA).')

    if usb_present('2bc5', '0403') is None:
        notify('REMA: Astra depth not connected', 'Depth picture-in-picture will show NO DEPTH until it is plugged in.')

    profile = Path(cfg['profile']).expanduser()
    if not profile.exists():
        notify('REMA: calibration profile missing', str(profile), urgent=True)
        return 1

    py = str(_P.DATA / 'envs/live-gpu/bin/python')
    cmd = [py, str(_P.ROOT / 'tools/gpu_session.py'), '--seconds', str(int(cfg['seconds'])), '--',
           py, str(_P.ROOT / 'tools/rig_live.py'), '--pose', 'bridge', '--alt-serial', cfg['alt_serial'],
           '--profile', str(profile), '--matte', 'cuda', '--view', 'composite', '--seconds', str(int(cfg['seconds'])),
           '--background', cfg['background']]
    if cfg['background'] == 'splat':
        cmd += ['--scene', str(Path(cfg['scene']).expanduser()), '--scene-align', str(Path(cfg['scene_align']).expanduser())]
    with LOG.open('a') as log:
        log.write(f'\n===== {datetime.datetime.now().isoformat(timespec="seconds")} REMA app start\n{" ".join(cmd)}\n')
        log.flush()
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(_P.ROOT))
        log.write(f'===== exit {rc}\n')
    if rc != 0:
        tail = LOG.read_text(errors='replace').strip().splitlines()[-3:]
        notify('REMA stopped with an error', ' | '.join(tail)[-300:] + f'  (log: {LOG})', urgent=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
