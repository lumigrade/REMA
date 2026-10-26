# REMA rig host (RED + Osprey + Antilatency + Astra)

Inventory 2026-10-24 on host `223`, updated 2026-10-26. Machine facts live in `~/rema-data/config/host.json`, not in source.

| Item | Observed | Status |
|---|---|---|
| OS | Ubuntu 24.04.5, kernel 7.0.0-38 | — |
| GPU | **RTX 5070 12 GB** (`GPU-f9996886-…`), driver 595.99.02, no other compute owners | Earlier docs assumed RTX 5080; this host has a 5070 |
| SDI capture | Osprey **VB-USL** Video Bridge `2ee4:00ef`, USB 5 Gbps, kernel `uvcvideo` → `/dev/video0` | Driver OK, no vendor driver needed |
| Capture mode | Only mode advertised: **1360×768 YUYV @ 25 fps** (bridge scales the SDI input) | 1080p not exposed over USB |
| RED feed | Measured **14.8 fps** (frame intervals 56/60/88 ms), zero driver drops, 16 ms start-of-frame → app | Bridge delivers fewer frames than advertised; lens IS must be off |
| Antilatency | Socket `9C58…` (HMD2) → AltTag ROOT → **ALT6** `4C0046001951383039383438` (fw 4.0.1, on the RED top handle) and ALT5 (fw 4.5.0). udev rule installed | ALT6 Tracking6Dof at 250 Hz via REMA bridge; sees all 12 ceiling LEDs (0.11° vs. model). The socket needs a powered USB port (500 mA): on a bus-powered hub it dropped ~15× in 2 min on 2026-10-26 |
| Antilatency config | `~/.config/AntilatencyService`; default environment **Truss_1x1**, LED height corrected by the user to **4.665 m** (2026-10-24), placement Identity | Alt→camera offset solved by REMA (`tools/rig_record.py`), stored in the REMA rig profile |
| Orbbec Astra Pro | RGB + depth `2bc5:0403`, fw 5.8.22, USB 2.0 high speed; found by name | Live depth ~29 fps via `tools/astra_depth_helper.py` (OpenNI2 ros_astra_camera 210f530 redist + descriptor workaround); factory calibration nonfinite; occasional USB dropouts on a strained cable, REMA reconnects |
| Toolchain | gcc/g++, git, ffmpeg, v4l2-ctl installed by user 12:10 | Python envs under `~/rema-data/envs` (no system pip) |

## One-time setup that needs sudo (run in a terminal)

```bash
sudo apt install -y build-essential git v4l-utils ffmpeg
```

Antilatency socket access for your user (plugdev group), then replug the socket:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="3237", MODE="0660", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/90-antilatency.rules
```

Astra Pro depth access (the same exact-device rule used on the source host):

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2bc5", ATTR{idProduct}=="0403", MODE="0660", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/56-orbbec-astra-pro.rules
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Then: `~/Desktop/VPcamtracking/alt2freed/alt2freed list` should show the socket and Alt, and `sh tools/build_alt_bridge.sh` builds the 250 Hz pose bridge.

## Environments

- `~/rema-data/envs/rig`: numpy 2.2.6, opencv-headless 4.12.0.88, PySide6 6.8.3 (CPU tools, tests, live app). Freeze: `envs/rig.freeze.txt`.
- `~/rema-data/envs/live-gpu`: same + onnxruntime-gpu 1.23.2 and CUDA 12.9 wheels from `requirements-live-gpu.lock` (RVM matting).
- pip bootstrapped from the official `bootstrap.pypa.io/get-pip.py` (no system pip available).

## Commands

```bash
cd ~/Desktop/REMA
~/rema-data/envs/rig/bin/python tools/probe_sdi.py --seconds 10            # SDI mode/fps/drops/signal + labeled H.264
~/rema-data/envs/rig/bin/python tools/rig_live.py --pose alt2freed         # live grid overlay (needs udev rule)
~/rema-data/envs/rig/bin/python tools/rig_live.py --pose synthetic --headless --record 12   # labeled pipeline demo
OPENBLAS_NUM_THREADS=2 ~/rema-data/envs/rig/bin/python -m unittest discover -s tests -q
```

Calibration UI (needs OpenCV, so use REMA's env, and stop REMA's tracker first: only one process can own the Alt):

```bash
cd ~/Desktop/VPcamtracking/alt2freed && ~/rema-data/envs/rig/bin/python ui/server.py
```

Live with the 250 Hz bridge on the camera Alt: `tools/rig_live.py --pose bridge --alt-serial <serial>`.

Add `--profile ~/Desktop/VPcamtracking/alt2freed/calib/profiles/<name>.json` after solving lens + nodal offset in the VPcamtracking calibration UI (`./alt2freed-ui.sh`). Its frames and REMA share one convention (`rema/pose.py`), checked by `tests/test_rig.py` against VPcamtracking's own `tracker_rh` and alt2freed's FreeD encoder.

## Devices are selected by name

`/dev/videoN` numbers change across reboots and re-plugging (on 2026-10-26 the Astra became `/dev/video0` and the Osprey `/dev/video2`). All tools default to `--device Osprey` and resolve the V4L2 card name; the live label prints the real capture card.

## 3DGS background

```bash
~/rema-data/envs/live-gpu/bin/python tools/gpu_session.py --seconds 3600 -- ~/rema-data/envs/live-gpu/bin/python tools/rig_live.py --pose bridge --alt-serial 4C0046001951383039383438 --profile ~/rema-data/calibration/rig/RED_VRaptor_Optimo2876_ALT6_rema.json --matte cuda --background splat --scene ~/rema-data/scenes/samples/train.splat --scene-align ~/rema-data/scenes/samples/train_align.json --view composite
```

Operator view (default GUI): clean composite fullscreen, camera + Astra depth picture-in-pictures top-right. Keys: **I** info on/off, **R** record take on/off (`~/Desktop/Take<N>_<date_time>.mp4`, H.265), **V** composite/overlay, **F** fullscreen, **Q** quit. Tracking bar: green/long = good, red/short = bad (stage, dropouts, jitter).

Renderer: `rema/splat.py` (OpenGL 4.3 via headless EGL; .ply or .splat; SH degree 0). Alignment JSON maps scene -> REMA world and can clear the stage area.

## Conventions (rema/pose.py)

Antilatency raw pose (left-handed, Y up) → S frame (Y negated, VPcamtracking calibration) → REMA world W (X right, Y forward, Z up, floor z = 0). REMA uses the raw tracker pose and its own `T_tracker_cam`; never also apply an Antilatency placement. Video frames use the V4L2 kernel start-of-frame time (CLOCK_MONOTONIC); the pose is interpolated to `kernel_ns − video_delay_ms`. The video-vs-tracking delay was measured at 76–80 ms (motion cross-correlation and registration sweep) and is stored in the rig profile (`timing.video_delay_ms`); `rig_live` uses it by default. It is a software-clock alignment, not glass-to-glass latency.
