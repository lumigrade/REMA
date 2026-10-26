# REMA operator guide

How to run a REMA virtual-production session on the rig host. One-time host setup (drivers, udev rules, Python environments) is in [../RIG_HOST.md](../RIG_HOST.md).

## 1. Rig check (before every session)

| Check | How | Good |
|---|---|---|
| RED SDI output is **Clean** | camera monitoring menu | no status bars, frame lines or centre mark on the SDI feed |
| RED picture reaches the Osprey | `~/rema-data/envs/rig/bin/python tools/probe_sdi.py --seconds 5` | `signal: live`, ~14.8 fps, 0 drops. `colour_bars` = no SDI input (camera, cable or SDI output setting). |
| Lens settings | on the camera | **Image stabilisation OFF, autofocus OFF**, same zoom/focus as calibration. IS shifts the picture inside the camera and breaks registration. |
| Antilatency socket | `lsusb -t` | on a **motherboard USB port or powered hub** — not a bus-powered hub (it needs 500 mA; brown-outs drop every Alt). |
| Alt on the camera tracks | `~/Desktop/VPcamtracking/alt2freed/alt2freed list` | ALT6 listed; live view shows the tracking bar green. |
| Astra depth | live view depth picture-in-picture | colour depth image; `NO DEPTH – reconnecting` means re-seat its USB cable. |

Cameras are found **by name** (Osprey, Astra), so `/dev/videoN` numbering changes after reboots do not matter. The live view prints the actual capture source; a red `SOURCE IS NOT THE OSPREY` line means the wrong device.

Only one program can use the Alt at a time: close AntilatencyService, the VPcamtracking UI stream and other REMA tools before starting a REMA tool.

## 2. Calibration

Redo after any change to lens, zoom, focus, sensor crop, Alt mounting or the tracking environment.

### Lens
Hold the ChArUco board (`calibration_targets/`, or the REMA 10×7 25 mm board) at 0.5–1.2 m, 20–30 views covering the frame edges and corners with tilts. Hand-held board views are ideal for the lens (camera static).

### Alt → camera offset (nodal)
```bash
~/rema-data/envs/rig/bin/python tools/rig_record.py --positions 14
```
- **Board fixed** (taped to a wall or on a stand), whole board in frame.
- On **MOVE**: move the *camera* — pan, **tilt**, move the tripod ≥ 0.5 m, change its height. On **HOLD**: hands off.
- The solve runs automatically and writes `~/rema-data/calibration/rig/<profile>_rema.json` plus a review video. Good: held-out rotation error ≲ 0.25°, offset tens of centimetres, camera vs. Alt ≈ 90° for an up-facing Alt.

### Registration test
```bash
~/rema-data/envs/rig/bin/python tools/rig_registration.py
```
Board fixed ~2 m away; follow the prompts (pan, tilt, dolly, faster moves). Reports error by camera speed and the best video delay; a desktop notification appears when the analysis is done.

## 3. Live session

**Easiest:** click **REMA Live** in the Dash (install once with `tools/install_desktop_app.sh`). It checks the Osprey signal, the tracking socket (warns if it is on a hub) and the Astra, then opens the operator view; a second click only reports that REMA is already running. Session log: `~/rema-data/logs/rema_app.log`. Optional overrides (profile, scene, alignment, background): `~/rema-data/config/rema_app.json`.

Manual start:

```bash
~/rema-data/envs/live-gpu/bin/python tools/gpu_session.py --seconds 3600 -- \
  ~/rema-data/envs/live-gpu/bin/python tools/rig_live.py --pose bridge \
  --profile ~/rema-data/calibration/rig/RED_VRaptor_Optimo2876_ALT6_rema.json \
  --matte cuda --background splat --view composite
```

- Fullscreen clean composite; camera and depth picture-in-pictures top right.
- **R** record take (red circle top left while recording; on stop a 3 s centre notification shows the saved file name), **I** text/indicators on/off (recording circle and notification stay), **V** composite/overlay, **F**/**Esc** fullscreen, **Q** quit.
- Takes: `~/Desktop/Take<N>_<YYYYmmdd_HHMMSS>.mp4` (H.265 via NVENC, clean composite, 25 fps real-time timeline). Metadata (settings, held-pose frame count, encoder status): `~/rema-data/recordings/takes/`.
- Background options: `--background grid` (calibration grid) or `--background splat --scene <file.ply|.splat> --scene-align <align.json>`.
- `--tracking-loss hold` (default) keeps compositing over the last good pose during a dropout and flags it; `--tracking-loss raw` shows the camera picture instead.

## 4. Using your own 3DGS scene

1. Capture the location (phone/drone) and train a standard 3DGS `.ply` in your preferred tool (Postshot, Polycam, Luma, the Inria trainer). Put a tape-measured object in the shot so real scale can be set from a measurement.
2. Create an alignment JSON mapping the scene into REMA's world (metres, Z up, floor at z = 0): `{"scale", "rotation_deg", "translation"}` or a 4×4 `"matrix"`, plus optional `"clear_cylinders"` to empty the stage area. `rema/splat.py` has a floor-plane helper; the sample `~/rema-data/scenes/samples/train_align.json` documents every step taken.
3. Check with `--view overlay` that the floor meets the real floor and the scale feels right; adjust and relaunch.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Composite cuts to the real room / tracking bar red "tracker samples late" | Antilatency radio or USB dropout | socket on a direct powered port; check Alt battery and Tag; kernel log `journalctl -k | grep 1-7` |
| Picture drifts after each move while tracking is still | lens image stabilisation | turn IS off |
| Wrong picture, 10 fps, 1280×720 | wrong capture device | use the default `--device Osprey` (devices are found by name) |
| Actor picture smaller than the background, camera status text/frame lines visible, red `CAMERA SDI OVERLAYS ON` | RED sends its monitoring overlays on SDI, which shrinks the picture into a window (calibration expects the clean full frame) | on the RED: Monitoring → Monitors → SDI → overlays **Clean** (frame guides off for SDI) |
| `colour_bars` / no composite | no SDI into the Osprey | camera SDI output on, cable seated |
| Depth PiP `NO DEPTH – reconnecting` | Astra USB dropped | re-seat the Astra cable (USB 2.0 high-speed port, strain relief) — REMA reconnects automatically |
| Edge fringe around the actor | bright, busy background behind the actor; colour-only matte | soften background lights; depth-assisted matting is the next step |
