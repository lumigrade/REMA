# P0: working Linux capture paths

> **Historical (source host, 2026-10-09).** Orbbec-only capture setup. On the rig host the Astra depth path is rebuilt from the official ros_astra_camera OpenNI2 redistributable; see [RIG_HOST.md](RIG_HOST.md).

Updated 2026-10-09. Both legacy OpenNI2 and OrbbecSDK v1 capture real Astra Pro depth on this Ubuntu 24.04.5 x86_64/kernel 7.0 host. P0 remains incomplete because calibration and exposure alignment are unqualified.

## Working baseline

From `/media/pc/WS_DATA/!WORK/REMA`:

```bash
python3 tools/build_capture.py
python3 tools/capture_astra.py --seconds 60
# Use the new Dataset path printed above:
python3 tools/launch.py --seconds 120 -- /home/pc/rema-data/envs/p0/bin/python tools/analyze_capture.py /home/pc/rema-data/recordings/DATASET
```

OpenNI2.3.0.86-beta6 provides depth; V4L2 opens only USB2bc5:0501 RGB. `capture_astra.py` loads the local `astra_usb_compat.so` only for this capture process. It corrects the SDK's one-byte language-table request on 2bc5:0403 by reading the full descriptor and returning only the requested byte. The returned descriptor is04030904. Device IDs, vendor commands, response IDs and image/depth data are not rewritten. System libusb remains unchanged. Compile uses the header shipped in the isolated SDK source bundle; no full ROS install is needed.

Confirmed60s dataset: `/home/pc/rema-data/recordings/astra_20261009_142049_369857`.1797 RGB frames at 30.05 measured arrivalFPS and 1783 depth at 29.69 FPS; no observed frame-ID gaps, all payloads checked and deterministic replay passed. This transport recording is not a guided motion/actor or anchoring acceptance test. No software timestamp metric is claimed as physical latency or exposure skew.

## OrbbecSDK v1 alternative

The user's repository, https://github.com/orbbec/OrbbecSDK , supplies v1.10.37. Its live test recognizes original Astra Pro PID 0403 despite the main support table omitting that name. The distributed udev rules also include0403. Basic depth streaming works without the OpenNI descriptor workaround. Use advertised 640x480/30 Y11 (SDK unpacks to16bit, reported depth scale 1 mm/unit); requesting Y16 as the input profile failed because it is not advertised. Five-second check:148 frames, no ID gaps, monotonic SDK timestamps. Do not assume the SDK's converted timestamps are native exposure times. Frame synchronization is reported unsupported. SDK stop produced cancellation warnings while returning success; preserve logs and assess during future soak.

```bash
python3 tools/build_orbbec_v1_checks.py
python3 tools/launch.py --seconds 30 -- /home/pc/rema-data/work/orbbec_v1_inventory
python3 tools/launch.py --seconds 35 -- /home/pc/rema-data/work/orbbec_v1_depth_check
```

Source release: https://github.com/orbbec/OrbbecSDK/releases/tag/v1.10.37 . No installer or broad udev script was run. Only the user's earlier exact 0403 permission rule is installed. SDKv2 is not selected for this legacy protocol.

## Calibration and timing

OpenNI reports protocol firmware 5.8.22; OrbbecSDK reports device string RD108C-009. These are separately labeled fields, not interchangeable version comparisons. The OpenNI camera-parameter query returns success but all floats are nonfinite; raw bytes are retained and calibration is rejected. Never use those values for reprojection. Obtain and physically validate RGB/depth intrinsics, distortion and depth_to_rgb transform in metres with a measured target. Alternative SDK calibration access remains to be checked. RGB sensor exposure timestamp remains unavailable in the current recorder; retain V4L2 flags/timestamps separately from host arrival. Nearest-arrival pairing is diagnostic only.

## Resource and evidence policy

CPU-only, 4 GiB, max 600 s, one owner, OOM stop, zero swap allowance and 25 GiB free reserve on both volumes. GPU admission guards respect other projects' locks. No training, firmware update or other-project changes occurred.11 tests cover geometry, recording integrity and compatibility-wrapper scope/error propagation. Preview generation initially exceeded the 64-task cap because OpenCV spawned 128 threads; corrected to 2 threads without raising the cap. Initial tracing shim had a symbol-resolution crash; corrected and retained as failed diagnostic evidence. A targeted USBreset caused descriptor enumeration errors, recovered by physical reconnect; it is not part of the working capture procedure.

Complete evidence: `/home/pc/rema-data/reports/driver_sdk_validation.json`, local SDK logs, raw recordings and dependency hashes. User-facing RGB/depth preview: `/home/pc/Documents/Codex/2026-10-09/rema/outputs/rema_p0_rgb_depth.mp4`. No hardware failure was established. Next: calibration/timing, then the planned moving-camera test.
