# REMA project outline

## Goal

Virtual production without an LED volume: keep the real actor from a moving cinema camera, remove the real environment and replace it with a virtual one that stays anchored in the room — in real time, on one workstation. First milestone: a metric floor grid and simple blocks; current work: photoreal 3D Gaussian Splatting (3DGS) environments.

## Rig (current)

| Role | Hardware | Notes |
|---|---|---|
| Picture | RED V-RAPTOR 8K VV, Canon 24-50 (IS and AF off) | SDI monitor output |
| Capture | Osprey VB-USL SDI→USB 3 bridge | 1360×768 YUYV, ~14.8 fps measured over USB (advertised 25) |
| Camera tracking | Antilatency Alt on the top handle, ceiling LED truss (12 LEDs, 3.1 m square) | 250 Hz raw pose via REMA's SDK bridge |
| Depth | Orbbec Astra Pro above the camera | live depth stream; RED alignment pending |
| Compute | Ubuntu 24.04, RTX 5070 12 GB | matte, 3DGS rendering, H.265 encoding on the GPU |

## Architecture

```
capture (V4L2, kernel SOF timestamp + sequence) ─┐
pose history (250 Hz, CLOCK_MONOTONIC) ──────────┼─▶ pose @ (frame time − video delay) ─▶ camera_from_world
calibration profile (lens K, distortion, Alt→camera, timing) ┘                               │
                                                                                              ▼
                       RVM matte (CUDA) ◀── frame            background: 3DGS (OpenGL) or grid ◀──┘
                                 └──────────▶ composite (actor original pixels over background)
                                                         ├─▶ operator view (fullscreen, PiPs, tracking bar)
                                                         └─▶ H.265 take (NVENC) + metadata
```

Design rules kept throughout:
- **One clock.** Video, pose and depth timestamps are CLOCK_MONOTONIC; arrival time, kernel start-of-frame and exposure are never confused.
- **Pose matched to the frame.** The pose is interpolated to each frame's estimated capture time (measured video delay), never "latest pose".
- **No silent staleness.** Invalid tracking is either shown as the raw camera or held and flagged (`--tracking-loss`); never presented as tracked.
- **Measure, then claim.** Calibrations report held-out errors; thresholds are not lowered after seeing results; synthetic tests are labelled synthetic.

## Coordinate conventions (`rema/pose.py`)

- Antilatency raw pose: left-handed, X right, Y up, Z forward (metres).
- S frame (VPcamtracking calibration): Antilatency with Y negated (right-handed).
- REMA world W: right-handed, X right, Y forward, Z up, floor at z = 0.
- Camera: OpenCV (x right, y down, z optical axis).
- REMA uses the raw tracker pose plus its own Alt→camera transform; an Antilatency placement is never applied on top.

## Phases and status

| Phase | Content | Status (2026-10-26) |
|---|---|---|
| P0 | Hardware bring-up: capture, tracking, depth, host setup | Done on the rig host (device lookup by name, udev rules, environments) |
| A | Pose adapter: units, axes, validity, timestamps, loss handling | Done (`rema/pose.py`, 250 Hz bridge, interpolation, quality score) |
| B | RED + Antilatency grid: lens, Alt→camera, timing, registration | Prototype done: lens 0.34 px held-out; nodal 0.21° held-out; delay 76–80 ms; registration ~3 px (slow motion). **Not qualified**: fast motion, floor check vs. physical marks, longer soak pending |
| C | Actor matte on RED frames + Astra depth assistance | RGB matte live (RVM, 9–13 ms). Depth streams live (PiP); Astra→RED calibration and depth-assisted matte pending |
| D | Latency and robustness | Software stage timings and dropout handling done; glass-to-glass latency not yet measured; Osprey frame rate limits the pipeline |
| P5 | Environment representation: 3DGS | Started: OpenGL splat renderer (1 M splats, 11–17 ms), sample scene aligned and live; own location capture next |

## Next steps

1. **Capture path**: the Osprey VB-USL caps the pipeline at ~14.8 fps / 1360×768 over USB. A PCIe SDI card would enable full-rate 1080p and lower latency.
2. **Depth-assisted matting**: calibrate the Astra to the RED view (same ChArUco workflow), reproject depth, use it to clean matte edges and reject background.
3. **Qualification**: fast-motion registration take, physical floor-mark check, 10-minute soak with dropout statistics, glass-to-glass latency measurement.
4. **3DGS**: the user's own location capture with a measured reference for scale; alignment tool with live overlay; view-dependent colour (spherical harmonics) once `.ply` scenes are used.
5. **Tracking robustness**: more LEDs near the shooting position (the rig sits at the edge of the marker square); powered USB for the radio socket.

## Evidence

Every result above has a dated entry in [../WORKLOG.md](../WORKLOG.md) with recording paths, exact commands and numbers; machine-readable status is in [../PROJECT_STATE.json](../PROJECT_STATE.json). Raw recordings, models, SDKs and environments live outside the repository under `~/rema-data`.
