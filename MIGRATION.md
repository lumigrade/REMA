# Continue REMA on the Antilatency Ubuntu machine

> **Historical (2026-10-12 handoff).** Migration to the rig host is complete; the destination turned out to have an RTX 5070, an Osprey VB-USL and a Canon 24-50. Current setup: [RIG_HOST.md](RIG_HOST.md).

The user selected Antilatency as the camera-pose source. Further ORB/RGB-D self-tracking tuning is no longer the primary task. Read AGENTS.md, the current decision at the top of PLAN_REMA.md, PROJECT_STATE.json, and the latest WORKLOG.md before implementation.

## Confirmed destination rig

- Camera: RED V-RAPTOR 8K VV.
- Lens: Angénieux Optimo 28–76 mm.
- Tracking: user's already-working Antilatency system; Alt mounted rigidly on the RED top handle.
- Depth: Orbbec Astra Pro mounted rigidly above the RED.
- GPU: RTX 5080 on the destination Ubuntu machine. Inspect its actual UUID, driver and ownership; do not reuse the source host's GPU identifiers or other-project locks.
- Capture: Osprey SDI-to-USB device; exact model, Linux behavior and negotiated pixel format/frame rate remain to be identified.
- Output: real-time REMA composite of RED foreground and a rendered environment. No LED wall.

The Optimo 28–76 is specified for S35+ with a 31.4 mm image diagonal. Select and verify an appropriate RED sensor crop rather than assuming full VistaVision coverage. Begin with fixed zoom and focus; later zoom/focus changes need lens calibration/profile updates and potentially tracking of lens settings. Manufacturer: https://www.angenieux.com/lenses/legacy-series/optimo-compact-lens-zoom-28-76/

## Repository and local assets

Repository: https://github.com/lumigrade/REMA (private).

The repository contains source, tests, dependency manifests, driver workaround source and project history/decisions. It does not contain installed environments, proprietary/distributed SDK binaries, pretrained model weights, recordings, credentials or machine-local configuration. Current external asset root is `/home/pc/rema-data` on the source host. See RUNTIME_ASSETS.json for asset provenance and hashes. No new software license is assigned by this upload; retain each dependency's own license.

Clone with the destination user's GitHub authentication:

```bash
git clone https://github.com/lumigrade/REMA.git
cd REMA
```

This is a source handoff, not a portable installer. Existing runtime scripts still reference the original host paths. Do not assume a clone is immediately runnable. Do not copy a Python virtual environment or source-host GPU lock configuration to the destination.

## First engineering steps

1. Inventory Ubuntu version, GPU/driver, exact Osprey model/USB mode, RED SDI format/crop/color pipeline, Antilatency SDK/Service version and connection. Preserve the working Antilatency environment configuration and firmware.
2. Externalize project/data/output roots and display settings. Audit `rema/` and `tools/` for `/home/pc`, `WS_DATA` and source-host GPU UUIDs. Preserve other-project resource admission checks while adapting them to the destination. Configure finite memory/runtime/VRAM limits before inference.
3. Recreate Python dependencies from requirements-gui.txt and, for GPU execution, assess requirements-live-gpu.lock against the RTX 5080 environment. The existing lock records the source RTX 3090 runtime; RTX 5080 compatibility has not been validated. Rebuild native depth/USB libraries from source. Install OpenNI2 separately for legacy Astra metric depth; retain the exact-device descriptor workaround and do not write the laser-enable control during depth startup.
4. Restore/download model assets using RUNTIME_ASSETS.json and verify hashes. Transfer calibration/recording evidence separately over a private local route if needed. Current Orbbec calibration remains provisional and does not qualify the new RED rig.
5. Implement the Antilatency pose adapter (units, axes, validity, world origin, timestamped history). Official native Linux sample: https://www.developers.antilatency.com/Software/Samples_en.html . Verify the installed API version rather than assuming the reference version.
6. Bring up RED SDI capture and a tracked grid before matting. Calibrate RED lens/crop, Alt-to-camera offset and video-to-pose timing. Match pose to RED exposure; newest/predicted pose alone does not correct delayed video.
7. Run RED-frame matting on one available GPU, then add Orbbec depth reprojected into the RED view. Missing/de-occluded depth must not erase confident actor pixels. Verify simultaneous tracking/depth operation.
8. Measure glass-to-glass latency and stage/frame-age telemetry. Test static camera/actor motion, camera motion/static subject and combined motion. Avoid stale queues and CPU/GPU round trips. Verify tracking loss/recovery explicitly.

## Current evidence and limits

Masking works according to the user. Legacy tracking loses lock and has excessive latency. The latest CPU live sample measured 12.2 fps; frame throughput is not a physical latency measurement. Offline refinement improved three non-fitting evaluation captures from 7.64/5.73/8.60 px p95 to 3.69/2.87/2.87 px. Full calibration remains unqualified because one exceeds the unchanged 3 px threshold. No active.json was written. Preserve failures and never use training captures to qualify their own fit.

105 software tests passed on the source machine. Antilatency, RED SDI, the new rig and destination hardware have not yet been tested by REMA. Existing source-host commands and historic measurements in P0_SETUP.md/WORKLOG.md are evidence, not destination setup instructions.
