# REMA

**Real-time Environment Matting and Anchoring**

Preserve a live actor while replacing the real environment with a 3D floor grid and shaded blocks that stay fixed in the room as the camera moves.

## Start here

- [PLAN_REMA.md](PLAN_REMA.md): scope, architecture, phases and validation.
- [AGENTS.md](AGENTS.md): operating instructions for this project.
- [PROJECT_STATE.json](PROJECT_STATE.json): current status and unresolved hardware questions.
- [WORKLOG.md](WORKLOG.md): implementation and validation record.

Project: `/media/pc/WS_DATA/!WORK/REMA`
Large files, environments, captures and logs: `/home/pc/rema-data`

Current direction (2026-10-12): RED V-RAPTOR 8K VV supplies SDI video, Antilatency Alt supplies camera pose, and the rigidly mounted Astra Pro supplies depth assistance. REMA composites the RED foreground over a rendered virtual background. See the current decision at the top of PLAN_REMA.md for rig calibration, frame/pose timing and the revised implementation sequence.

The existing Orbbec-only prototype preserves the actor but remains unqualified: camera tracking loses lock and latency needs improvement. Antilatency integration, RED SDI capture and the new rig calibration have not been implemented. No per-actor training is planned.

Migration to a private GitHub repository and another Ubuntu host is planned. This source tree still contains host-specific paths; moving it alone does not recreate the environments, model assets, SDKs or calibration evidence under the external data root. Native libraries must be rebuilt. Raw recordings and machine-specific assets remain outside Git. Destination: RTX 5080, Osprey SDI-USB capture and Angénieux Optimo 28–76 mm; exact capture model and RED crop remain to be verified. See [MIGRATION.md](MIGRATION.md) for the next-machine handoff and [RUNTIME_ASSETS.json](RUNTIME_ASSETS.json) for external assets.

For the working Astra driver setup, see [P0_SETUP.md](P0_SETUP.md). Original capture, calibration and tracking results remain in WORKLOG.md and PROJECT_STATE.json.
