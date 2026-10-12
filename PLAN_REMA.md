# REMA — moving-camera prototype plan

Date: 2026-10-09, Asia/Ho_Chi_Minh.
Name: **Real-time Environment Matting and Anchoring**.

## Current decision — Antilatency + RED rig (2026-10-12)

This decision supersedes the Orbbec-only tracking development sequence below. The user reports satisfactory live masking, unusable camera tracking and excessive update latency. The user already owns a working Antilatency virtual-production system on another Ubuntu machine and chooses it as REMA's authoritative camera pose source. Stop prioritizing ORB/SLAM recovery work; retain its implementation and evidence as a legacy prototype. Do not require the old Orbbec RGB/depth six-check workflow to pass before implementing Antilatency pose and RED SDI capture.

Target rig: RED V-RAPTOR 8K VV; Astra Pro rigidly mounted above the camera; Antilatency Alt rigidly mounted on its top handle. The existing Antilatency tracking environment supplies the world reference. REMA composites the real RED camera image with a virtual background in real time. An LED wall is not part of this workflow.

Signal responsibilities:
- RED SDI capture supplies the final actor pixels and the RGB input for matting. Preserve actual capture dimensions, crop, pixel aspect, transfer function, color range and frame metadata.
- Antilatency supplies world-space camera tracking through a calibrated Alt-to-camera transform. The native SDK has an official Linux C++ sample; confirm the installed SDK/Service versions and hardware interface on the destination host before integration. No claim of Ubuntu 24.04 qualification yet.
- Astra Pro supplies supporting depth and foreground confidence. Reproject its depth into the RED image with calibrated extrinsics and lens intrinsics. Handle non-overlapping views, missing depth and disocclusions conservatively; do not transfer the Orbbec RGB matte as though it were already a RED-view matte.
- REMA performs RGB matting, depth assistance, background rendering and compositing. Start with the grid/blocks scene and one selected SDI format; propose 1080p first, then evaluate higher capture/output resolutions. The camera's 8K recording capability is not the chosen live processing format.

Calibration remains necessary but serves the rig, not camera self-tracking:
1. RED lens intrinsics/distortion for the actual SDI framing. Start at fixed focal length and focus; changing zoom/focus may require lens profiles and updated optical-center offsets.
2. Rigid transform from Alt tracking coordinates to the RED camera's projection center, with coordinate convention and world-origin tests. Avoid applying Antilatency Placement twice if SDK output already includes it.
3. Rigid Orbbec-depth-to-RED transform and verified depth scale. The saved Orbbec internal calibration is supporting evidence, not a calibration of the new mounted RED rig.
4. Measure RED capture delay, depth delay and tracking clock/arrival offsets. Keep pose history and select/interpolate the pose for each RED exposure. SDK prediction is not a replacement for matching pose to the actual video frame. Device timestamps, host arrival timestamps and estimated exposure times must remain distinct.

Revised next milestones on the destination Ubuntu machine:
A. Portable source/setup migration, inventory capture card/GPU/Ubuntu/SDK versions, verify existing Antilatency tracking unchanged, implement a pose-provider adapter with validity, units, axes, timestamps and loss handling.
B. RED SDI + Antilatency grid-only test. Validate calibrated lens/Alt offset, metric scale, pan/dolly/return-to-start behavior and video/pose timing before involving the person mask.
C. Run matte inference on RED frames with one available GPU; add calibrated Orbbec depth support, verify sensor coexistence with the Antilatency environment, test actor motion and camera motion separately then together.
D. Measure stage latency and physical end-to-end delay, bounded buffer age and dropped frames at the selected format. Eliminate stale frame queues and unnecessary CPU/GPU copies. Retain current-RGB fallback on invalid tracking. Existing P2/P3 visual and accuracy requirements still apply to this rig.

Migration: user intends a private GitHub repository and continuing on the Ubuntu host with Antilatency connected. User authorized private lumigrade/REMA creation and source push. Destination GPU is RTX 5080, capture is Osprey SDI-USB (exact model pending), lens is Angenieux Optimo 28–76 mm (S35+, 31.4 mm image circle; verify RED crop). See MIGRATION.md. Source is checkpointed in Git and published to the private repository through authenticated browser commits; at least 23 Python/source files still contain host-specific paths/identifiers. Portability work must externalize project/data roots, display/session settings, device discovery and GPU admission; rebuild native libraries and Python environments on the destination. Commit source, tests, dependency manifests and documentation. Keep raw footage, environments, model/SDK binaries, logs, credentials and machine settings outside Git; transfer local evidence separately when needed. GitHub main contains the source handoff; CLI push was unavailable. Clone GitHub main on the destination because the source-host checkpoint has separate history. Current live prototype remains available and explicitly unqualified.

Primary references checked 2026-10-12:
- Linux native sample: https://www.developers.antilatency.com/Software/Samples_en.html
- Tracking state and prediction/placement semantics: https://developers.antilatency.com/Api/V4_6_1/Antilatency/Alt/Tracking/ITrackingCotask_en.html
- Placement configuration: https://www.developers.antilatency.com/Software/AntilatencyService/Placements_en.html

The older phases below record the original route and implemented prototype; where they require self-tracking before RED integration, this decision takes precedence.

## Goal and first deliverable

Keep the actual actor from a moving camera feed, erase the captured environment, and show a metric floor grid with a few solid, simply shaded blocks anchored in the room. The actor stays in front of the virtual content. The floor and blocks must react correctly to camera translation and rotation. A fixed-camera replacement is not the product goal; stationary shots are only diagnostic controls.

Initial capture uses the user's **Orbbec Astra Pro** RGB and depth streams. The device is now detected as USB `2bc5:0403` at 480M. Do not assume an IMU or hardware-synchronized colour/depth. The official legacy Astra Pro launch uses OpenNI depth plus a separate UVC colour path and defaults to 640×480/30 FPS; verify the actual firmware and available modes. At initial connected inventory the Astra colour endpoint was missing; reconnection restored it. Both RGB and depth now stream. OpenNI2 requires a local, exact-device language-descriptor compatibility wrapper; OrbbecSDK v1.10.37 independently recognizes and streams this original Astra Pro without that wrapper. See P0_SETUP.md and the driver validation report. Do not confuse Astra Pro with Astra Pro Plus, Astra 2, or Astra Mini Pro. Ubuntu host has an RTX 5080 and RTX 3090. Later, use the Orbbec as a rigidly mounted depth/tracking sensor beside a main camera delivered through SDI.

Defer background painting, scene training, 3DGS, furniture, virtual foreground occlusion, actor relighting and photorealistic contact shadows until this prototype passes. Do not retrain or synthesize the actor. No VTO work is resumed by this project.

## Feasibility and key constraints

RGB-D helps identify the actor, estimate metric room geometry and track camera movement. It does not itself deliver a stable world pose or a perfect hair matte. A plane constrains floor height/tilt, but cannot fix horizontal position and yaw; use static visual features and optionally measured markers. The moving actor must be excluded from camera tracking and static-map updates. Keep the raw room visible to tracking even though it is removed from the composited output.

A camera above another camera's lens has a different optical centre and field of view. Factory Orbbec depth-to-colour alignment applies to its own camera, not the SDI camera. Later integration requires both cameras' intrinsics/distortion, rigid extrinsics, time alignment and depth reprojection with validity handling. IMU availability, depth modes and synchronization depend on the exact Orbbec model/firmware.

Use a rigid bracket, overlapping fields of view and locked lens zoom/focus/stabilization settings during calibrated tests. Smaller sensor separation reduces parallax but cannot remove it. The SDI stage must explicitly test depth holes, disocclusions, rolling shutter and variable capture delay; internal Orbbec frame synchronization does not synchronize an external SDI camera.

## Architecture

1. Capture adapters: Orbbec RGB/depth initially; SDI colour plus Orbbec RGB/depth later.
2. Frame broker: pair calibrated streams by exposure timestamps where available; map device clocks to one monotonic host clock. Bound queues and discard stale frames explicitly.
3. Actor matting: a pretrained temporal RGB matte, assisted by aligned depth and actor identity/connected support. Use depth confidence to avoid erasing thin fingers, hair or areas with invalid depth. A single distance threshold is a diagnostic baseline only.
4. Camera pose: metric RGB-D odometry/SLAM from static room evidence, with the actor masked out. RTAB-Map is the initial integration candidate; retain a simple pose-provider interface. Use a calibrated ChArUco/ArUco board baseline for independent geometry verification and as an explicitly labeled tracking aid when the room lacks usable features. Do not claim marker-free tracking while markers supply pose.
5. World anchor: fit a floor from static depth, confirm its normal and origin, define metre units and stabilize in world coordinates. Do not re-anchor every frame to conceal drift. Handle SLAM map corrections explicitly so rendered objects do not jump silently.
6. Renderer: calibrated OpenGL viewport, floor lines at known spacing, 2–3 blocks with simple lighting. Pose and projection must correspond to the SAME captured RGB exposure. Do not pair an old actor frame with the latest pose or predict only the background into the future.
7. Compositor: original actor foreground/soft alpha over virtual background. Keep colour management consistent. No scene depth occlusion in milestone 1.
8. Recorder/telemetry: source, output, frame correspondence, pose confidence, depth validity, alpha and stage timings. Use recorded RGB-D replays to compare versions on identical motion.

Small validation utilities can use Python/OpenCV; the final critical capture/render path should avoid repeated GPU-to-CPU copies. Start on one GPU with preprocessing/pose work on CPU; profile before choosing dual-GPU partitioning. The second GPU can run independent offline matte comparisons if useful, but memory is not pooled automatically.

## Phases and checks

## Current provisional route — explicitly authorized by user

Professional printing is deferred. Use `python3 tools/run_gui.py --room-tracking` with separate `provisional.json`, never active measured calibration. The marker-free prototype uses static ORB RGB-D landmarks and bounded keyframe PnP, plus validated current-frame optical flow between descriptor refreshes. RVM matting now uses guarded RTX3090 GPU1; tracking/rendering stay on CPU, with explicit --cpu fallback. The original world origin survives tracking loss; relocalization must be confirmed twice, and loss displays current RGB. Floor is a frozen depth-plane candidate or a labelled camera-height preview. No measured metric, floor, drift or exposure-sync qualification is implied. User can reset world origin or estimate floor with5-second timed buttons. Retain existing strict measured-calibration checks for the new professional print; start a new session after remeasuring square size so23.17mm old-print views are not mixed with the new target. Full SLAM/loop closure and later SDI remain future integration, subject to physical validation.

### Current user-requested static-camera CPU preview

Before completing camera calibration/tracking, provide a native equal-split RGB/Composited GUI: RGB with raw depth inset, nearest separated person over an illustrative Blender-style grid. The user has now made both GPUs available. Check actual ownership and shared locks before any GPU work; use one GPU only when it improves measured performance. The current calibration and initial floor-target tracker remain CPU implementations. Use OpenNI2 metric depth, UVC RGB, CPU RVM temporal matting and Qt raster; keep PP-HumanSeg as a baseline. Preserve confident body pixels when uncalibrated depth disagrees; use depth for actor selection. Optional frozen empty-room reference rejects uncertain background pixels while the camera stays fixed. This diagnostic preview does not pass moving-camera P1/P2/P3; measured calibration, floor anchoring and exposure synchronization remain pending. Current live SDK v1 Y11 samples are not established metric depth and its conversion property is unsupported, so the GUI uses OpenNI2 DEPTH_1_MM.

### Easier new-print acquisition (2026-10-12)

Use `python3 tools/run_gui.py --calibrate --easy --new-print`. Confirm the new actual square dimension; the previous23.17mm target and saved views stay separate. Move and pause naturally: steady useful poses save automatically with a beep. The UI shows detected corners, pattern/depth readiness and coverage hints, rather than a prescribed flat overlay. At least24 diverse views are required; acquisition can continue to40 to fill gaps, with fit-only diversity computed excluding held-out views. Measured solver, six independent depth checks and timing-proxy qualification remain unchanged. Reflective print detection retries local contrast enhancement, preserving raw pixels; saturated information cannot be recovered. A separated rigid A4silhouette is still required for depth verification. Current user reserves5080 and both GPUs have other compute owners; calibration runs CPU-only.

### Guided calibration workflow now available

`python3 tools/run_gui.py --calibrate` captures a measured A4 ChArUco target. Default route uses RGB corners plus native depth board planes because this room/print has low IR pattern contrast. Depth rays come from OpenNI CoordinateConverter, with zero extra distortion an explicit assumption; a separate A4 depth-outline check must validate it. RGB/IR stereo remains available in the calibration library. The GUI now uses hands-free depth-plane acquisition: one measured-print confirmation and Start, then a10-second countdown, automatic board selection, per-pose distance/tilt/position prompts, steady-hold capture, automatic fitting, depth checks and timing sweep. All six independent checks are recorded, including numerical failures, before the app reports the complete set. Failed sets remain saved; strict activation still requires every check and diversity test to pass. A first-check miss no longer causes an automatic restart. Optional local voice cues accompany the large screen prompts. At least24 varied stationary poses, everyfourth held out, plus6 new depth plane/outline checks acrossdistance/tilt precede activation. A12-second front-facing axial sweep estimates an arrival-time alignment proxy, not exposure synchronization. Printed dimensions/backing outline must be physically checked. Current state: tools/synthetic checks ready, real calibration not yet qualified.

### P0 — identify hardware and establish a reproducible capture

- Determine Orbbec model, serial (local only), firmware, USB link, Ubuntu version, supported RGB/depth/IMU modes, intrinsic/extrinsic calibration and SDK compatibility.
- Select a supported 30 FPS RGB-D mode; begin with the legacy Astra Pro baseline of 640×480 if needed and test higher RGB modes only when advertised by this device. Keep native aspect ratio. Choose depth resolution from native supported modes. Record negotiated modes rather than requesting unsupported combinations.
- Record 60 seconds of paired frames and metadata with short camera movement and an actor entering/exiting. Save lossless depth plus scale and validity; compressed colour previews alone cannot serve as the replay dataset.
- Record native sensor timestamps, sequence IDs, exposure and dropped-frame counts. Distinguish device clock skew from frame arrival jitter.

Check: monotonic timestamps, deterministic replay, reported pair skew and drop rate, calibration physically checked at several depths. Proposed initial pair-skew target p95 <=10 ms when the device exposes meaningful comparable timestamps; otherwise mark sync unqualified and establish a measurement. Both endpoints now stream, and a 60-second simultaneous OpenNI2/UVC recording passed payload integrity and deterministic replay checks. OrbbecSDK v1.10.37 also passed a five-second depth check. P0 remains incomplete: factory camera-parameter values returned by OpenNI are nonfinite, exposure alignment is unqualified, and a guided moving-camera/actor take is still required. Basic streaming is not a full hardware/calibration pass.

### P1 — moving camera, stable world floor and blocks

Implemented first baseline: a calibrated fixed ChArUco floor target provides a direct per-frame world-to-camera pose. REMA Live projects a25cm metric grid and3shaded blocks, composites original actor RGB, and returns to current raw RGB on lost/ambiguous/jumping poses. Static clean-plate references are cleared/disabled while tracking. Qualified calibration is required to enable it. This is marker-assisted, requires the target to remain fixed and visible, and is only synthetically verified; the P1 physical checks below still apply.

- Establish known scale and floor coordinates from a measured board/floor references. Start with a short actor-free room pass to isolate calibration and tracking.
- Render a world grid and blocks during lateral translation, dolly-in/out, pan, tilt, lift and a small arc. Compare against physical marks not used as the only fitting evidence.
- Add pose confidence, tracking-loss handling and relocalization. If tracking fails, clearly mark/fallback to raw camera; never silently present stale anchoring as tracked.

Check: proposed targets are median held-out calibration reprojection <=1 px and p95 <=3 px at the selected resolution; floor scale error <=2%; return-to-start anchor displacement <=3 cm and <=1 degree on a measured 2-minute path. Report independent reference error separately from internal SLAM residuals. If no independent pose reference is available, label drift measurements as proxy/return tests. A smooth but drifting overlay fails.

### P2 — depth-assisted actor preservation while moving

- Compare a temporal matting baseline (RVM) with a quality alternative (MatAnyone 2) on the same local recordings when dependencies/hardware are ready. Choose by measured edge quality and runtime, not publication recency.
- Fuse reliable depth conservatively; retain RGB evidence where depth is invalid. Track the selected actor through depth changes, raised hands, loose sleeves and entry/exit.
- Mask the actor out of pose estimation/static mapping before integration. Reject tracking that follows the actor instead of the room.
- Composite the actor over the P1 grid; blocks stay behind the actor by design.

Check: opaque actor interiors retain source pixels before intentional colour conversion; report interior error and the transition-band policy. Review native-motion videos for hair, fingers, foot contact, motion blur, background leakage and alpha lag. Test actor-only movement, camera-only movement and simultaneous movement. No actor anatomy is synthesized. Depth thresholding alone is not a quality pass.

### P3 — qualify the complete Orbbec-only moving-camera prototype

Use at least these five 30–60 second takes, then a 10-minute soak: lateral track, forward/backward dolly, pan/tilt/height changes, small arc around a turning actor, and combined camera motion with fast arms plus actor entry/exit. Include low-texture and poor-depth cases and deliberate brief tracking loss. Temporary room markers are permitted but must be stated in each report.

Proposed initial engineering targets, not established results:
- 30 fresh composites/second at the negotiated capture mode; report rolling FPS and percent dropped/repeated frames.
- Software capture-to-present p95 <=150 ms and physically measured camera-motion-to-display p95 <200 ms. Software timestamps alone do not verify screen latency. Include grid response as well as actor motion in the physical check.
- P1 scale/anchoring targets retained with the actor present; bounded queue age; no uncontrolled pose jumps or relocalization resets.
- No persistent lost actor parts or visible real-background patches in the agreed normal-motion takes; record every rejected condition.

Deliver original/output side-by-side H.264 videos, reproduction config, timings p50/p95/max, failure counts and a phase conclusion. Stationary success cannot pass this phase. Any proposed threshold change must be explicit, not made after inspecting the result to manufacture a pass.

### P4 — main camera through SDI with rigid Orbbec mounting

Proceed after P3 passes and the hardware is identified. Lock a lens setting, calibrate the main camera with a measured board, estimate Orbbec-to-main transform from shared observations, and independently validate at multiple actor depths and image positions. Preserve any crop/resize transform and colour range from the capture card.

Map Orbbec depth points into the main camera with its calibrated projection and a z-buffer; mark missing/uncovered pixels instead of copying depth blindly. Estimate exposure-time offset and drift using a visible timed event or supported hardware synchronization; interpolate camera pose to the main exposure. Depth reprojection and camera pose synchronization are separate requirements.

Check: board reprojection and edge alignment at several depths; fast-pan/dolly tests for time mismatch; synchronized actor/virtual floor under movement; repeat P3 at the actual SDI format. Changing the bracket, zoom, focus or stabilization can invalidate calibration. Hardware genlock/trigger support is unconfirmed until model inventory. Do not promise perfect correspondence across different viewpoints.

### P5 — choose the virtual environment representation

Only after moving-camera preservation and anchoring pass: compare a prepared 3D scene, an image-derived layered environment, a scanned/learned 3DGS environment, and offline generated scene assets. Render a stable representation at runtime first. Continuous background generation is a separate decision based on the required behaviour. No scene training now.

## Hardware and resource policy

CPU: capture, bounded queues, calibration, room tracking and recording as measured. RTX 5080: primary matte/render candidate. RTX 3090: optional independent validation or heavier matte comparison. Do not force utilization when work is dependency-blocked; explain the blocking evidence instead. Separate environment and data roots from VTO and DomeMatte; never reuse their controllers. No data uploads.

## Needed from the user

Astra Pro model/connection are now confirmed. First resolve RGB and depth capture and ask for a short camera-motion recording when the app is ready. Later: main camera model, lens/zoom, SDI capture card and intended format, approximate subject distance and camera motion range. Capture instructions can be prepared before those answers, but real sensor validation cannot.

## Primary references checked on 2026-10-09

- Legacy Astra Pro OpenNI/UVC launch and actual default profiles: https://github.com/orbbec/ros_astra_camera/blob/main/launch/astra_pro.launch
- Legacy Astra driver integration: https://github.com/orbbec/ros_astra_camera

- Orbbec SDK device/firmware support and Linux compatibility: https://github.com/orbbec/OrbbecSDK_v2
- Orbbec alignment and timestamp-based supported-device frame synchronization: https://github.com/orbbec/OrbbecSDK_v2/blob/main/docs/tutorial/orbbecviewer.md
- RGB-D odometry and distinction between continuous local odometry and map corrections: https://github.com/introlab/rtabmap_ros/blob/ros2/rtabmap_odom/README.md
- ChArUco calibration: https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html
- Temporal foreground matting baseline and throughput caveats: https://github.com/PeterL1n/RobustVideoMatting
- Quality comparison candidate: https://github.com/pq-yang/MatAnyone2

Published component capability is evidence of feasibility, not a measurement of this integrated system. Research-only use was confirmed by the user.
