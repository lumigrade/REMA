# REMA operating instructions

## Authority and scope

The latest direct user request takes priority. Read PLAN_REMA.md, PROJECT_STATE.json and the latest WORKLOG before work. This is an independent project; do not inherit VirtualTryOn's training plan, lane budgets, watchers, checkpoints or release procedures.

The user requested a new research project and session, with an Orbbec RGB-D moving-camera prototype first: preserve the actor, replace the background with a stationary 3D grid and shaded blocks. Main-camera SDI integration follows. No furniture interaction, virtual occlusions, environment generation, 3DGS training or actor resynthesis in the first milestone.

## Execution

- Continue routine implementation and checks within the plan without asking for each small transition. Report genuine hardware or validation blockers.
- Device-independent CPU scaffolding and offline tests may proceed while the exact Orbbec model is pending. Confirm model/firmware support before selecting an SDK or installing hardware-specific dependencies. Do not upgrade firmware or make privileged system changes without a concrete need and user instruction.
- Use this project as the working directory explicitly if the chat initially opens elsewhere. Do not modify VirtualTryOn or DomeMatte, restart their jobs, or stop their processes.
- Inspect existing processes and physical GPU UUIDs before GPU work. Start with one GPU for live inference/rendering; use the other only for measured independent work. Avoid GPU contention. Do not launch duplicate owners or train to keep hardware busy.
- Implement an independent REMA memory-capped process launcher and GPU locks before heavy compute. Respect any existing shared per-GPU locks under /home/pc/vto-data/locks; do not treat these as permission to use another project's jobs. Determine RAM/VRAM caps from inventory, record finite job limits and stop on OOM/disk/error. Reserve 25 GB free space on both volumes.
- Store environments, downloads, logs, recordings and caches under /home/pc/rema-data on ext4. Use whole-file replacement through an ext4 temporary file for project files on WS_DATA. No raw footage uploads.
- Reuse pretrained inference components. No training run is authorized for the first grid prototype. Research use is intended; avoid repeatedly asking about commercial licensing.
- Validate each phase with its stated check. Keep a last working configuration. Never silently lower a failed threshold, discard difficult footage or label synthetic tests as hardware validation.

## Evidence and reporting

- Preserve frame IDs, sensor timestamps, host timestamps, calibration versions and model versions. Do not confuse capture arrival time with exposure time or throughput with latency.
- Distinguish confirmed device capabilities, assumptions and proposed targets. A real floor plane alone cannot determine all camera translation and yaw.
- Report progress with labeled H.264 motion videos, original beside output, correct aspect ratio and large image height. No still-image progress reports. Hardware changes require new calibration validation.
- Keep concise phase conclusions, exact reproduction commands and the next action in WORKLOG.md and PROJECT_STATE.json.
- No frequent periodic AI checks. If long work becomes necessary, use completion-based notifications or the user's requested schedule, without creating an automation just for this setup.
- Do not claim human visual acceptance, end-to-end latency qualification, or real camera validation before the relevant evidence exists.
