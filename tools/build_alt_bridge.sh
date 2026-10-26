#!/bin/sh
# Build tools/alt_pose_bridge.cpp against the Antilatency SDK 4.6.2 copy shipped with VPcamtracking.
# Output: $REMA_DATA/work/alt_pose_bridge (+ lib/ with the SDK .so files next to it).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${REMA_DATA:-$HOME/rema-data}
SDK=${ANTILATENCY_SDK:-$HOME/Desktop/VPcamtracking/sdk/src}
[ -d "$SDK/Api" ] || SDK=$ROOT/VPcamtracking/VPcamtracking/sdk/src
OUT=$DATA/work
mkdir -p "$OUT/lib"
cp "$SDK"/Bin/Linux/x86_64/*.so "$OUT/lib/"
g++ -std=c++17 -O2 -Wall -I"$SDK/Api" "$ROOT/tools/alt_pose_bridge.cpp" -o "$OUT/alt_pose_bridge" -ldl -pthread
echo "Built $OUT/alt_pose_bridge (run with --sdk-lib $OUT/lib)"
