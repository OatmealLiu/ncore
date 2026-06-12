#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Project the nuScenes lidar point cloud onto each camera image and export the overlay PNGs,
# using //tools:ncore_project_pc_to_img (see docs/tools/data_vis.rst). This is the geometric
# correctness check: if the range-colored points land on the road/vehicles in the image, then
# the lidar extrinsics + camera extrinsics + camera intrinsics + rig trajectory + the
# structured spinning-lidar model are all correct together.
#
# lidar_top is a 360 deg roof lidar, so it projects usefully onto ALL 6 cameras. It loops
# CAMERAS x LIDARS and writes:  <PROJ_DIR>/<camera>/<lidar>/*.png
#
# By default (LIDAR_MODEL=1) it passes --lidar-model, so the projected points are RECONSTRUCTED
# from the structured spinning-lidar model (per-column azimuths x measured distance) -- this is
# what actually validates the lidar model (e.g. the azimuth-span fix). Set LIDAR_MODEL=0 to
# project the native stored rays instead (validates only extrinsics/intrinsics/poses, not the
# model). The native vs model comparison is itself a useful check: large divergence => bad model.
#
# Usage (runs anywhere with the scene + repo + bazel; a GPU is used if present):
#   tools/data_converter/nuscenes/project_lidar_to_cameras.sh                 # model-reconstructed (default)
#   LIDAR_MODEL=0 tools/data_converter/nuscenes/project_lidar_to_cameras.sh   # native rays
#   SCENE_NAME=scene-0007 CAMERAS="camera_front" START=0 STOP=10 \
#     tools/data_converter/nuscenes/project_lidar_to_cameras.sh
#
# Related export tools (docs/tools/data_vis.rst), if you also want them:
#   bazel run //tools:ncore_export_ply        -- --output-dir <D> --source-id lidar_top --frame world v4 --component-group=<META>
#   bazel run //tools:ncore_export_colored_pc -- --output-dir <D> --source-id lidar_top --camera-id camera_front v4 --component-group=<META>
#   bazel run //tools:ncore_export_camera     -- --output-dir <D> --camera-id camera_front v4 --component-group=<META>

set -uo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/localhome/local-mingxuanl/miuspace/datasets/ncoreV4/nuscenes_fix_validation}"
SCENE_NAME="${SCENE_NAME:-}"
CAMERAS="${CAMERAS:-camera_front camera_front_left camera_front_right camera_back camera_back_left camera_back_right}"
LIDARS="${LIDARS:-lidar_top}"
START="${START:-0}"; STOP="${STOP:-3}"; STEP="${STEP:-1}"   # frames [START, STOP) by STEP
DEVICE="${DEVICE:-cuda}"        # cuda if a GPU is present, else set DEVICE=cpu
POSE="${POSE:-mean}"            # cameras are global-shutter, so 'mean' (single pose) is apt
POINT_SIZE="${POINT_SIZE:-2.0}"
RANGE_CYCLE="${RANGE_CYCLE:-25.0}"
# LIDAR_MODEL=1 (default): reconstruct points FROM the structured spinning lidar model
# (model azimuths/elevations x measured distance) -- this exercises the lidar model itself
# (the azimuth-span fix). LIDAR_MODEL=0: project the native stored rays (does NOT test the
# model; validates only extrinsics/intrinsics/poses). nuScenes has a structured model; the
# truck datasets (ray bundles, no model) must use 0.
LIDAR_MODEL="${LIDAR_MODEL:-1}"
LM_FLAG="--lidar-model"
[[ "$LIDAR_MODEL" == "0" ]] && LM_FLAG="--no-lidar-model"
TARGET="//tools:ncore_project_pc_to_img"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
command -v bazel >/dev/null 2>&1 || { echo "ERROR: 'bazel' not found on PATH." >&2; exit 1; }

if [[ -z "$SCENE_NAME" ]]; then
    SCENE_NAME="$(cd "$OUTPUT_DIR" 2>/dev/null && ls -d scene-* 2>/dev/null | sort | head -1)"
fi
[[ -n "$SCENE_NAME" ]] || { echo "ERROR: no converted scene-* dir under $OUTPUT_DIR" >&2; exit 1; }
META="$OUTPUT_DIR/$SCENE_NAME/$SCENE_NAME.json"
[[ -f "$META" ]] || { echo "ERROR: sequence-meta JSON not found: $META" >&2; exit 1; }

PROJ_DIR="${PROJ_DIR:-$OUTPUT_DIR/_projections/$SCENE_NAME}"
LOG_DIR="$PROJ_DIR/_logs"
mkdir -p "$LOG_DIR"

echo "Scene : $SCENE_NAME"
echo "Frames: [$START,$STOP) step $STEP   device=$DEVICE pose=$POSE   $LM_FLAG"
echo "        ($([ "$LM_FLAG" = "--lidar-model" ] && echo 'reconstructs points FROM the structured lidar model -- validates the model' || echo 'native stored rays -- does NOT exercise the lidar model'))"
echo "Out   : $PROJ_DIR"
echo "Building $TARGET (first build pulls torch; may take a few minutes) ..."
if ! bazel build "$TARGET" 2>&1 | tail -n 5; then
    echo "ERROR: build failed." >&2; exit 1
fi

ok=0; fail=0
for cam in $CAMERAS; do
    for lid in $LIDARS; do
        out="$PROJ_DIR/$cam/$lid"; mkdir -p "$out"
        log="$LOG_DIR/${cam}__${lid}.log"
        # shellcheck disable=SC2086
        if bazel run "$TARGET" -- \
                --source-id "$lid" --camera-id "$cam" \
                --device "$DEVICE" --pose "$POSE" \
                "$LM_FLAG" \
                --start-frame "$START" --stop-frame "$STOP" --step-frame "$STEP" \
                --point-size "$POINT_SIZE" --range-cycle "$RANGE_CYCLE" \
                --output-dir "$out" \
                v4 --component-group="$META" >"$log" 2>&1; then
            n=$(ls "$out"/*.png 2>/dev/null | wc -l)
            printf '  OK   %-20s <- %-12s (%s imgs)\n' "$cam" "$lid" "$n"; ok=$((ok+1))
        else
            printf '  FAIL %-20s <- %-12s  see %s\n' "$cam" "$lid" "$log"; fail=$((fail+1))
        fi
    done
done

echo
echo "Done: $ok combos OK, $fail failed."
echo "Overlay PNGs: $PROJ_DIR/<camera>/<lidar>/*.png"
echo "Copy them to your laptop to view, e.g.:"
echo "  scp -r <this-host>:$PROJ_DIR /tmp/nusc_proj && open /tmp/nusc_proj   # (macOS)"
