#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Project lidar point clouds onto each camera image and export the overlay PNGs, using
# //tools:ncore_project_pc_to_img (see docs/tools/data_vis.rst). This is the geometric
# correctness check: if the range-colored points land on the road/vehicles in the image,
# then the lidar extrinsics + camera extrinsics + camera intrinsics + rig trajectory are
# all correct together. (Especially useful for TruckDrive's mix of pinhole + fisheye cameras.)
#
# It loops over CAMERAS x LIDARS and writes:  <PROJ_DIR>/<camera>/<lidar>/*.png
#
# NOTE: the default is the full 11x4 grid (44 combos -- slow). Many pairs do NOT overlap (the
# rig has NO rear lidar, so rearward cameras get no points; a left lidar barely hits a right
# camera), so those PNGs are empty/uninformative -- that's expected, not a bug. For a quick
# check, subset to overlapping pairs, e.g. the forward view:
#   CAMERAS="forward_center_medium" LIDARS="aeva ouster_forward_center" \
#     tools/data_converter/truckdrive/project_lidar_to_cameras.sh
#
# Usage (runs anywhere with the scene + repo + bazel; a GPU is used if present):
#   tools/data_converter/truckdrive/project_lidar_to_cameras.sh
#   SCENE_NAME=scene_28_3 START=0 STOP=10 tools/data_converter/truckdrive/project_lidar_to_cameras.sh
#
# Related export tools (docs/tools/data_vis.rst), if you also want them:
#   bazel run //tools:ncore_export_ply        -- --output-dir <D> --source-id aeva --frame world v4 --component-group=<META>
#   bazel run //tools:ncore_export_colored_pc -- --output-dir <D> --source-id ouster_forward_center --camera-id forward_center_medium v4 --component-group=<META>
#   bazel run //tools:ncore_export_camera     -- --output-dir <D> --camera-id forward_center_medium v4 --component-group=<META>

set -uo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/localhome/local-mingxuanl/miuspace/datasets/ncoreV4/truckdrive_fix_validation}"
SCENE_NAME="${SCENE_NAME:-}"
CAMERAS="${CAMERAS:-forward_center_medium forward_left_narrow forward_left_wide forward_right_narrow forward_right_wide rearward_left_bottom_medium rearward_right_bottom_medium sideward_left_back_wide sideward_left_front_wide sideward_right_back_wide sideward_right_front_wide}"
LIDARS="${LIDARS:-aeva ouster_forward_center ouster_sideward_left ouster_sideward_right}"
START="${START:-0}"; STOP="${STOP:-3}"; STEP="${STEP:-1}"   # frames [START, STOP) by STEP
DEVICE="${DEVICE:-cuda}"        # cuda if a GPU is present, else set DEVICE=cpu
POSE="${POSE:-mean}"            # cameras are global-shutter, so 'mean' (single pose) is apt
POINT_SIZE="${POINT_SIZE:-2.0}"
RANGE_CYCLE="${RANGE_CYCLE:-25.0}"
TARGET="//tools:ncore_project_pc_to_img"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
command -v bazel >/dev/null 2>&1 || { echo "ERROR: 'bazel' not found on PATH." >&2; exit 1; }

if [[ -z "$SCENE_NAME" ]]; then
    SCENE_NAME="$(cd "$OUTPUT_DIR" 2>/dev/null && ls -d scene_* 2>/dev/null | sort -V | head -1)"
fi
[[ -n "$SCENE_NAME" ]] || { echo "ERROR: no converted scene_* dir under $OUTPUT_DIR" >&2; exit 1; }
META="$OUTPUT_DIR/$SCENE_NAME/$SCENE_NAME.json"
[[ -f "$META" ]] || { echo "ERROR: sequence-meta JSON not found: $META" >&2; exit 1; }

PROJ_DIR="${PROJ_DIR:-$OUTPUT_DIR/_projections/$SCENE_NAME}"
LOG_DIR="$PROJ_DIR/_logs"
mkdir -p "$LOG_DIR"

echo "Scene : $SCENE_NAME"
echo "Frames: [$START,$STOP) step $STEP   device=$DEVICE pose=$POSE"
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
                --start-frame "$START" --stop-frame "$STOP" --step-frame "$STEP" \
                --point-size "$POINT_SIZE" --range-cycle "$RANGE_CYCLE" \
                --output-dir "$out" \
                v4 --component-group="$META" >"$log" 2>&1; then
            n=$(ls "$out"/*.png 2>/dev/null | wc -l)
            printf '  OK   %-28s <- %-22s (%s imgs)\n' "$cam" "$lid" "$n"; ok=$((ok+1))
        else
            printf '  FAIL %-28s <- %-22s  see %s\n' "$cam" "$lid" "$log"; fail=$((fail+1))
        fi
    done
done

echo
echo "Done: $ok combos OK, $fail failed."
echo "Overlay PNGs: $PROJ_DIR/<camera>/<lidar>/*.png"
echo "Copy them to your laptop to view, e.g.:"
echo "  scp -r <this-host>:$PROJ_DIR /tmp/td_proj && open /tmp/td_proj   # (macOS)"
