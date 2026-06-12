#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert a SINGLE TruckDrive scene to NCore V4 -- a quick smoke test.
#
# Picks the first scene_* directory under the dataset root (override with SCENE_NAME=...).
# Run on a node that has both the dataset (lustre) and a warm bazel.
#
# Output goes to a SEPARATE validation dir by default (not the production tree), and the full
# conversion log is saved to <out>/_conversion_logs/<scene>.log. Override OUTPUT_DIR to convert
# into the production tree instead.
#
# Usage:
#   tools/data_converter/truckdrive/convert_one_scene.sh
#   SCENE_NAME=scene_28_3 tools/data_converter/truckdrive/convert_one_scene.sh

set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_dvl_research/users/mingxuanl/datasets/raw/TruckDrive}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_dvl_research/users/mingxuanl/datasets/ncoreV4/truckdrive_fix_validation}"
TARGET="//tools/data_converter/truckdrive:truckdrive"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

command -v bazel >/dev/null 2>&1 || { echo "ERROR: 'bazel' not found on PATH." >&2; exit 1; }
[[ -d "$ROOT_DIR" ]] || { echo "ERROR: dataset root not found: $ROOT_DIR" >&2; exit 1; }

# Pick the scene: explicit SCENE_NAME, else the first scene_* directory.
SCENE_NAME="${SCENE_NAME:-}"
if [[ -z "$SCENE_NAME" ]]; then
    SCENE_NAME="$(cd "$ROOT_DIR" && ls -d scene_* 2>/dev/null | sort | head -1)"
fi
[[ -n "$SCENE_NAME" ]] || { echo "ERROR: no scene_* directory found under $ROOT_DIR" >&2; exit 1; }

LOG_DIR="${OUTPUT_DIR%/}/_conversion_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SCENE_NAME}.log"

echo "Converting ONE TruckDrive scene:"
echo "  root  = $ROOT_DIR"
echo "  out   = $OUTPUT_DIR"
echo "  scene = $SCENE_NAME"
echo "  log   = $LOG"
echo

bazel run "$TARGET" -- \
    --root-dir "$ROOT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    truckdrive-v4 \
    --scene-name "$SCENE_NAME" 2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo
if [[ $status -eq 0 ]]; then
    echo "OK -> $OUTPUT_DIR/$SCENE_NAME   (log: $LOG)"
else
    echo "FAILED (exit $status)   (log: $LOG)"
fi
exit $status
