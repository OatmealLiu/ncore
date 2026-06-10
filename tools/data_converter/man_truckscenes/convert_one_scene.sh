#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert a SINGLE MAN TruckScenes scene to NCore V4 -- a quick smoke test.
#
# Picks the first scene from the dataset's scene.json (override with SCENE_NAME=...),
# and converts keyframes only by default (fast; set KEYFRAMES_ONLY=0 for the full sweep
# cadence). Run on a node that has both the dataset (lustre) and a warm bazel.
#
# Usage:
#   tools/data_converter/man_truckscenes/convert_one_scene.sh
#   SCENE_NAME=scene-<...> KEYFRAMES_ONLY=0 tools/data_converter/man_truckscenes/convert_one_scene.sh
#
# NOTE: the first build needs the pip lock regenerated to include pypcd4 (one-time):
#   bazel run //deps/pip:requirements_3_11
#   bazel build //... --lockfile_mode=update --nobuild
# (see this directory's README.md).

set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/lustre/fs11/portfolios/nvr/projects/nvr_dvl_research/datasets/man-truckscenes}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_dvl_research/users/mingxuanl/datasets/ncoreV4/man_truckscenes}"
VERSION="${VERSION:-v1.2-trainval}"
KEYFRAMES_ONLY="${KEYFRAMES_ONLY:-1}"
TARGET="//tools/data_converter/man_truckscenes:man_truckscenes"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

command -v bazel >/dev/null 2>&1 || { echo "ERROR: 'bazel' not found on PATH." >&2; exit 1; }

SCENE_JSON="${ROOT_DIR%/}/${VERSION}/scene.json"
[[ -f "$SCENE_JSON" ]] || { echo "ERROR: scene metadata not found: $SCENE_JSON" >&2; exit 1; }

# Pick the scene to convert: explicit SCENE_NAME, else the first scene in scene.json.
SCENE_NAME="${SCENE_NAME:-}"
if [[ -z "$SCENE_NAME" ]]; then
    SCENE_NAME="$(python3 - "$SCENE_JSON" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(sorted(s["name"] for s in data)[0])
PY
)"
fi

KF_FLAG="--keyframes-only"
[[ "$KEYFRAMES_ONLY" == "0" ]] && KF_FLAG="--all-sweeps"

echo "Converting ONE scene:"
echo "  root  = $ROOT_DIR"
echo "  out   = $OUTPUT_DIR"
echo "  ver   = $VERSION"
echo "  scene = $SCENE_NAME"
echo "  mode  = $KF_FLAG"
echo

bazel run "$TARGET" -- \
    --root-dir "$ROOT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    man-truckscenes-v4 \
    --version "$VERSION" \
    --scene-name "$SCENE_NAME" \
    "$KF_FLAG"

status=$?
echo
if [[ $status -eq 0 ]]; then
    echo "OK -> $OUTPUT_DIR/$SCENE_NAME"
else
    echo "FAILED (exit $status)"
fi
exit $status
