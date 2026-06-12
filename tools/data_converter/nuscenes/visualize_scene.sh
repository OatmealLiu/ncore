#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch the NCore interactive 3D viewer (//tools/ncore_vis, viser-based) for a converted
# nuScenes scene, and print how to open it from your laptop over an SSH tunnel.
#
# The viewer renders: camera frustums (RGB) with optional in-image lidar projection + cuboid
# edge overlay; the lidar point cloud (colorizable / fusable); 3D wireframe cuboids; and the
# rig trajectory. It runs a web server on THIS node; you tunnel its port to your laptop.
#
# Runs on whatever machine has the converted scene + this repo + bazel (cluster node OR the
# local H100 dev box). Default points at the local H100 validation copy; override OUTPUT_DIR
# for the cluster. A GPU is optional -- projections use CUDA if available, else CPU.
#
# Usage:
#   tools/data_converter/nuscenes/visualize_scene.sh
#   OUTPUT_DIR=/lustre/.../ncoreV4/nuscenes SCENE_NAME=scene-0007 PORT=8090 \
#     tools/data_converter/nuscenes/visualize_scene.sh

set -uo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/localhome/local-mingxuanl/miuspace/datasets/ncoreV4/nuscenes_fix_validation}"
SCENE_NAME="${SCENE_NAME:-}"
PORT="${PORT:-8080}"
TARGET="//tools/ncore_vis"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

command -v bazel >/dev/null 2>&1 || { echo "ERROR: 'bazel' not found on PATH." >&2; exit 1; }

# Pick the scene: explicit SCENE_NAME, else the first converted scene-* dir.
if [[ -z "$SCENE_NAME" ]]; then
    SCENE_NAME="$(cd "$OUTPUT_DIR" 2>/dev/null && ls -d scene-* 2>/dev/null | sort | head -1)"
fi
[[ -n "$SCENE_NAME" ]] || { echo "ERROR: no converted scene-* dir under $OUTPUT_DIR" >&2; exit 1; }

META="$OUTPUT_DIR/$SCENE_NAME/$SCENE_NAME.json"
[[ -f "$META" ]] || { echo "ERROR: sequence-meta JSON not found: $META" >&2; exit 1; }

NODE="$(hostname -f 2>/dev/null || hostname)"
cat <<EOF

================ NCore viewer ================
scene : $SCENE_NAME
meta  : $META
serves: ${NODE}:${PORT}

--- View it on your LAPTOP ---
In a separate terminal ON YOUR LAPTOP, forward the port, then open the browser:

  # if your laptop can ssh straight to this node:
  ssh -N -L ${PORT}:localhost:${PORT} ${NODE}

  # or hop through a cluster login node that can reach this node:
  ssh -N -L ${PORT}:${NODE}:${PORT} <your-cluster-login-alias>

  then open:  http://localhost:${PORT}

--- Once the browser is open (verify everything visually) ---
  * Sequence tab : scrub the Reference Frame slider / Play to step through the ~20s scene;
                   toggle Rig Trajectory + Show Rig Frame.
  * Lidars tab   : enable lidar_top (single 360 deg HDL-32E, structured spinning model); try
                   Color Style = Range/Height/Timestamp; turn on Fuse to accumulate -- a
                   coherent road/scene = poses + extrinsics + lidar model OK.
  * Cameras tab  : see the 6 RGB frustums. In Overlay Settings, turn on "Project Lidar" and
                   "Overlay Cuboids" -- if the projected points + box edges line up with the
                   image, the extrinsics/intrinsics/poses are all correct.
  * Cuboids tab  : 3D boxes should sit on the lidar points of real objects. NOTE: cuboids are
                   KEYFRAME-only (~2 Hz) while the cameras/lidar are stored at the full sweep
                   cadence (~12-20 Hz), so boxes appear only on keyframe reference frames and
                   vanish on the in-between sweeps -- expected.
==============================================

Starting the viewer (Ctrl-C to stop)...
EOF

exec bazel run "$TARGET" -- --host 0.0.0.0 --port "$PORT" v4 --component-group="$META"
