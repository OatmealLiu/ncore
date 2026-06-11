#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Validate the structured-lidar azimuth-span-overflow fix on the SPECIFIC nuScenes scenes
# that crashed in the pre-fix run, plus scene-0007 (the originally-reported failure). This is
# the pre-flight check before the full 850-scene rerun: a clean pass here means the fix holds
# on exactly the scenes that used to break, so the full run should not hit the azimuth crash.
#
# It converts ONLY the listed scenes (no scene.json scan), one per `bazel run` so a bad scene
# can't abort the rest, with per-scene logs + an OK/FAIL summary, and exits non-zero if any
# scene fails. It uses the converter DEFAULTS -- empirical HDL-32E model, lidar_model_resolution=4,
# optimization_passes=1 -- i.e. the exact config convert_scene_by_scene.sh runs (the most
# crash-exposed configuration), so this is a faithful rehearsal of the full run.
#
# Output goes to a SEPARATE validation dir by default so it never mixes with / is mistaken for
# the production conversion output.
#
# Usage:
#   tools/data_converter/nuscenes/test_failed_scenes.sh
#   ROOT_DIR=... OUTPUT_DIR=... VERSION=... tools/data_converter/nuscenes/test_failed_scenes.sh
#   SCENES="scene-0007 scene-0055" tools/data_converter/nuscenes/test_failed_scenes.sh   # custom list

set -uo pipefail

# --- Configuration (override via env) ----------------------------------------
ROOT_DIR="${ROOT_DIR:-/lustre/fs11/portfolios/nvr/projects/nvr_dvl_research/datasets/nuscenes}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_dvl_research/users/mingxuanl/datasets/ncoreV4/nuscenes_fix_validation}"
VERSION="${VERSION:-v1.0-trainval}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR%/}/_conversion_logs}"
# Remove a scene's partial output before each attempt and after a failure.
CLEAN_PARTIAL="${CLEAN_PARTIAL:-1}"

# scene-0007 = originally-reported azimuth crash; the rest = the scenes recorded FAIL in the
# pre-fix run. Override SCENES to test a different set.
SCENES="${SCENES:-scene-0007 scene-0055 scene-0057 scene-0129 scene-0180 scene-0241 scene-0246 scene-0367 scene-0403 scene-0457}"

TARGET="//tools/data_converter/nuscenes"

# --- Locate repo root & tooling ----------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v bazel >/dev/null 2>&1; then
    echo "ERROR: 'bazel' not found on PATH." >&2
    exit 1
fi
if [[ ! -d "$ROOT_DIR" ]]; then
    echo "ERROR: nuScenes root not found: $ROOT_DIR" >&2
    exit 1
fi

read -ra SCENE_ARR <<< "$SCENES"
N=${#SCENE_ARR[@]}
if [[ "$N" -eq 0 ]]; then
    echo "ERROR: no scenes to test (SCENES is empty)." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
RESULTS_CSV="$LOG_DIR/results.csv"
[[ -f "$RESULTS_CSV" ]] || echo "timestamp,scene,status,seconds" >"$RESULTS_CSV"

# --- Build the converter once (fail fast on build/network/lock errors) -------
echo "Building converter ($TARGET) ..."
if ! bazel build "$TARGET" 2>&1 | tail -n 20; then
    echo "ERROR: build failed. Resolve the build before testing." >&2
    exit 1
fi

# --- Run ---------------------------------------------------------------------
OK=0; FAIL=0; i=0
FAILED_SCENES=()
START_TS=$SECONDS

echo
echo "Validating the azimuth-span fix on $N scene(s) from '$VERSION' (empirical/4x defaults) ..."
echo "  root=$ROOT_DIR"
echo "  out =$OUTPUT_DIR"
echo "  scenes: ${SCENE_ARR[*]}"
echo

for scene in "${SCENE_ARR[@]}"; do
    i=$((i+1))
    scene_out="${OUTPUT_DIR%/}/$scene"
    if [[ "$CLEAN_PARTIAL" == "1" && -d "$scene_out" ]]; then
        rm -rf "${scene_out:?}"
    fi

    log="$LOG_DIR/$scene.log"
    start=$SECONDS

    if bazel run "$TARGET" -- \
            --root-dir "$ROOT_DIR" \
            --output-dir "$OUTPUT_DIR" \
            nuscenes-v4 \
            --version "$VERSION" \
            --scene-name "$scene" \
            >"$log" 2>&1; then
        status="OK"
        OK=$((OK+1))
    else
        status="FAIL"
        FAIL=$((FAIL+1))
        FAILED_SCENES+=("$scene")
        if [[ "$CLEAN_PARTIAL" == "1" && -d "$scene_out" ]]; then
            rm -rf "${scene_out:?}"
        fi
    fi

    dur=$((SECONDS-start))
    printf '%s,%s,%s,%d\n' "$(date -Is)" "$scene" "$status" "$dur" >>"$RESULTS_CSV" 2>/dev/null || true
    printf '[%2d/%2d] %-12s %-4s (%ds)  log: %s\n' "$i" "$N" "$scene" "$status" "$dur" "$log"

    # Abort early on a disk-full / no-space write error -- nothing else will succeed.
    if [[ "$status" == "FAIL" ]] && grep -qiE 'disk quota exceeded|no space left' "$log" 2>/dev/null; then
        echo
        echo "ABORT: disk quota / no space left while converting '$scene' (see $log)."
        echo "Free space or raise the quota, then re-run."
        break
    fi
done

# --- Summary -----------------------------------------------------------------
echo
echo "==================== SUMMARY ===================="
echo "Version   : $VERSION"
echo "Tested    : $N   OK=$OK  FAIL=$FAIL   (elapsed $((SECONDS-START_TS))s)"
echo "Output    : $OUTPUT_DIR"
echo "Logs      : $LOG_DIR  (<scene>.log, results.csv)"
if [[ "$FAIL" -eq 0 ]]; then
    echo "RESULT    : ALL PASSED -- azimuth-span fix validated on the previously-failing scenes."
    echo "            Safe to launch the full run: tools/data_converter/nuscenes/convert_scene_by_scene.sh"
    echo "================================================="
    exit 0
else
    echo "RESULT    : ${FAIL} scene(s) STILL FAILING -- investigate before the full run:"
    for s in "${FAILED_SCENES[@]}"; do echo "              $s  -> $LOG_DIR/$s.log"; done
    echo "================================================="
    exit 1
fi
