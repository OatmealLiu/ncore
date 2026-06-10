#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert nuScenes to NCore V4 one scene at a time, isolating failures.
#
# Each scene is converted in its own `bazel run` invocation. If a scene crashes
# (e.g. the known optimize_model azimuth-span bug on some scenes), it is recorded
# as FAIL and the run continues with the next scene -- so a single bad scene no
# longer aborts the whole dataset. At the end you get a tally of OK vs FAIL.
#
# Resumable: completed scenes are recorded in succeeded.txt and skipped on re-run.
# After fixing the bug, just re-run this script and only the previously-failed
# scenes will be retried.
#
# Usage:
#   tools/data_converter/nuscenes/convert_scene_by_scene.sh
#
# Long job (850 scenes). Run it detached so an SSH drop won't kill it, e.g.:
#   tmux new -s nusc 'tools/data_converter/nuscenes/convert_scene_by_scene.sh; bash'
#   # or:  nohup tools/data_converter/nuscenes/convert_scene_by_scene.sh &> nohup.out &
#
# Override any default via environment variables:
#   ROOT_DIR=...  OUTPUT_DIR=...  VERSION=...  LOG_DIR=...
#   EXTRA_ARGS="--lidar-model-optimization-passes 0"   # e.g. to dodge the bug entirely
#   CLEAN_PARTIAL=0                                     # keep partial output of failed scenes
#   ONLY_FAILED=1                                       # retry only scenes in failed.txt

set -uo pipefail

# --- Configuration (override via env) ----------------------------------------
ROOT_DIR="${ROOT_DIR:-/lustre/fs11/portfolios/nvr/projects/nvr_dvl_research/datasets/nuscenes}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_dvl_research/users/mingxuanl/datasets/ncoreV4/nuscenes}"
VERSION="${VERSION:-v1.0-trainval}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR%/}/_conversion_logs}"

# Lidar-model quality settings. These mirror the converter's defaults / the
# README "recommended for best quality". Optimization is ON so good scenes get
# the refined model; scenes that hit the optimizer bug fail and are recorded.
# Set EXTRA_ARGS="--lidar-model-optimization-passes 0" to avoid the bug entirely.
EXTRA_ARGS="${EXTRA_ARGS:---lidar-model-source nominal --lidar-model-resolution 4 --lidar-model-optimization-passes 1}"

# Remove a scene's partial output dir before a (re)attempt and after a failure.
CLEAN_PARTIAL="${CLEAN_PARTIAL:-1}"
# Only retry scenes listed in failed.txt (set to 1 after a first full pass).
ONLY_FAILED="${ONLY_FAILED:-0}"

TARGET="//tools/data_converter/nuscenes"

# --- Locate repo root & tooling ----------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v bazel >/dev/null 2>&1; then
    echo "ERROR: 'bazel' not found on PATH." >&2
    exit 1
fi

SCENE_JSON="${ROOT_DIR%/}/${VERSION}/scene.json"
if [[ ! -f "$SCENE_JSON" ]]; then
    echo "ERROR: scene metadata not found: $SCENE_JSON" >&2
    echo "       Check ROOT_DIR ('$ROOT_DIR') and VERSION ('$VERSION')." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
SUCCEEDED="$LOG_DIR/succeeded.txt"
FAILED="$LOG_DIR/failed.txt"
RESULTS_CSV="$LOG_DIR/results.csv"
touch "$SUCCEEDED"
[[ -f "$RESULTS_CSV" ]] || echo "timestamp,scene,status,seconds" >"$RESULTS_CSV"

# --- Build the converter once (fail fast on build/network errors) ------------
echo "Building converter ($TARGET) ..."
if ! bazel build "$TARGET" 2>&1 | tail -n 20; then
    echo "ERROR: build failed. Resolve the build (network/deps) before converting." >&2
    exit 1
fi

# --- Resolve the list of scenes to process -----------------------------------
# Scene names in nuScenes are NOT contiguous (gaps + the 850/150 trainval/test
# split), so we read the actual names from scene.json and sort them ascending.
if [[ "$ONLY_FAILED" == "1" ]]; then
    if [[ ! -s "$FAILED" ]]; then
        echo "ONLY_FAILED=1 but $FAILED is empty -- nothing to retry."
        exit 0
    fi
    mapfile -t SCENES < <(sort -u "$FAILED")
    echo "Retrying ${#SCENES[@]} previously-failed scene(s) from $FAILED"
else
    mapfile -t SCENES < <(python3 - "$SCENE_JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for name in sorted(s["name"] for s in data):
    print(name)
PY
)
fi

N=${#SCENES[@]}
if [[ "$N" -eq 0 ]]; then
    echo "ERROR: no scenes resolved from $SCENE_JSON" >&2
    exit 1
fi

# --- Run ---------------------------------------------------------------------
OK_RUN=0; FAIL_RUN=0; SKIP_RUN=0; i=0
START_TS=$SECONDS

print_summary() {
    local succeeded_total failed_total
    succeeded_total=$(grep -cxFf <(printf '%s\n' "${SCENES[@]}") "$SUCCEEDED" 2>/dev/null || echo 0)
    failed_total=$(( N - succeeded_total ))
    # Rewrite failed.txt as the authoritative outstanding set (scenes not yet succeeded).
    comm -23 <(printf '%s\n' "${SCENES[@]}" | sort -u) <(sort -u "$SUCCEEDED") >"$FAILED"
    echo
    echo "==================== SUMMARY ===================="
    echo "Version            : $VERSION   (scenes: $N)"
    echo "This run           : OK=$OK_RUN  FAIL=$FAIL_RUN  SKIP=$SKIP_RUN  (elapsed $((SECONDS-START_TS))s)"
    echo "Cumulative         : SUCCEEDED=$succeeded_total  OUTSTANDING/FAILED=$failed_total"
    echo "Output dir         : $OUTPUT_DIR"
    echo "Logs               : $LOG_DIR"
    echo "  succeeded list   : $SUCCEEDED"
    echo "  failed list      : $FAILED"
    echo "  per-scene logs   : $LOG_DIR/<scene>.log"
    echo "  results csv      : $RESULTS_CSV"
    if [[ "$failed_total" -gt 0 ]]; then
        echo "Retry failures after the fix with:  ONLY_FAILED=1 $0"
    fi
    echo "================================================="
}

# Print a partial summary if interrupted (Ctrl-C / job timeout).
trap 'echo; echo "Interrupted."; print_summary; exit 130' INT TERM

echo "Converting $N scene(s) from '$VERSION' one by one ..."
echo "  root=$ROOT_DIR"
echo "  out =$OUTPUT_DIR"
echo "  args=$EXTRA_ARGS"
echo

for scene in "${SCENES[@]}"; do
    i=$((i+1))

    # Skip scenes already converted (resume support).
    if grep -qxF "$scene" "$SUCCEEDED"; then
        SKIP_RUN=$((SKIP_RUN+1))
        printf '[%4d/%4d] %-12s SKIP (already converted)\n' "$i" "$N" "$scene"
        continue
    fi

    # Clear any stale/partial output from a prior failed attempt for a clean retry.
    scene_out="${OUTPUT_DIR%/}/$scene"
    if [[ "$CLEAN_PARTIAL" == "1" && -d "$scene_out" ]]; then
        rm -rf "${scene_out:?}"
    fi

    log="$LOG_DIR/$scene.log"
    start=$SECONDS

    # shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
    if bazel run "$TARGET" -- \
            --root-dir "$ROOT_DIR" \
            --output-dir "$OUTPUT_DIR" \
            nuscenes-v4 \
            --version "$VERSION" \
            --scene-name "$scene" \
            $EXTRA_ARGS \
            >"$log" 2>&1; then
        status="OK"
        OK_RUN=$((OK_RUN+1))
        echo "$scene" >>"$SUCCEEDED"
    else
        status="FAIL"
        FAIL_RUN=$((FAIL_RUN+1))
        # Drop the incomplete output so it can't pollute later reads / retries.
        if [[ "$CLEAN_PARTIAL" == "1" && -d "$scene_out" ]]; then
            rm -rf "${scene_out:?}"
        fi
    fi

    dur=$((SECONDS-start))
    printf '%s,%s,%s,%d\n' "$(date -Is)" "$scene" "$status" "$dur" >>"$RESULTS_CSV"
    printf '[%4d/%4d] %-12s %-4s (%ds)  log: %s\n' "$i" "$N" "$scene" "$status" "$dur" "$log"
done

print_summary
