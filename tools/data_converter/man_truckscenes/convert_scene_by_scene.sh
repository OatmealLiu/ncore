#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert MAN TruckScenes to NCore V4 one scene at a time, isolating failures, with resume.
# Mirrors tools/data_converter/nuscenes/convert_scene_by_scene.sh.
#
# Each scene is converted in its own `bazel run` invocation, so one bad scene does not abort
# the whole dataset. Completed scenes are recorded in succeeded.txt and SKIPPED on re-run --
# so the canonical way to resume after an interruption is simply to re-run this script.
#
# It also ABORTS EARLY on a disk-quota / no-space write error (or a run of instant failures),
# so a full disk stops the job in seconds instead of marking every remaining scene FAIL.
#
# Usage (long job: 598 scenes -- run it detached):
#   tmux new -s mts 'tools/data_converter/man_truckscenes/convert_scene_by_scene.sh; bash'
#   # or:  nohup tools/data_converter/man_truckscenes/convert_scene_by_scene.sh &> nohup.out &
#
# Override any default via environment variables:
#   ROOT_DIR=...  OUTPUT_DIR=...  VERSION=...  LOG_DIR=...
#   EXTRA_ARGS="--all-sweeps"     # default --keyframes-only (~2 Hz). --all-sweeps = full 10-20 Hz cadence (~5-9x data)
#   CLEAN_PARTIAL=0               # keep partial output of failed scenes
#   ONLY_FAILED=1                 # retry only scenes in failed.txt
#
# Disk budget: keyframes-only is ~0.3 GB/scene => ~170 GB for all 598 scenes; --all-sweeps is
# several times larger. Make sure the quota has headroom before launching.

set -uo pipefail

# --- Configuration (override via env) ----------------------------------------
ROOT_DIR="${ROOT_DIR:-/lustre/fs11/portfolios/nvr/projects/nvr_dvl_research/datasets/man-truckscenes}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_dvl_research/users/mingxuanl/datasets/ncoreV4/man_truckscenes}"
VERSION="${VERSION:-v1.2-trainval}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR%/}/_conversion_logs}"
# Per-scene converter flags (passed after the man-truckscenes-v4 subcommand). Default keyframes
# only (manageable); use EXTRA_ARGS="--all-sweeps" for the full sweep cadence.
EXTRA_ARGS="${EXTRA_ARGS:---keyframes-only}"
CLEAN_PARTIAL="${CLEAN_PARTIAL:-1}"
ONLY_FAILED="${ONLY_FAILED:-0}"
# Abort after this many consecutive instant (<5s) failures -- a backstop for systemic problems
# (full disk, broken build/env) that should not be retried 598 times.
MAX_CONSECUTIVE_FAST_FAILS="${MAX_CONSECUTIVE_FAST_FAILS:-5}"

TARGET="//tools/data_converter/man_truckscenes:man_truckscenes"

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

# --- Build the converter once (fail fast on build/network/lock errors) -------
echo "Building converter ($TARGET) ..."
if ! bazel build "$TARGET" 2>&1 | tail -n 20; then
    echo "ERROR: build failed. Resolve the build (e.g. regenerate the pip lock for pypcd4) first." >&2
    exit 1
fi

# --- Resolve the list of scenes to process -----------------------------------
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
OK_RUN=0; FAIL_RUN=0; SKIP_RUN=0; i=0; consecutive_fast_fails=0
START_TS=$SECONDS

print_summary() {
    local succeeded_total failed_total
    succeeded_total=$(grep -cxFf <(printf '%s\n' "${SCENES[@]}") "$SUCCEEDED" 2>/dev/null || echo 0)
    failed_total=$(( N - succeeded_total ))
    comm -23 <(printf '%s\n' "${SCENES[@]}" | sort -u) <(sort -u "$SUCCEEDED") >"$FAILED" 2>/dev/null || true
    echo
    echo "==================== SUMMARY ===================="
    echo "Version            : $VERSION   (scenes: $N)"
    echo "This run           : OK=$OK_RUN  FAIL=$FAIL_RUN  SKIP=$SKIP_RUN  (elapsed $((SECONDS-START_TS))s)"
    echo "Cumulative         : SUCCEEDED=$succeeded_total  OUTSTANDING/FAILED=$failed_total"
    echo "Output dir         : $OUTPUT_DIR"
    echo "Logs               : $LOG_DIR  (succeeded.txt / failed.txt / results.csv / <scene>.log)"
    if [[ "$failed_total" -gt 0 ]]; then
        echo "Resume / retry with:  $0   (skips succeeded; re-runs the rest)"
    fi
    echo "================================================="
}

trap 'echo; echo "Interrupted."; print_summary; exit 130' INT TERM

echo "Converting $N scene(s) from '$VERSION' one by one ..."
echo "  root=$ROOT_DIR"
echo "  out =$OUTPUT_DIR"
echo "  args=$EXTRA_ARGS"
echo

for scene in "${SCENES[@]}"; do
    i=$((i+1))

    if grep -qxF "$scene" "$SUCCEEDED"; then
        SKIP_RUN=$((SKIP_RUN+1))
        printf '[%4d/%4d] %-44s SKIP (already converted)\n' "$i" "$N" "$scene"
        continue
    fi

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
            man-truckscenes-v4 \
            --version "$VERSION" \
            --scene-name "$scene" \
            $EXTRA_ARGS \
            >"$log" 2>&1; then
        status="OK"
        OK_RUN=$((OK_RUN+1)); consecutive_fast_fails=0
        echo "$scene" >>"$SUCCEEDED"
    else
        status="FAIL"
        FAIL_RUN=$((FAIL_RUN+1))
        if [[ "$CLEAN_PARTIAL" == "1" && -d "$scene_out" ]]; then
            rm -rf "${scene_out:?}"
        fi
    fi

    dur=$((SECONDS-start))
    printf '%s,%s,%s,%d\n' "$(date -Is)" "$scene" "$status" "$dur" >>"$RESULTS_CSV" 2>/dev/null || true
    printf '[%4d/%4d] %-44s %-4s (%ds)  log: %s\n' "$i" "$N" "$scene" "$status" "$dur" "$log"

    # --- Disk-full / systemic-failure guard ----------------------------------
    if [[ "$status" == "FAIL" ]]; then
        if grep -qiE 'disk quota exceeded|no space left' "$log" 2>/dev/null; then
            echo
            echo "ABORT: disk quota / no space left while converting '$scene' (see $log)."
            echo "Free space or raise the quota, then re-run to resume."
            print_summary
            exit 1
        fi
        if [[ "$dur" -lt 5 ]]; then
            consecutive_fast_fails=$((consecutive_fast_fails+1))
        else
            consecutive_fast_fails=0
        fi
        if [[ "$consecutive_fast_fails" -ge "$MAX_CONSECUTIVE_FAST_FAILS" ]]; then
            echo
            echo "ABORT: $consecutive_fast_fails consecutive instant failures -- likely systemic"
            echo "       (full disk, broken build/env). Inspect $log; not retrying the rest."
            print_summary
            exit 1
        fi
    fi
done

print_summary
