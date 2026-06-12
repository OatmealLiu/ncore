#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert TruckDrive to NCore V4 one scene at a time, isolating failures, with resume.
# Mirrors tools/data_converter/man_truckscenes/convert_scene_by_scene.sh.
#
# Each scene is converted in its own `bazel run` invocation, so one bad scene does not abort
# the whole dataset. Completed scenes are recorded in succeeded.txt and SKIPPED on re-run --
# so the canonical way to resume after an interruption is simply to re-run this script.
#
# TruckDrive has no scene.json; scenes are the `scene_*` directories under --root-dir (the
# same set the converter's get_sequence_ids globs). mmdet_annotations/ and sensor_sync/ are
# NOT scenes and are skipped by the glob.
#
# It also ABORTS EARLY on a disk-quota / no-space write error (or a run of instant failures),
# so a full disk stops the job in seconds instead of marking every remaining scene FAIL.
#
# Usage (long job -- run it detached):
#   tmux new -s td 'tools/data_converter/truckdrive/convert_scene_by_scene.sh; bash'
#   # or:  nohup tools/data_converter/truckdrive/convert_scene_by_scene.sh &> nohup.out &
#
# Override any default via environment variables:
#   ROOT_DIR=...  OUTPUT_DIR=...  LOG_DIR=...  SCENE_GLOB=scene_*
#   EXTRA_ARGS="--store-type directory"  # extra truckdrive-v4 flags (default: converter defaults)
#   CLEAN_PARTIAL=0                       # keep partial output of failed scenes
#   ONLY_FAILED=1                         # retry only scenes in failed.txt
#
# Disk budget: TruckDrive stores 11 cameras + 4 lidars (3 Ouster + Aeva) + radar + cuboids per
# scene at full frame cadence, so each scene is multi-GB. Convert one scene first
# (convert_one_scene.sh), check its size, and confirm the quota covers (per-scene size x #scenes)
# before launching the full sweep.

set -uo pipefail

# --- Configuration (override via env) ----------------------------------------
ROOT_DIR="${ROOT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_dvl_research/users/mingxuanl/datasets/raw/TruckDrive}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fs12/portfolios/nvr/projects/nvr_dvl_research/users/mingxuanl/datasets/ncoreV4/truckdrive}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR%/}/_conversion_logs}"
SCENE_GLOB="${SCENE_GLOB:-scene_*}"
# Extra per-scene converter flags (passed after the truckdrive-v4 subcommand). Empty by default
# = the converter defaults (itar store, separate-sensors profile, sequence-meta on).
EXTRA_ARGS="${EXTRA_ARGS:-}"
CLEAN_PARTIAL="${CLEAN_PARTIAL:-1}"
ONLY_FAILED="${ONLY_FAILED:-0}"
# Abort after this many consecutive instant (<5s) failures -- a backstop for systemic problems
# (full disk, broken build/env, missing devkit) that should not be retried for every scene.
MAX_CONSECUTIVE_FAST_FAILS="${MAX_CONSECUTIVE_FAST_FAILS:-5}"

TARGET="//tools/data_converter/truckdrive:truckdrive"

# --- Locate repo root & tooling ----------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v bazel >/dev/null 2>&1; then
    echo "ERROR: 'bazel' not found on PATH." >&2
    exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
    echo "ERROR: dataset root not found: $ROOT_DIR" >&2
    echo "       Check ROOT_DIR ('$ROOT_DIR')." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
SUCCEEDED="$LOG_DIR/succeeded.txt"
FAILED="$LOG_DIR/failed.txt"
RESULTS_CSV="$LOG_DIR/results.csv"
touch "$SUCCEEDED"
[[ -f "$RESULTS_CSV" ]] || echo "timestamp,scene,status,seconds" >"$RESULTS_CSV"

# --- Build the converter once (fail fast on build/env errors) ----------------
echo "Building converter ($TARGET) ..."
if ! bazel build "$TARGET" 2>&1 | tail -n 20; then
    echo "ERROR: build failed. Resolve the build first (e.g. ensure truckdrive-devkit/metainfo.json is present)." >&2
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
    mapfile -t SCENES < <(find "$ROOT_DIR" -maxdepth 1 -type d -name "$SCENE_GLOB" -printf '%f\n' 2>/dev/null | sort -V)
fi

N=${#SCENES[@]}
if [[ "$N" -eq 0 ]]; then
    echo "ERROR: no '$SCENE_GLOB' scene directories found under $ROOT_DIR" >&2
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
    echo "Dataset            : TruckDrive   (scenes: $N)"
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

echo "Converting $N TruckDrive scene(s) one by one ..."
echo "  root=$ROOT_DIR"
echo "  out =$OUTPUT_DIR"
echo "  args=${EXTRA_ARGS:-<converter defaults>}"
echo

for scene in "${SCENES[@]}"; do
    i=$((i+1))

    if grep -qxF "$scene" "$SUCCEEDED"; then
        SKIP_RUN=$((SKIP_RUN+1))
        printf '[%4d/%4d] %-24s SKIP (already converted)\n' "$i" "$N" "$scene"
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
            truckdrive-v4 \
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
    printf '[%4d/%4d] %-24s %-4s (%ds)  log: %s\n' "$i" "$N" "$scene" "$status" "$dur" "$log"

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
            echo "       (full disk, broken build/env, missing devkit). Inspect $log; not retrying the rest."
            print_summary
            exit 1
        fi
    fi
done

print_summary
