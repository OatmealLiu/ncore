# Bug report: `optimize_model` produces invalid lidar models (azimuth span ≥ 2π) on some nuScenes scenes

| | |
|---|---|
| **Date** | 2026-06-10 |
| **Component** | `tools/data_converter/structured_lidar_model.py` (`optimize_model`, `upsample_model`) |
| **Trigger** | nuScenes → NCore V4 conversion (default settings) |
| **Severity** | High — aborts conversion of affected scenes; scene-dependent |
| **Status** | Fixed (see *Fix* below) |

## Summary

The structured-spinning-lidar model optimizer can emit a column-azimuth array whose total angular
span reaches or exceeds a full revolution (2π). That violates the
`RowOffsetStructuredSpinningLidarModelParameters` invariant and raises an `AssertionError` while
constructing the model, **aborting the conversion of that scene**. Whether it triggers depends on the
per-scene lidar statistics, so the conversion succeeds on most scenes and fails on others (first
observed on `scene-0007` of `v1.0-trainval`).

## Symptom

```
File ".../tools/data_converter/structured_lidar_model.py", line 459, in optimize_model
    return RowOffsetStructuredSpinningLidarModelParameters(
File ".../ncore/impl/data/types.py", line 727, in __post_init__
    assert np.all(np.diff(relative_column_azimuths_rad.relative_angle_rad) > 0), (
AssertionError: Column azimuth angles must be sorted in the spinning direction so the diff
between relative angles of consecutive columns should always be positive
```

The per-frame processing (`Process lidar_top: 100%`) completes; the crash happens afterward, when the
multi-frame–optimized model is constructed.

## Root cause

The invariant at `ncore/impl/data/types.py:727` requires `relative_angle(col[0], col, direction)` to be
**strictly increasing**. Because `relative_angle` is taken modulo 2π
(`ncore/impl/data/util.py:186-191`), that holds **iff** the column azimuths are (a) strictly monotone
in the spinning direction **and** (b) span strictly less than one full revolution (2π). The moment the
cumulative span reaches 2π, the last relative angle wraps back toward 0 and the diff goes negative.

The CW monotonicity-enforcement block in `optimize_model` (pre-fix lines 450–457) could break the
invariant in two ways:

1. **Early-column flip + cascade.** After the per-column median correction (line 439), a noisy *early*
   column can be nudged just above `col[0]`. The line
   `column_azimuths[column_azimuths > column_azimuths[0]] -= 2*np.pi` then displaced that single column
   a full **2π** to the bottom of the range, and the subsequent **unbounded** clamp loop cascaded every
   later column below the `col[0] − 2π` boundary, giving a total span ≥ 2π.
2. **Endpoint span drift.** Even with no reordering, independent corrections on the first vs. last
   column inflate the total unwrapped span past 2π. The enforcement never bounded the total span, so
   this passed straight through.

The base nominal HDL-32E model already spans ~0.999 × 2π (margin to a full revolution ≈ **0.33°**), so
small per-column corrections are enough to cross 2π.

### Why it is scene-dependent and how common it is

The per-column corrections are medians of far-range, motion-decompensated point azimuths, so they vary
per scene. Under an i.i.d. per-column noise model (σ comparable to the converter's own quoted alignment
precision of 0.03–0.10°), the modeled crash rate is:

| per-column noise σ | native (1085 cols) | 4× upsampled (4340 cols) |
|---|---|---|
| 0.10° | ~1% | ~40% |
| 0.15° | ~6% | ~54% |
| 0.20° | ~11% | ~58% |
| 0.30° | ~24% | ~66% |

The 4× rate is much higher because upsampling shrinks the inter-column step to ~0.08°, *below* the
correction noise, so adjacent columns reorder constantly. The default/recommended converter settings
use 4× resolution with optimization enabled (`--lidar-model-resolution 4
--lidar-model-optimization-passes 1`), i.e. the most exposed configuration. Real corrections are likely
spatially correlated and sparser than i.i.d., so the true rate may be lower — but the failure is real
(`scene-0007`).

## Fix

Replace the buggy enforcement with a single shared helper, `_enforce_monotone_azimuths(...)`, used by
both `optimize_model` and `upsample_model`. It enforces monotonicity **in the continuous (`np.unwrap`'d)
domain** and **caps the total span just below 2π**:

- No single column is ever displaced by a full 2π — a noisy early column is pulled back by one
  `min_step` instead of being flipped across the whole revolution.
- The cumulative span is compressed to `< 2π` only when it would otherwise reach a full revolution
  (emitting a `WARNING`), so endpoint drift can no longer produce an invalid model.
- Both spin directions are handled (the previous block ran only for `cw` and silently did nothing for
  `ccw`).

```python
def _enforce_monotone_azimuths(column_azimuths, n_columns, spinning_direction):
    min_step = 2.0 * np.pi / n_columns / 100.0
    unwrapped = np.unwrap(np.asarray(column_azimuths, dtype=np.float64))
    sign = 1.0 if spinning_direction == "cw" else -1.0
    offset = sign * (unwrapped[0] - unwrapped)           # must increase from 0
    for i in range(1, len(offset)):                      # strict monotonicity, min gap
        if offset[i] <= offset[i - 1] + min_step:
            offset[i] = offset[i - 1] + min_step
    max_span = 2.0 * np.pi - min_step                    # keep below one revolution
    if offset[-1] > max_span:
        offset *= max_span / offset[-1]
    return unwrapped[0] - sign * offset
```

## Backward compatibility (does it change scenes that worked?)

**Genuinely well-formed models are unchanged.** For any input that is already strictly monotone with
span < 2π — every scene the converter accepted in the intended sub-step-correction regime — the
per-column step never fires and the span cap never triggers, so the helper returns the input untouched.
Verified **byte-for-byte identical** to the previous code on 500/500 random clean models (and on the
nominal model) at both 1× and 4×.

The fix changes output only on inputs that the previous code either (1) **crashed** on (column
reorderings — no prior behavior to preserve) or (2) silently distorted (span ≥ 2π, which the old code
wrapped by 2π producing a non-physical model). In both cases the new model is valid and more faithful.

## Validation

- Reproduced the crash through the **real** `optimize_model` (early-column overshoot) and confirmed the
  fix returns a valid model.
- Fix yields **0% failures** across all modeled noise levels at both 1× and 4× (vs. the table above).
- Existing test suite: **18/18 pass**; added 4 regression tests (early-column overshoot does not crash;
  clean input is a no-op; span > 2π is capped; ccw is repaired) → **22/22 pass**.
- `ruff` import-sort and format checks clean.

To verify end-to-end on the originally-failing scene:

```bash
bazel run //tools/data_converter/nuscenes -- \
    --root-dir /path/to/nuscenes --output-dir /path/to/out \
    nuscenes-v4 --version v1.0-trainval --scene-name scene-0007 \
    --lidar-model-source nominal --lidar-model-resolution 4 --lidar-model-optimization-passes 1
```

## Secondary findings (follow-ups, not fixed here)

- **`ncore/impl/data/types.py:730`** is a copy-paste bug: the second column-azimuth assertion re-checks
  `relative_row_elevations_rad.wrap_around_flag` (the *row elevations*) instead of the column relative
  angles, so the column wrap-around guard is effectively dead code. The strict-increase assertion at
  line 727 is the only effective column guard.
- At 4× resolution the inter-column step (~0.08°) is below the alignment noise, so the per-column median
  refinement is operating near its statistical floor. Worth revisiting (e.g. optimize at native
  resolution then upsample, or smooth/regularize the per-column corrections).

## Scope

Only the nuScenes converter exercises this code path. Other converters (Waymo, PAI) construct
`RowOffsetStructuredSpinningLidarModelParameters` differently and never call `optimize_model` /
`upsample_model`, so they are unaffected.
