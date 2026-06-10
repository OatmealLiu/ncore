# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**NVIDIA NCore** (`nvidia-ncore` on PyPI) — a data format, library, and tools for data-driven neural 3D reconstruction of robotics / autonomous-vehicle data. Three pillars: the **V4 component data format**, **GPU-accelerated sensor models** (camera + lidar), and **dataset converters** that map external AV datasets into the V4 format.

## Build system: Bazel (bzlmod)

Everything is built and tested through Bazel (invoke via the `bazelisk` wrapper; version pinned in `.bazelversion`). There is no `setup.py`/`pip install -e` dev flow — use Bazel targets.

```bash
bazel build //...                  # build everything (also runs the `ty` type-check aspect — see below)
bazel test //...                   # run all tests (both Python versions, includes GPU tests)
bazel test --config=no-gpu //...   # skip GPU tests (sets NCORE_NO_GPU_TESTS=1)
bazel run //:format                # auto-format (ruff for .py, buildifier for Bazel files)
bazel run //:format.check          # CI format gate — fails on violations
```

**Running a single test.** The `pytest_test` macro (`//bazel/pytest:defs.bzl`) generates one target *per Python version* from each test rule's `name`, suffixed `_3_11` and `_3_8`. So a rule `pytest_types` in `ncore/impl/data/BUILD.bazel` becomes two targets:

```bash
bazel test //ncore/impl/data:pytest_types_3_11    # 3.11 only
bazel test //ncore/impl/data:pytest_types_3_8     # 3.8 only
```

**Type checking is part of the build.** `ty` runs as a Bazel aspect on every `bazel build` (wired in `.bazelrc` → `//bazel/lint:linters.bzl%ty`, with `--fail_on_violation`). There is no separate type-check command — a build failure may be a type error.

**Docs** (needs the `pandoc` system package): `bazel build //docs:ncore` (output in `bazel-bin/docs/ncore_html/`) or `bazel run //docs:view_ncore` to build and open in a browser.

### Two Python versions are mandatory

The library must work on **both Python 3.8 and 3.11** (both toolchains are registered in `MODULE.bazel`; tests run against both). This is why you'll see `sys.version_info` guards and conditional dataclass kwargs (`slots`/`kw_only`) throughout `impl/`. Don't use 3.9+-only syntax in library code without guarding it.

### Dependencies are fully pinned

`MODULE.bazel.lock` is committed and `.bazelrc` sets `--lockfile_mode=error`, so Bazel never hits the network to resolve modules. Pip deps live in `deps/pip/requirements_3_{8,11}.txt` and are referenced in BUILD files as `requirement("name")` from `@ncore_pip_deps`. **After editing `MODULE.bazel` or any `deps/pip/requirements_*` file**, regenerate the lock or the build breaks:

```bash
bazel build //... --lockfile_mode=update --nobuild   # then commit MODULE.bazel.lock
```

Some external archives (unit-test data, docs binaries) are pulled from GitHub Packages and require `~/.netrc` GitHub credentials with `read:packages` scope (see `CONTRIBUTING.md`). torch is patched (`deps/torch/preload-cuda-deps.patch`) to preload CUDA libs, because `--incompatible_strict_action_env` strips `LD_LIBRARY_PATH` from the sandbox.

## Architecture

### Public API is a thin shim over `impl/`

The top-level packages — `ncore/data/`, `ncore/data/v4/`, `ncore/sensors/`, `ncore/data_converter/` — contain **only `__init__.py` re-export shims** plus `test.py` import smoke-tests. All real code lives under **`ncore/impl/`**. The published wheel ships the whole `ncore` package, but external code (and examples/docs) import from the public packages (`from ncore.sensors import LidarModel`), **not** `ncore.impl.*`. When adding public API, implement under `impl/` and re-export from the matching `__init__.py` (and update its `__all__`).

`impl/` layout:

- **`impl/data/`** — core data types (`types.py`), storage backends (`stores.py`), abstract protocols + `SequenceLoader` adapters (`compat.py`), helpers (`util.py`).
- **`impl/data/v4/`** — the **V4 component-based format**. `components.py` is the heart of the repo (~2.2k lines): `ComponentReader`/`ComponentWriter`, `SequenceComponentGroups{Reader,Writer}`, and all component types (`Poses`, `Intrinsics`, `Masks`, `{Camera,Lidar,Radar}Sensor`, `Cuboids`, `PointClouds`, `CameraLabels`). `compat.py` provides `SequenceLoaderV4`.
- **`impl/sensors/`** — **GPU-accelerated (torch) sensor intrinsic models**. `camera.py` (FTheta, OpenCV pinhole/fisheye, windshield/external-distortion), `lidar.py` (structured spinning lidar), `common.py`, rolling-shutter. Models run on CPU **or** CUDA; tests parametrize over devices via a `_get_test_devices()` helper that drops GPU when `NCORE_NO_GPU_TESTS` is set.
- **`impl/common/`** — coordinate `transformations.py` and shared `util.py`.
- **`impl/data_converter/base.py`** — converter base classes (see below).

### V4 data format (the storage model)

A sequence is a collection of **component groups**; each group is a `zarr` store. Groups serialize two ways, both behind the same `SequenceComponentGroupsReader`/`Writer` API:

- **directory store** (`*.zarr`) — chunk-per-file; good for debugging/incremental edits.
- **indexed-tar** (`*.zarr.itar`) — NCore's custom single-file container: tar members + an appended compressed index giving O(1) key lookup and random seeks. Implemented as a drop-in `zarr.Store` (`IndexedTarStore` in `impl/data/stores.py`) and works over local **and** cloud paths via `UPath`/fsspec. Components are independently versioned/composable. See `docs/data/{formats,storage_and_access,conventions}.rst`.

### Tools (apps built on the library) — `tools/`

- **`tools/data_converter/`** — converters from external datasets (`kitti`, `nuscenes`, `waymo`, `pai`, `colmap`) into V4. A shared `click` group lives in `tools/data_converter/cli.py`; each dataset's `converter.py` subclasses `BaseDataConverter`/`FileBasedDataConverter` (from `ncore/impl/data_converter/base.py`), implements `get_sequence_ids` / `from_config` / `convert_sequence`, and registers a `click` subcommand that `main.py` imports for its side-effect. Each dataset exposes a binary aliased to the dataset name:
  ```bash
  bazel run //tools/data_converter/kitti:kitti -- <subcommand> --root-dir <raw> --output-dir <out>
  ```
  Converter integration tests are tagged `manual` (need external data / env vars like `KITTI_RAW_DIR`) and are excluded from `bazel test //...`. `structured_lidar_model.py` is a shared library for deriving structured spinning-lidar models from motion-compensated point clouds (e.g. nuScenes) — see `tools/data_converter/README.md`.
- **`tools/ncore_vis/`** — web visualizer: `bazel run //tools/ncore_vis:ncore_vis`.
- **`tools/ncore_*.py`** — standalone `py_binary` CLIs: export PLY / camera / colored point cloud, project point cloud to image, evaluate lidar model, dump sequence metadata.

## Contribution mechanics (CI will reject otherwise)

- **Conventional Commits** required (`feat`/`fix`/`docs`/`refactor`/`perf`/`test`/`build`/`ci`/`chore`...), validated by cocogitto. **Linear history, rebase-only**, gated by a GitHub merge queue.
- All commits must be **GPG-signed**.
- Every source file needs the **SPDX license header** (Apache-2.0, NVIDIA copyright) — copy from any existing file.
- Versioning is automatic from commits via cocogitto (`cog --config .cog.toml ...`, config in `.cog.toml`). The wheel version is injected through `--embed_label` in `.bazelrc` (currently `19.3.0`); cocogitto's pre-bump hook rewrites that line — don't hand-edit it.
- ruff config (line length **120**, isort with first-party `ncore`/`tools`) and `ty` settings live in `pyproject.toml`.
