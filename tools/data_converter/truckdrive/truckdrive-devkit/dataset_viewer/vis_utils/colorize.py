"""
Sensor-aware colorization helpers for the TruckDrive dataset.

The point-cloud layouts are:

Aeva joint lidar, float64, shape (-1, 11):
    0 x, 1 y, 2 z, 3 intensity, 4 velocity, 5 reflectivity,
    6 time_offset_ns, 7 sensor_id, 8 vx, 9 vy, 10 vz

Ouster lidar, float32, shape (-1, 7):
    0 x, 1 y, 2 z, 3 intensity, 4 rel_time_ns, 5 reflectivity, 6 ring

Conti542 joint radar, float64, shape (-1, 33):
    0 x, 1 y, 2 z, 3 rangerate, 4 rcs, 5 amplitude,
    6 range_radar, 7 azimuth, 8 elevation, 9 scan_id, 10 detection_id,
    11 quality, 12 is_rangerate_ambiguous, 13 rangerate_ambiguity,
    14 cluster_range_spread, 15 cluster_azimuth_spread, 16 cluster_elevation_spread,
    17 visited, 18 clustered, 19 pose_x, 20 pose_y, 21 pose_z,
    22 orientation_x, 23 orientation_y, 24 orientation_z, 25 orientation_w,
    26 sensor_id, 27 vx, 28 vy, 29 vz, 30 vx0, 31 vy0, 32 vz0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import colorsys
import math

import numpy as np


ArrayLike = np.ndarray
FieldKey = Union[str, int]


@dataclass(frozen=True)
class FieldSpec:
    label: str
    columns: Union[int, Tuple[int, ...]]
    kind: str = (
        "scalar"  # scalar | log | log_abs | discrete | norm | signed | signed_norm
    )
    vmin: Optional[float] = None
    vmax: Optional[float] = None


AEVA_FIELDS: Dict[str, FieldSpec] = {
    "ID": FieldSpec("ID", 7, "discrete"),
    "Velocity": FieldSpec("Velocity", 4, "signed", -30.0, 30.0),
    "Intensity": FieldSpec("Intensity", 3, "scalar"),
    "Reflectivity": FieldSpec("Reflectivity", 5, "scalar"),
    "Time Offset": FieldSpec("Time Offset", 6, "scalar"),
    "Vx": FieldSpec("Vx", 8, "signed"),
    "Vy": FieldSpec("Vy", 9, "signed"),
    "Vz": FieldSpec("Vz", 10, "signed"),
}

OUSTER_FIELDS: Dict[str, FieldSpec] = {
    "Ring": FieldSpec("Ring", 6, "discrete"),
    "Intensity": FieldSpec("Intensity", 3, "scalar"),
    "Reflectivity": FieldSpec("Reflectivity", 5, "scalar"),
    "Relative Time": FieldSpec("Relative Time", 4, "scalar"),
}

RADAR_FIELDS: Dict[str, FieldSpec] = {
    "ID": FieldSpec("ID", 26, "discrete"),
    "RangeRate": FieldSpec("RangeRate", 3, "signed"),
    "Velocity": FieldSpec("Velocity", (27, 28, 29), "norm"),
    "RCS": FieldSpec("RCS", 4, "log_abs"),
    "Amplitude": FieldSpec("Amplitude", 5, "log"),
    "Range": FieldSpec("Range", 6, "scalar"),
    "Azimuth": FieldSpec("Azimuth", 7, "signed"),
    "Elevation": FieldSpec("Elevation", 8, "signed"),
    "Quality": FieldSpec("Quality", 11, "scalar"),
    "Vx": FieldSpec("Vx", 27, "signed"),
    "Vy": FieldSpec("Vy", 28, "signed"),
    "Vz": FieldSpec("Vz", 29, "signed"),
}

LEGACY_RADAR_INDEX_TO_FIELD = {
    0: "ID",
    1: "Velocity",
    2: "RCS",
    3: "Amplitude",
}
LEGACY_AEVA_INDEX_TO_FIELD = {
    0: "ID",
    1: "Velocity",
    2: "Reflectivity",
    3: "Intensity",
}
LEGACY_OUSTER_INDEX_TO_FIELD = {
    0: "Ring",
    1: "Intensity",
    2: "Reflectivity",
    3: "Relative Time",
}


def _empty_colors(n: int) -> np.ndarray:
    return np.zeros((int(n), 3), dtype=np.float64)


def _safe_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values.reshape(-1)
    good = np.isfinite(values)
    if not np.any(good):
        return values
    values = values.copy()
    median = float(np.nanmedian(values[good]))
    values[~good] = median
    return values


def _normalize(
    values: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None
) -> np.ndarray:
    values = _safe_values(values)
    if values.size == 0:
        return values

    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(values, dtype=np.float64)

    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)


def _scalar_to_gray(
    values: np.ndarray, *, vmin: Optional[float] = None, vmax: Optional[float] = None
) -> np.ndarray:
    normalized = _normalize(values, vmin=vmin, vmax=vmax)
    return np.repeat(normalized[:, None], 3, axis=1)


def _signed_to_red_blue(
    values: np.ndarray, *, vmin: Optional[float] = None, vmax: Optional[float] = None
) -> np.ndarray:
    values = _safe_values(values)
    if values.size == 0:
        return _empty_colors(0)

    if vmin is None or vmax is None:
        max_abs = float(np.nanmax(np.abs(values))) if values.size else 0.0
        if max_abs <= 0 or not np.isfinite(max_abs):
            return _empty_colors(values.size)
        vmin = -max_abs
        vmax = max_abs

    colors = np.zeros((values.size, 3), dtype=np.float64)

    positive = values >= 0
    if np.any(positive):
        denom = max(float(vmax), 1e-12)
        p = np.clip(values[positive] / denom, 0.0, 1.0)
        colors[positive, 2] = p  # positive -> blue

    negative = values < 0
    if np.any(negative):
        denom = max(abs(float(vmin)), 1e-12)
        n = np.clip(np.abs(values[negative]) / denom, 0.0, 1.0)
        colors[negative, 0] = n  # negative -> red

    return colors


def _stable_color_for_value(value: object) -> List[float]:
    try:
        f = float(value)
        if not math.isfinite(f):
            return [0.0, 0.0, 0.0]
        seed = int(f)
    except Exception:
        seed = abs(hash(str(value)))

    # Golden-ratio hue spacing gives stable, reasonably separated colors.
    hue = (seed * 0.6180339887498949) % 1.0
    sat = 0.78
    val = 1.0
    return list(colorsys.hsv_to_rgb(hue, sat, val))


def get_colors_for_ids(id_vector: Iterable[object], ptype: str = "auto") -> np.ndarray:
    values = np.asarray(list(id_vector)).reshape(-1)
    colors = [_stable_color_for_value(v) for v in values]
    return np.asarray(colors, dtype=np.float64)


def infer_pointcloud_type(
    sensor_name: Optional[str] = None, pc_data: Optional[np.ndarray] = None
) -> str:
    s = (sensor_name or "").lower()
    if "radar" in s:
        return "radar"
    if "ouster" in s:
        return "ouster"
    if "aeva" in s or "lidar" in s:
        if pc_data is not None and pc_data.ndim == 2 and pc_data.shape[1] == 7:
            return "ouster"
        return "aeva"

    if pc_data is not None and pc_data.ndim == 2:
        cols = pc_data.shape[1]
        if cols == 33:
            return "radar"
        if cols == 11:
            return "aeva"
        if cols == 7:
            return "ouster"

    return "unknown"


def get_field_specs_for_sensor(
    sensor_name: Optional[str] = None, pc_data: Optional[np.ndarray] = None
) -> Dict[str, FieldSpec]:
    ptype = infer_pointcloud_type(sensor_name=sensor_name, pc_data=pc_data)
    if ptype == "radar":
        return RADAR_FIELDS
    if ptype == "ouster":
        return OUSTER_FIELDS
    if ptype == "aeva":
        return AEVA_FIELDS
    return {"Distance": FieldSpec("Distance", (0, 1, 2), "norm")}


def get_color_fields_for_sensor(
    sensor_name: Optional[str] = None, pc_data: Optional[np.ndarray] = None
) -> List[str]:
    return list(
        get_field_specs_for_sensor(sensor_name=sensor_name, pc_data=pc_data).keys()
    )


def _legacy_index_to_field(ptype: str, column_index: int) -> str:
    if ptype == "radar":
        return LEGACY_RADAR_INDEX_TO_FIELD.get(column_index, "ID")
    if ptype == "ouster":
        return LEGACY_OUSTER_INDEX_TO_FIELD.get(column_index, "Ring")
    if ptype == "aeva":
        return LEGACY_AEVA_INDEX_TO_FIELD.get(column_index, "ID")
    return "Distance"


def _resolve_field_name(field: FieldKey, ptype: str) -> str:
    if isinstance(field, str):
        return field
    try:
        return _legacy_index_to_field(ptype, int(field))
    except Exception:
        return "ID"


def _extract_field_values(pc_data: np.ndarray, spec: FieldSpec) -> Optional[np.ndarray]:
    if pc_data.ndim != 2 or pc_data.shape[0] == 0:
        return np.asarray([], dtype=np.float64)

    cols = spec.columns
    if isinstance(cols, int):
        if cols >= pc_data.shape[1]:
            return None
        return pc_data[:, cols]

    if any(c >= pc_data.shape[1] for c in cols):
        return None

    sub = pc_data[:, list(cols)]
    if spec.kind in ("norm", "signed_norm"):
        return np.linalg.norm(sub, axis=1)
    return sub


def colorize_pointcloud(
    pc_data: np.ndarray,
    field_name: FieldKey = "ID",
    *,
    sensor_name: Optional[str] = None,
    default_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:

    pc_data = np.asarray(pc_data)
    n = int(pc_data.shape[0]) if pc_data.ndim >= 1 else 0
    if n == 0:
        return _empty_colors(0)

    ptype = infer_pointcloud_type(sensor_name=sensor_name, pc_data=pc_data)
    specs = get_field_specs_for_sensor(sensor_name=sensor_name, pc_data=pc_data)
    resolved = _resolve_field_name(field_name, ptype)

    if resolved not in specs:
        # Fall back gracefully to the first legal field for this sensor.
        resolved = next(iter(specs.keys()))

    spec = specs[resolved]
    values = _extract_field_values(pc_data, spec)
    if values is None:
        return np.tile(np.asarray(default_color, dtype=np.float64), (n, 1))

    if spec.kind == "discrete":
        return get_colors_for_ids(values, ptype=ptype)
    if spec.kind == "log":
        return _scalar_to_gray(
            np.log(np.maximum(_safe_values(values), 0.0) + 1.0),
            vmin=spec.vmin,
            vmax=spec.vmax,
        )
    if spec.kind == "log_abs":
        return _scalar_to_gray(
            np.log(np.abs(_safe_values(values)) + 1.0), vmin=spec.vmin, vmax=spec.vmax
        )
    if spec.kind in ("signed", "signed_norm"):
        return _signed_to_red_blue(values, vmin=spec.vmin, vmax=spec.vmax)
    return _scalar_to_gray(values, vmin=spec.vmin, vmax=spec.vmax)


def extract_column_as_colors_radar(
    pc_data: np.ndarray, column_index: FieldKey
) -> np.ndarray:
    return colorize_pointcloud(pc_data, column_index, sensor_name="radar")


def extract_column_as_colors_lidar(
    pc_data: np.ndarray, column_index: FieldKey, sensor_name: Optional[str] = None
) -> np.ndarray:
    return colorize_pointcloud(
        pc_data, column_index, sensor_name=sensor_name or "lidar"
    )


def extract_column_as_colors_camera(pc_data, column_index: FieldKey):
    return pc_data
