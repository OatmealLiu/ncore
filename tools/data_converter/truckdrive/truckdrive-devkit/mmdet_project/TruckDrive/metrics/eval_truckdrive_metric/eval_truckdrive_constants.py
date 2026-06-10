"""Defines constants used by BEV OD metrics evaluation."""

# Derived from nuScenes devkit (Oscar Beijbom and Varun Bankiti, 2019).
# Modified by Torc Robotics, 2026.

ATTRIBUTE_NAMES = []

TP_METRICS = ["trans_err", "scale_err", "orient_err", "vel_err", "attr_err"]

PRETTY_TP_METRICS = {
    "trans_err": "Trans.",
    "scale_err": "Scale",
    "orient_err": "Orient.",
    "vel_err": "Vel.",
    "attr_err": "Attr.",
}

TP_METRICS_UNITS = {
    "trans_err": "m",
    "scale_err": "1-IOU",
    "orient_err": "rad.",
    "vel_err": "m/s",
    "attr_err": "1-acc.",
}

# TP metrics for size and scale.
TP_METRICS_SIZE_PLUS_SCALE = [
    "pos_err_x",
    "pos_err_y",
    "pos_err_z",
    "len_err",
    "wid_err",
    "ht_err",
]

TP_METRICS_EXTENTS = ["long_ext_acc", "lat_ext_acc"]

PRETTY_TP_METRICS_SIZE_PLUS_SCALE = {
    "pos_err_x": "Pos Err X",
    "pos_err_y": "Pos Err Y",
    "pos_err_z": "Pos Err Z",
    "len_err": "Length Err",
    "wid_err": "Width Err",
    "ht_err": "Height Err",
}

TP_METRICS_UNITS_S = {
    "pos_err_x": "m",
    "pos_err_y": "m",
    "pos_err_z": "m",
    "len_err": "m",
    "wid_err": "m",
    "ht_err": "m",
}

VALID_VELOCITY_LIMIT = 50.0
