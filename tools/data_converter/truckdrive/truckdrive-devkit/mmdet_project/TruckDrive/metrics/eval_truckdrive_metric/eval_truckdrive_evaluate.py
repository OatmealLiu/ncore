"""BEV OD Metrics evaluation."""

# Derived from nuScenes devkit (Oscar Beijbom, 2019).
# Modified by Torc Robotics, 2026.

import multiprocessing
import time
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion

from .eval_truckdrive_constants import (
    TP_METRICS,
    TP_METRICS_EXTENTS,
    TP_METRICS_SIZE_PLUS_SCALE,
)
from .eval_truckdrive_dataclasses import (
    DetectionBox,
    DetectionConfig,
    DetectionMetricData,
    DetectionMetricDataList,
    DetectionMetrics,
    EvalBox,
    EvalBoxes,
    center_distance,
    iou_2d_inverse,
    quaternion_yaw,
)


def get_box_corners(
    translation: Tuple[float, float, float],
    size: Tuple[float, float, float],
    rotation: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Get the 8 corners of the 3D bounding box.

    Parameters
    ----------
    translation:
        Center of the box in bbox frame (x, y, z).
    size:
        Size of the box (width, length, height).
    rotation:
        Box's orientation given as a quaternion (x, y, z, w).

    Returns
    -------
        An array of shape (8, 3) containing the 8 corner points of the box in the target frame.
    """
    # Create a quaternion object from the rotation tuple
    q = Quaternion(rotation)

    # Define the local corner points of the box
    length, width, height = size
    x_corners = length / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = height / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    local_corners = np.vstack((x_corners, y_corners, z_corners)).T

    # Rotate the corners by the quaternion
    rotated_corners = np.dot(q.rotation_matrix, local_corners.T).T

    # Translate the corners to the target frame
    target_box_corners = rotated_corners + np.array(translation)

    return target_box_corners


def longitudinal_extent_accuracy(
    gt_box: EvalBox,
    pred_box: EvalBox,
    ego_front_dist: Optional[float] = None,
    ego_rear_dist: Optional[float] = None,
) -> float:
    """
    Compute the longitudinal extent accuracy for non-axis aligned boxes.

    Note: The x axis of the frame of the bbox should aligned with the vehicle coordinate frame,
    As we are using it for identifying the ego front and ego rear distance

    Parameters
    ----------
    gt_box:
        Ground truth box with attributes sample_token, translation, size, rotation.
    pred_box:
        Predicted box with attributes sample_token, translation, size, rotation.
    ego_front_dist:
        Distance from the ego to the front of the truck.
    ego_rear_dist:
        Distance from the ego to the rear of the truck.

    Returns
    -------
    float
        longitudinal extent accuracy
    """
    # Get the corners of both the ground truth and predicted boxes
    gt_corners = get_box_corners(gt_box.translation, gt_box.size, gt_box.rotation)
    pred_corners = get_box_corners(
        pred_box.translation, pred_box.size, pred_box.rotation
    )

    # Calculate the longitudinal extents (front and rear)
    front_measurement = np.abs(np.min(gt_corners[:, 0]) - np.min(pred_corners[:, 0]))
    rear_measurement = np.abs(np.max(gt_corners[:, 0]) - np.max(pred_corners[:, 0]))

    # The side measurement is the maximum of the front and rear measurements
    side_measurement = max(front_measurement, rear_measurement)

    if min(np.min(gt_corners[:, 0]), np.min(pred_corners[:, 0])) > ego_front_dist:
        return front_measurement
    elif max(np.max(gt_corners[:, 0]), np.max(pred_corners[:, 0])) < ego_rear_dist:
        return rear_measurement
    else:
        return side_measurement


def lateral_extent_accuracy(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    Compute the lateral extent accuracy for a pair of ground truth and predicted bounding boxes.

    Note: The x axis of the frame of the bbox should aligned with the vehicle coordinate frame,
    As we are using it for identifying the ego front and ego rear distance

    Parameters
    ----------
    gt_box:
        Ground truth box with attributes sample_token, translation, size, rotation.
    pred_box:
        Predicted box with attributes sample_token, translation, size, rotation.

    Returns
    -------
    float
        lateral extent accuracy
    """
    # Get the corners of both the ground truth and predicted boxes
    gt_corners = get_box_corners(gt_box.translation, gt_box.size, gt_box.rotation)
    pred_corners = get_box_corners(
        pred_box.translation, pred_box.size, pred_box.rotation
    )

    # Identify the leftmost and rightmost corners on the Y-axis for the ground truth
    gt_leftmost = np.min(gt_corners[:, 1])
    gt_rightmost = np.max(gt_corners[:, 1])

    # Identify the leftmost and rightmost corners on the Y-axis for the prediction
    pred_leftmost = np.min(pred_corners[:, 1])
    pred_rightmost = np.max(pred_corners[:, 1])

    # The lateral extent error is the maximum difference between the ground truth's
    # and prediction's leftmost or rightmost extents
    left_error = np.abs(gt_leftmost - pred_leftmost)
    right_error = np.abs(gt_rightmost - pred_rightmost)

    lateral_error = max(left_error, right_error)

    return lateral_error


def pos_err_x_l1(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    Absolute distance between the bounding box positions in x (forward) direction.

    Parameters
    ----------
    gt_box:
        GT annotation sample.
    pred_box:
        Predicted sample.

    Returns
    -------
    float
        absolute distance
    """
    return abs(gt_box.translation[0] - pred_box.translation[0])


def pos_err_y_l1(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    Absolute distance between the bounding box positions in y (lateral) direction.

    Parameters
    ----------
    gt_box:
        GT annotation sample.
    pred_box:
        Predicted sample.

    Returns
    -------
    float
        absolute distance.

    """
    return abs(gt_box.translation[1] - pred_box.translation[1])


def pos_err_z_l1(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    Absolute distance between the bounding box positions in z (height) direction.

    Parameters
    ----------
    gt_box:
        GT annotation sample.
    pred_box:
        Predicted sample.

    Returns
    -------
    float
        absolute distance.
    """
    return abs(gt_box.translation[2] - pred_box.translation[2])


def length_err_l1(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    Absolute difference between the bounding box sizes in x (forward) direction.

    Parameters
    ----------
    gt_box:
        GT annotation sample.
    pred_box:
        Predicted sample.

    Returns
    -------
    float
        absolute difference.
    """
    return abs(gt_box.size[0] - pred_box.size[0])


def width_err_l1(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    Absolute difference between the bounding box sizes in y (lateral) direction.

    Parameters
    ----------
    gt_box:
        GT annotation sample.
    pred_box:
        Predicted sample.

    Returns
    -------
    float
        absolute difference.
    """
    return abs(gt_box.size[1] - pred_box.size[1])


def height_err_l1(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    Absolute difference between the bounding box sizes in z (height) direction.

    Parameters
    ----------
    gt_box:
        GT annotation sample.
    pred_box:
        Predicted sample.

    Returns
    -------
    float
        absolute difference.
    """
    return abs(gt_box.size[2] - pred_box.size[2])


def velocity_l2(gt_box: EvalBox, pred_box: EvalBox) -> np.floating:
    """
    L2 distance between the velocity vectors (xy only).

    If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.

    Parameters
    ----------
    gt_box:
        GT annotation sample.
    pred_box:
        Predicted sample.

    Returns
    -------
    float
        L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.velocity) - np.array(gt_box.velocity))


def yaw_diff(gt_box: EvalBox, eval_box: EvalBox, period: float = 2 * np.pi) -> float:
    """
    Return the yaw angle difference between the orientation of two boxes.

    Parameters
    ----------
    gt_box:
        Ground truth box.
    eval_box:
        Predicted box.
    period:
        Periodicity in radians for assessing angle difference.

    Returns
    -------
    float
        Yaw angle difference in radians in [0, pi].
    """
    yaw_gt = quaternion_yaw(Quaternion(gt_box.rotation))
    yaw_est = quaternion_yaw(Quaternion(eval_box.rotation))

    return abs(angle_diff(yaw_gt, yaw_est, period))


def angle_diff(x: float, y: float, period: float) -> float:
    """
    Return the smallest angle difference between 2 angles: the angle from y to x.

    Parameters
    ----------
    x:
        To angle.
    y:
        From angle.
    period:
        Periodicity in radians for assessing angle difference.

    Returns
    -------
    float
        Signed smallest between-angle difference in range (-pi, pi).
    """
    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


def attr_acc(gt_box: DetectionBox, pred_box: DetectionBox) -> float:
    """
    Compute the classification accuracy for the attribute of this class (if any).

    If the GT class has no attributes or the annotation is missing attributes, we assign an accuracy of nan, which is
    ignored later on.

    Parameters
    ----------
    gt_box:
        GT annotation sample.
    pred_box:
        Predicted sample.

    Returns
    -------
    float
        Attribute classification accuracy (0 or 1) or nan if GT annotation does not have any attributes.
    """
    acc = (
        np.nan
        if gt_box.attribute_name == ""
        else float(gt_box.attribute_name == pred_box.attribute_name)
    )
    return acc


def scale_iou(sample_annotation: EvalBox, sample_result: EvalBox) -> np.float32:
    """
    Compare predictions to the ground truth in terms of scale.

    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.

    Parameters
    ----------
    sample_annotation:
        GT annotation sample.
    sample_result:
        Predicted sample.

    Returns
    -------
    float
        Scale IOU.
    """
    # Validate inputs.
    sa_size = np.array(sample_annotation.size)
    sr_size = np.array(sample_result.size)
    # assert all(sa_size > 0), "Error: sample_annotation sizes must be >0."
    # assert all(sr_size > 0), "Error: sample_result sizes must be >0."
    if all(sa_size > 0):
        "Error: sample_annotation sizes must be >0."
        sa_size = np.abs(sa_size)
    if all(sr_size > 0):
        "Error: sample_result sizes must be >0."
        sr_size = np.abs(sr_size)

    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size, dtype=np.float32)
    volume_annotation = np.prod(sa_size, dtype=np.float32)
    volume_result = np.prod(sr_size, dtype=np.float32)
    intersection = np.prod(min_wlh, dtype=np.float32)
    union = volume_annotation + volume_result - intersection
    iou = intersection / union

    return iou


def cummean(x: npt.NDArray) -> npt.NDArray:
    """
    Compute the cumulative mean up to each position in a NaN sensitive way.

    - If all values are NaN return an array of ones.
    - If some values are NaN, accumulate arrays discording those entries.

    """
    if sum(np.isnan(x)) == len(x):
        # Is all numbers in array are NaN's.
        return np.ones(
            len(x)
        )  # If all errors are NaN set to error to 1 for all operating points.
    else:
        # Accumulate in a nan-aware manner.
        sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
        count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
        return np.divide(
            sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0
        )


def filter_boxes_by_range(boxes, filter_range):
    """
    Filter bounding boxes by a given range.

    Parameters
    ----------
    boxes:
        list of ground truth  or predicted bounding boxes.
    filter_range:
        A range with a lower limit and upper limit.

    Returns
    -------
    List
        list of boxes that are in the given range.
    """
    boxes_in_range = []
    for box in boxes:
        box_distance = np.linalg.norm(np.array((0, 0)) - np.array(box.translation[:2]))
        if (
            box_distance >= filter_range[0]
            and box_distance < filter_range[1]
            and not np.isin(np.array(box.size), [0, np.inf]).any()
        ):
            boxes_in_range.append(box)

    return boxes_in_range


def filter_boxes_by_roi(eval_boxes, filter_roi):
    """
    Keep bounding boxes in a given Region of Interest (ROI).

    Parameters
    ----------
    boxes:
        list of ground truth  or predicted bounding boxes.
    filter_range:
        A range with a lower limit and upper limit.

    Returns
    -------
    List
        list of boxes that are in the given range.
    """
    for sample_token in eval_boxes.boxes:
        boxes_in_range = []
        for box in eval_boxes.boxes[sample_token]:
            within_x = (
                float(box.translation[0]) >= filter_roi["xrange"][0]
                and float(box.translation[0]) <= filter_roi["xrange"][-1]
            )
            within_y = (
                float(box.translation[1]) >= filter_roi["yrange"][0]
                and float(box.translation[1]) <= filter_roi["yrange"][-1]
            )
            within_z = (
                float(box.translation[2]) >= filter_roi["zrange"][0]
                and float(box.translation[2]) <= filter_roi["zrange"][-1]
            )

            if within_x and within_y and within_z:
                boxes_in_range.append(box)

        eval_boxes.boxes[sample_token] = boxes_in_range
    return eval_boxes


def filter_boxes_by_class(boxes, class_name):
    """
    Filter bounding boxes by class.

    Parameters
    ----------
    boxes:
        list of ground truth  or predicted bounding boxes.
    class_name:
        class_name of the bounding box we are interested in.

    Returns
    -------
    List
        list of boxes that matches the given class_name.
    """
    boxes_in_range = []
    for box in boxes:
        if box.detection_name == class_name:
            boxes_in_range.append(box)

    return boxes_in_range


def accumulate(  # noqa: C901
    gt_boxes: EvalBoxes,
    pred_boxes: EvalBoxes,
    class_name: str,
    dist_fcn: Callable,
    dist_th: float,
    range_bin: list,
    verbose: bool = False,
    gt_difficult: Union[list, None] = None,
    ego_front_dist: Optional[float] = None,
    ego_rear_dist: Optional[float] = None,
) -> Tuple[str, float, List, DetectionMetricData]:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.

    The recall/conf thresholds and other raw metrics will be used in secondary metrics.

    Parameters
    ----------
    gt_boxes:
        Maps every sample_token to a list of its sample_annotations.
    pred_boxes:
        Maps every sample_token to a list of its sample_results.
    class_name:
        Class to compute AP on, None if running in class agnostic mode (weighted mAP).
    dist_fcn:
        Distance function used to match detections and ground truths.
    dist_th:
        Distance threshold for a match.
    verbose:
        If true, print debug messages.
    ego_front_dist:
        Distance from the ego to the front of the truck.
    ego_rear_dist:
        Distance from the ego to the rear of the truck.

    Returns
    -------
    DetectionMetricData
        (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------
    class_agnostic_eval = class_name == "class_agnostic"
    range_str = str(int(range_bin[0])) + "-" + str(int(range_bin[1]))
    # Count the positives.
    if gt_difficult is None:
        if not class_agnostic_eval:
            range_boxes = filter_boxes_by_range(gt_boxes.all, range_bin)
            class_boxes = filter_boxes_by_class(range_boxes, class_name)
            # npositive = len(filter_boxes(, class_name, range_bin))
            npositive = len(class_boxes)
        else:
            npositive = len(gt_boxes.all)

    else:
        if not class_agnostic_eval:
            npositive = len(
                [1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name]
            ) - len(
                [
                    1
                    for idx, gt_box in enumerate(gt_boxes.all)
                    if gt_box.detection_name == class_name and gt_difficult[idx]
                ]
            )
        else:
            npositive = len(gt_boxes.all) - len(
                [1 for idx, _ in enumerate(gt_boxes.all) if gt_difficult[idx]]
            )

    if verbose:
        print(
            "Found {} GT of class {} out of {} total across {} samples.".format(
                npositive, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)
            )
        )

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npositive == 0:
        return (class_name, dist_th, range_bin, DetectionMetricData.no_predictions())

    # pred_boxes_list = filter_boxes(pred_boxes.all, class_name, range_bin)
    # Organize the predictions in a single list.
    pred_range_boxes = filter_boxes_by_range(pred_boxes.all, range_bin)
    if not class_agnostic_eval:
        pred_boxes_list = filter_boxes_by_class(pred_range_boxes, class_name)
    else:
        pred_boxes_list = pred_range_boxes

    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print(
            "Found {} PRED of class {} out of {} total across {} samples.".format(
                len(pred_confs),
                class_name,
                len(pred_boxes.all),
                len(pred_boxes.sample_tokens),
            )
        )

    # Sort by confidence, high to low
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {
        "trans_err": [],
        "vel_err": [],
        "scale_err": [],
        "iou_2d": [],
        "orient_err": [],
        "attr_err": [],
        "conf": [],
        "pos_err_x": [],
        "pos_err_y": [],
        "pos_err_z": [],
        "len_err": [],
        "wid_err": [],
        "ht_err": [],
        "long_ext_acc": [],
        "lat_ext_acc": [],
    }

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        # Iterate through all the ground truth bounding boxes
        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):
            # Find the closest match among ground truth boxes if the box is of the same class and not already associated
            if not class_agnostic_eval:
                if (
                    gt_box.detection_name == class_name
                    and (pred_box.sample_token, gt_idx) not in taken
                ):
                    this_distance = dist_fcn(gt_box, pred_box)
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = int(gt_idx)
            else:
                if (pred_box.sample_token, gt_idx) not in taken:
                    this_distance = dist_fcn(gt_box, pred_box)
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = int(gt_idx)

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            #  Update tp, fp and confs.
            # if match with difficult GT bbox - (not counted as TP)
            if gt_difficult is not None:
                if gt_difficult[match_gt_idx]:  # type: ignore
                    tp.append(0)
                else:
                    taken.add((pred_box.sample_token, match_gt_idx))
                    tp.append(1)
            else:
                taken.add((pred_box.sample_token, match_gt_idx))
                tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]  # type: ignore

            match_data["trans_err"].append(center_distance(gt_box_match, pred_box))
            match_data["pos_err_x"].append(pos_err_x_l1(gt_box_match, pred_box))
            match_data["pos_err_y"].append(pos_err_y_l1(gt_box_match, pred_box))
            match_data["pos_err_z"].append(pos_err_z_l1(gt_box_match, pred_box))
            match_data["len_err"].append(length_err_l1(gt_box_match, pred_box))
            match_data["wid_err"].append(width_err_l1(gt_box_match, pred_box))
            match_data["ht_err"].append(height_err_l1(gt_box_match, pred_box))
            match_data["vel_err"].append(velocity_l2(gt_box_match, pred_box))
            match_data["scale_err"].append(1 - scale_iou(gt_box_match, pred_box))
            match_data["iou_2d"].append(1.0 - iou_2d_inverse(gt_box_match, pred_box))
            match_data["long_ext_acc"].append(
                longitudinal_extent_accuracy(
                    gt_box_match, pred_box, ego_front_dist, ego_rear_dist
                )
            )
            match_data["lat_ext_acc"].append(
                lateral_extent_accuracy(gt_box_match, pred_box)
            )

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == "barrier" else 2 * np.pi
            match_data["orient_err"].append(
                yaw_diff(gt_box_match, pred_box, period=period)
            )
            match_data["attr_err"].append(1 - attr_acc(gt_box_match, pred_box))
            match_data["conf"].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data["trans_err"]) == 0:
        return (class_name, dist_th, range_bin, DetectionMetricData.no_predictions())

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp_cum = np.cumsum(tp).astype(float)  # prefix sum, with the same length
    fp_cum = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # # Recall at defined confidence thresholds
    # for confidence_threshold in np.linspace(0.0, 1.0, 11):
    #     thresholded_idxs = np.argwhere(conf >=  confidence_threshold)[:, 0]
    #     if thresholded_idxs.shape[0] == 0:
    #         sum_tp, sum_fp = 0, 0
    #         prec_thresh, rec_thresh, f1_thresh = 0.0, 0.0, 0.0
    #     else:
    #         sum_tp = np.sum(np.array(tp)[thresholded_idxs])
    #         sum_fp = np.sum(np.array(fp)[thresholded_idxs])
    #         sum_fn = npositive - sum_tp
    #         prec_thresh = sum_tp / float(sum_fp + sum_tp)
    #         rec_thresh = sum_tp / float(sum_fn + sum_tp)
    #         f1_thresh = 2 * sum_tp / (2 * sum_tp + sum_fp + sum_fn)
    #     print(
    #         "For class {}, with confidence threshold of {} - Precision: {}, Recall: {}, F1: {} (TP: {}, FP: {}, FN: {}).".
    #         format(class_name, round(confidence_threshold, 2), round(prec_thresh, 2), round(rec_thresh, 2), round(f1_thresh, 2), sum_tp, sum_fp, sum_fn)
    #     )

    # Calculate precision and recall.
    # the values at index k means the prec and recall up to the first k samples
    prec = tp_cum / (fp_cum + tp_cum)
    rec = tp_cum / float(npositive)

    if verbose:
        print(
            "For class {}, in range {}m-{}m, with threshold of {}m the precision is {} and the recall is {}.".format(
                class_name, range_bin[0], range_bin[1], dist_th, prec[-1], rec[-1]
            )
        )

    # linear interpolation of the prec vs rec at 101 points
    rec_interp = np.linspace(
        0, 1, DetectionMetricData.nelem
    )  # 101 steps, from 0% to 100% recall.
    # prec = np.maximum.accumulate(prec[::-1])[::-1]
    prec_interp = np.interp(rec_interp, rec, prec, right=0)
    conf_interp = np.interp(rec_interp, rec, conf, right=0)

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data:
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            cur_metric = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            # Based on the confidence vs. metric, interpolate at these 101 recall values
            match_data[key] = np.interp(conf_interp[::-1], match_data["conf"][::-1], cur_metric[::-1])[::-1]  # type: ignore

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return (
        class_name,
        dist_th,
        range_bin,
        DetectionMetricData(
            recall=rec_interp,
            precision=prec_interp,
            confidence=conf_interp,
            trans_err=np.array(match_data["trans_err"]),
            vel_err=np.array(match_data["vel_err"]),
            scale_err=np.array(match_data["scale_err"]),
            iou_2d=np.array(match_data["iou_2d"]),
            orient_err=np.array(match_data["orient_err"]),
            attr_err=np.array(match_data["attr_err"]),
            pos_err_x=np.array(match_data["pos_err_x"]),
            pos_err_y=np.array(match_data["pos_err_y"]),
            pos_err_z=np.array(match_data["pos_err_z"]),
            len_err=np.array(match_data["len_err"]),
            wid_err=np.array(match_data["wid_err"]),
            ht_err=np.array(match_data["ht_err"]),
            long_ext_acc=np.array(match_data["long_ext_acc"]),
            lat_ext_acc=np.array(match_data["lat_ext_acc"]),
            gt_filtered=npositive,
            gt_unfiltered=len(gt_boxes.all),
            pred_filtered=len(pred_boxes_list),
            pred_unfiltered=len(pred_boxes.all),
            samples=len(gt_boxes.sample_tokens),
            range_str=range_str,
        ),
    )


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """Calculate average precision."""
    # Case where there is no ground truth. Will stay commented until we have alignment
    # within team.
    # if md.zero_division:
    #     return float('NaN')

    assert 0 <= min_precision < 1  # 0.1
    assert 0 <= min_recall <= 1  # 0.1

    prec = np.copy(md.precision)
    prec = prec[
        round(100 * min_recall) + 1 :
    ]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0

    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """Calculate true positive errors."""
    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = (
        md.max_recall_ind
    )  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind : last_ind + 1]))


def calc_std(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """Calculate standard deviation for true positive errors."""
    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = (
        md.max_recall_ind
    )  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.std(getattr(md, metric_name)[first_ind : last_ind + 1]))


def evaluate(
    gt_boxes: EvalBoxes,
    pred_boxes: EvalBoxes,
    cfg: DetectionConfig,
    verbose: bool = False,
    gt_difficult: Union[list, None] = None,
) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
    """
    Evaluate BEV OD metrics.

    Returns
    -------
    Tuple
        high-level and the raw metric data.
    """
    start_time = time.time()
    # return {}, None

    # -----------------------------------
    # Step 1: Accumulate metric data for all classes and distance thresholds.
    # -----------------------------------
    if verbose:
        print("Accumulating metric data...")
    metric_data_list = DetectionMetricDataList()

    if cfg.roi_eval:
        gt_boxes = filter_boxes_by_roi(gt_boxes, cfg.roi)
        pred_boxes = filter_boxes_by_roi(pred_boxes, cfg.roi)

    accumulate_params = []
    for class_name in [
        *list(cfg.class_names),
        "class_agnostic",
    ]:  # keys: [car, truck, bus, trailer, etc. ] TODO make class_agnostic eval optional in the config file
        for dist_th in cfg.dist_ths:  # [0.5, 1.0, 2.0, 4.0]
            for range_bin in cfg.range_bin_list:
                if cfg.dist_fcn == "iou_2d":
                    current_params = [
                        gt_boxes,
                        pred_boxes,
                        class_name,
                        cfg.dist_fcn_callable,
                        1 - dist_th,
                        range_bin,
                        verbose,
                        gt_difficult,
                    ]
                else:
                    current_params = [
                        gt_boxes,
                        pred_boxes,
                        class_name,
                        cfg.dist_fcn_callable,
                        dist_th,
                        range_bin,
                        verbose,
                        gt_difficult,
                    ]
                accumulate_params.append(current_params)

    n = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n)
    processes = []
    for i in range(len(accumulate_params)):
        gt_boxes = accumulate_params[i][0]
        pred_boxes = accumulate_params[i][1]
        class_name = accumulate_params[i][2]
        dist_fcn_callable = accumulate_params[i][3]
        dist_th = accumulate_params[i][4]
        range_bin = accumulate_params[i][5]
        verbose = accumulate_params[i][6]
        gt_difficult = accumulate_params[i][7]
        processes.append(
            pool.apply_async(
                accumulate,
                args=(
                    gt_boxes,
                    pred_boxes,
                    class_name,
                    dist_fcn_callable,
                    dist_th,
                    range_bin,
                    verbose,
                    gt_difficult,
                    cfg.ego_front_dist,
                    cfg.ego_rear_dist,
                ),
            )
        )
        # processes.append(accumulate(
        #     gt_boxes,
        #     pred_boxes,
        #     class_name,
        #     dist_fcn_callable,
        #     dist_th,
        #     range_bin,
        #     verbose,
        #     gt_difficult,
        #     cfg.ego_front_dist,
        #     cfg.ego_rear_dist,
        #     )
        # )
    print("done accumulating")
    results = [p.get() for p in processes]
    # results = [p for p in processes]
    for result in results:
        metric_data_list.set(
            detection_name=result[0],
            match_distance=result[1],
            bins=result[2],
            data=result[3],
        )

    print("Calculating metrics...")
    # -----------------------------------
    # Step 2: Calculate metrics from the data.
    # -----------------------------------
    if verbose:
        print("Calculating metrics...")
    metrics = DetectionMetrics(cfg)

    for class_name in cfg.class_names:
        # Compute APs.
        for dist_th in cfg.dist_ths:
            for range_bin in cfg.range_bin_list:
                range_str = str(int(range_bin[0])) + "-" + str(int(range_bin[1]))
                metric_data = metric_data_list[(class_name, dist_th, range_bin)]
                ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)

                metrics.add_label_ap_binned(class_name, dist_th, ap, range_str)
                metrics._label_aps_binned[range_str][class_name][
                    "gt_filtered"
                ] = metric_data.gt_filtered
        # Compute TP metrics.
        for metric_name in TP_METRICS + TP_METRICS_SIZE_PLUS_SCALE + TP_METRICS_EXTENTS:
            for range_bin in cfg.range_bin_list:
                range_str = str(int(range_bin[0])) + "-" + str(int(range_bin[1]))
                metric_data = metric_data_list[(class_name, cfg.dist_th_tp, range_bin)]
                tp = calc_tp(metric_data, cfg.min_recall, metric_name)
                std = calc_std(metric_data, cfg.min_recall, metric_name)
                metrics.add_label_tp_binned(class_name, metric_name, tp, range_str)
                metrics.add_label_tp_std_binned(class_name, metric_name, std, range_str)

    # # Add  class agnostic evaluation
    # class_agnostic_metrics = DetectionMetrics(cfg)
    # # Compute weighted mean AP
    # for dist_th in cfg.dist_ths:
    #     metric_data = metric_data_list[("class_agnostic", dist_th)]
    #     ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
    #     class_agnostic_metrics.add_label_ap("class_agnostic", dist_th, ap)

    # Compute evaluation time.
    overall_range_str = "-".join(
        [str(int(x)) for x in cfg.range_bin_list[cfg.overall_range_id]]
    )
    metrics.add_runtime(time.time() - start_time)
    metrics._label_aps = metrics._label_aps_binned[overall_range_str]
    metrics._label_tp_errors = metrics._label_tp_errors_binned[overall_range_str]
    return metrics, metric_data_list
