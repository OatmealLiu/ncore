"""Implements data classes to evaluate metrics on Bounding Box detections."""

# Derived from nuScenes devkit (Holger Caesar & Oscar Beijbom, 2019).
# Modified by Torc Robotics, 2026.

import abc
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion
from shapely.geometry import Polygon

from .eval_truckdrive_constants import (
    ATTRIBUTE_NAMES,
    TP_METRICS,
    TP_METRICS_SIZE_PLUS_SCALE,
    VALID_VELOCITY_LIMIT,
)


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.

    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.

    Parameters
    ----------
    q:
        Quaternion of interest.

    Returns
    -------
    float
      Yaw angle in radians.
    """
    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def yaw_quaternion(y: float) -> Quaternion:
    """
    Calculate the quaternion from a yaw angle.

    Parameters
    ----------
    y: float
        yaw angle in radians

    Returns
    -------
    q: Quaternion
        Quaternion of interest
    """
    q = Quaternion(axis=(0.0, 0.0, 1.0), radians=y)

    return q


class EvalBox(abc.ABC):
    """Abstract base class for data classes used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(
        self,
        sample_token: str = "",
        translation: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        velocity: Tuple[float, float] = (0, 0),
        ego_translation: Tuple[float, float, float] = (
            0,
            0,
            0,
        ),  # Translation to ego vehicle in meters.
        num_pts: int = -1,
    ):  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
        # Assert data for shape and NaNs.
        assert isinstance(sample_token, str), "Error: sample_token must be a string!"

        assert len(translation) == 3, "Error: Translation must have 3 elements!"
        assert not np.any(np.isnan(translation)), "Error: Translation may not be NaN!"

        assert len(size) == 3, "Error: Size must have 3 elements!"
        assert not np.any(np.isnan(size)), "Error: Size may not be NaN!"

        assert len(rotation) == 4, "Error: Rotation must have 4 elements!"
        assert not np.any(np.isnan(rotation)), "Error: Rotation may not be NaN!"

        # Velocity can be NaN from our database for certain annotations.
        assert len(velocity) == 2, "Error: Velocity must have 2 elements!"

        assert len(ego_translation) == 3, "Error: Translation must have 3 elements!"
        assert not np.any(
            np.isnan(ego_translation)
        ), "Error: Translation may not be NaN!"

        assert isinstance(num_pts, int), "Error: num_pts must be int!"
        assert not np.any(np.isnan(num_pts)), "Error: num_pts may not be NaN!"

        # Assign.
        self.sample_token = sample_token
        self.translation = translation
        self.size = size
        self.rotation = rotation
        self.velocity = velocity
        self.ego_translation = ego_translation
        self.num_pts = num_pts

    @property
    def ego_dist(self) -> float:
        """Compute the distance from this box to the ego vehicle in BEV-2D."""
        return np.sqrt(np.sum(np.array(self.ego_translation[:2]) ** 2))

    def __repr__(self):
        """Abstract method."""
        return str(self.serialize())

    @abc.abstractmethod
    def serialize(self) -> dict:
        """Abstract method."""

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content: dict):
        """Abstract method."""


class DetectionBox(EvalBox):
    """Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(
        self,
        sample_token: str = "",
        translation: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        velocity: Tuple[float, float] = (0, 0),
        ego_translation: Tuple[float, float, float] = (
            0,
            0,
            0,
        ),  # Translation to ego vehicle in meters.
        num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
        detection_name: str = "car",  # The class name used in the detection challenge.
        detection_score: float = -1.0,  # GT samples do not have a score.
        attribute_name: str = "",  # Box attribute. Each box can have at most 1 attribute.
        label: int = 0,
    ):
        super().__init__(
            sample_token,
            translation,
            size,
            rotation,
            velocity,
            ego_translation,
            num_pts,
        )

        assert detection_name is not None, "Error: detection_name cannot be empty!"
        # assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == "", (
            "Error: Unknown attribute_name %s" % attribute_name
        )

        assert isinstance(
            detection_score, float
        ), "Error: detection_score must be a float!"
        assert not np.any(
            np.isnan(detection_score)
        ), "Error: detection_score may not be NaN!"

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name
        self.label = label

    def get_range_2d(self):
        """
        Get the 2d range of the box.

        Returns
        -------
            The 2d range of the box
        """
        return np.linalg.norm(np.array([self.translation[0], self.translation[1]]))

    def get_azimuth(self):
        """
        Get the azimuth angle of the box in rad.

        Returns
        -------
            The azimuth angle of the box in rad
        """
        return quaternion_yaw(Quaternion(self.rotation))

    def __eq__(self, other):
        """Check if two instances of the class are equal."""
        return (
            self.sample_token == other.sample_token
            and self.translation == other.translation
            and self.size == other.size
            and self.rotation == other.rotation
            and self.velocity == other.velocity
            and self.ego_translation == other.ego_translation
            and self.num_pts == other.num_pts
            and self.detection_name == other.detection_name
            and self.detection_score == other.detection_score
            and self.attribute_name == other.attribute_name
            and self.label == other.label
        )

    def serialize(self) -> dict:
        """Serialize instance into json-friendly format."""
        return {
            "sample_token": self.sample_token,
            "translation": self.translation,
            "size": self.size,
            "rotation": self.rotation,
            "velocity": self.velocity,
            "ego_translation": self.ego_translation,
            "num_pts": self.num_pts,
            "detection_name": self.detection_name,
            "detection_score": self.detection_score,
            "attribute_name": self.attribute_name,
            "label": self.label,
        }

    @classmethod
    def deserialize(cls, content: dict):
        """Initialize from serialized content."""
        return cls(
            sample_token=content["sample_token"],
            translation=tuple(content["translation"]),
            size=tuple(content["size"]),
            rotation=tuple(content["rotation"]),
            velocity=tuple(content["velocity"]),
            ego_translation=(
                (0.0, 0.0, 0.0)
                if "ego_translation" not in content
                else tuple(content["ego_translation"])
            ),
            num_pts=-1 if "num_pts" not in content else int(content["num_pts"]),
            detection_name=content["detection_name"],
            detection_score=(
                -1.0
                if "detection_score" not in content
                else float(content["detection_score"])
            ),
            attribute_name=content.get("attribute_name", ""),
            label=content["label"],
        )

    @classmethod
    def to_kitti_bbox(cls, detection_box):
        """Convert a detection box into a kitti bounding box of [x, y, z, l, w, h, yaw]."""
        kitti_bbox = np.empty((7,), dtype=float)
        kitti_bbox[:3] = np.array(detection_box.translation)
        kitti_bbox[3:6] = np.array(detection_box.size)
        kitti_bbox[6] = quaternion_yaw(Quaternion(detection_box.rotation))
        return kitti_bbox

    @classmethod
    def from_kitti(
        cls, kitti_box, idx: int, score: float, detection_name: str, label: int, vel
    ):
        """Convert kitti bounding boxes [x, y, z, l, w, h, yaw] into detection boxes."""
        content = dict()
        content["sample_token"] = str(idx)
        content["translation"] = [kitti_box[0], kitti_box[1], kitti_box[2]]
        content["size"] = [kitti_box[3], kitti_box[4], kitti_box[5]]
        q = yaw_quaternion(kitti_box[6])
        content["rotation"] = list(q.elements)
        content["velocity"] = [vel[0], vel[1]]
        content["detection_name"] = detection_name
        content["detection_score"] = -1.0 if score is None else score
        content["attribute_name"] = ""
        content["label"] = label
        return cls.deserialize(content)

    def get_velocity(self):
        """
        Get the velocity of the bounding box.

        Returns
        -------
            A numpy array of [status, vx, vy]
        """
        velocity = np.zeros((3,), dtype=float)
        # velocity = np.empty((3, ), dtype=float)
        velocity[1:3] = np.array(self.velocity)
        if np.isclose(self.velocity[0], 0.0) and np.isclose(self.velocity[1], 0.0):
            velocity[0] = 0.0  # static objects
        elif np.isclose(self.velocity[0], VALID_VELOCITY_LIMIT) and np.isclose(
            self.velocity[1], VALID_VELOCITY_LIMIT
        ):
            velocity[0] = 2.0  # unknown
        else:
            velocity[0] = 1.0  # dynamic objects

        return velocity


class EvalBoxes:
    """Data class that groups EvalBox instances by sample."""

    def __init__(self):
        """Initialize the EvalBoxes for GT or predictions."""
        self.boxes = defaultdict(list)

    def __repr__(self):
        """Describe the number of boxes and sample size in EvalBoxes instance."""
        return "EvalBoxes with {} boxes across {} samples".format(
            len(self.all), len(self.sample_tokens)
        )

    def __getitem__(self, item: str) -> List[DetectionBox]:
        """Return list of DetectionBox instances."""
        return self.boxes[item]

    def __eq__(self, other):
        """Check if two instances of the class are equal."""
        if not set(self.sample_tokens) == set(other.sample_tokens):
            return False
        for token in self.sample_tokens:
            if not len(self[token]) == len(other[token]):
                return False
            for box1, box2 in zip(self[token], other[token]):
                if box1 != box2:
                    return False
        return True

    def __len__(self):
        """Return number of DetectionBox in EvalBoxes instance."""
        return len(self.boxes)

    @property
    def all(self) -> List[DetectionBox]:  # noqa: A003
        """Returns all EvalBoxes in a list."""
        ab = []
        for sample_token in self.sample_tokens:
            ab.extend(self[sample_token])
        return ab

    @property
    def sample_tokens(self) -> List[str]:
        """Returns a list of all keys."""
        return list(self.boxes.keys())

    def add_boxes(self, sample_token: str, boxes: List[DetectionBox]) -> None:
        """Add a list of boxes."""
        self.boxes[sample_token].extend(boxes)

    def serialize(self) -> dict:
        """Serialize instance into json-friendly format."""
        return {
            key: [box.serialize() for box in boxes] for key, boxes in self.boxes.items()
        }

    @classmethod
    def deserialize(cls, content: dict, box_cls):
        """
        Initialize from serialized content.

        Parameters
        ----------
        content:
            A dictionary with the serialized content of the box.
        box_cls:
            The class of the boxes, DetectionBox or TrackingBox.

        Returns
        -------
        EvalBoxes
            Class with deserialized content of the box
        """
        eb = cls()
        for sample_token, boxes in content.items():
            eb.add_boxes(sample_token, [box_cls.deserialize(box) for box in boxes])
        return eb


class MetricData(abc.ABC):
    """Abstract base class for the *MetricData classes specific to each task."""

    @abc.abstractmethod
    def serialize(self):
        """Serialize instance into json-friendly format."""

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content: dict):
        """Initialize from serialized content."""


def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> np.floating:
    """
    L2 distance between the box centers (xy only).

    Parameters
    ----------
    gt_box:
        GT annotation sample.
    pred_box:
        Predicted sample.

    Returns
    -------
        L2 distance.
    """
    return np.linalg.norm(
        np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2])
    )


def _rotation(pts: np.ndarray, theta: float) -> np.ndarray:
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = pts @ r.T
    return pts


def _make_box_pts(
    pos_x: float, pos_y: float, yaw: float, dim_x: float, dim_y: float
) -> np.ndarray:
    hx = dim_x / 2
    hy = dim_y / 2

    pts = np.asarray([(-hx, hy), (hx, hy), (hx, -hy), (-hx, -hy)])
    pts = _rotation(pts, yaw)
    pts += (pos_x, pos_y)
    return pts


def iou_2d_inverse(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """Calculate the Inverse IoU between GT and Pred boxes.

    Parameters
    ----------
    gt_box : EvalBox
        GT bounding box
    pred_box : EvalBox
        Predicted bounding box

    Returns
    -------
    float
        Inverse IoU (1 - IoU) between GT and Pred boxes
    """
    t = Polygon(
        _make_box_pts(
            gt_box.translation[0],
            gt_box.translation[1],
            quaternion_yaw(Quaternion(gt_box.rotation)),
            gt_box.size[0],
            gt_box.size[1],
        )
    )
    p = Polygon(
        _make_box_pts(
            pred_box.translation[0],
            pred_box.translation[1],
            quaternion_yaw(Quaternion(pred_box.rotation)),
            pred_box.size[0],
            pred_box.size[1],
        )
    )
    if t.union(p).area == 0:
        return 1.0
    iou = t.intersection(p).area / t.union(p).area
    return 1.0 - iou


class DetectionConfig:
    """Data class that specifies the detection evaluation settings."""

    def __init__(
        self,
        class_range: Dict[str, int],
        dist_fcn: str,
        dist_ths: List[float],
        dist_th_tp: float,
        range_bins: List[List[float]],
        overall_range_id: int,
        min_recall: float,
        min_precision: float,
        mean_ap_weight: int,
        ego_front_dist: float,
        ego_rear_dist: float,
        roi_eval: bool,
        roi: Dict[str, List[float]],
    ):
        # assert set(class_range.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self._range_bins = range_bins
        self.overall_range_id = overall_range_id
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.mean_ap_weight = mean_ap_weight
        self.ego_front_dist = ego_front_dist
        self.ego_rear_dist = ego_rear_dist
        self.roi_eval = roi_eval
        self.roi = roi
        self.range_bin_list = self.get_range_bin_list()
        self.class_names = self.class_range.keys()

    def __eq__(self, other):
        """Check if two instances of the class are equal."""
        eq = True
        for key in self.serialize():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """Serialize instance into json-friendly format."""
        range_bin_str = "_".join(
            ["-".join(str(x) for x in y) for y in self._range_bins]
        )
        return {
            "class_range": self.class_range,
            "dist_fcn": self.dist_fcn,
            "dist_ths": self.dist_ths,
            "dist_th_tp": self.dist_th_tp,
            "range_bins": range_bin_str,
            "min_recall": self.min_recall,
            "min_precision": self.min_precision,
            "mean_ap_weight": self.mean_ap_weight,
            "ego_front_dist": self.ego_front_dist,
            "ego_rear_dist": self.ego_rear_dist,
            "roi_dict": {"eval": self.roi_eval, "roi": self.roi},
        }

    @classmethod
    def deserialize(cls, content: dict):
        """Initialize from serialized dictionary."""
        return cls(
            content["class_range"],
            content["dist_fcn"],
            content["dist_ths"],
            content["dist_th_tp"],
            content["range_bins"],
            content["overall_range_id"],
            content["min_recall"],
            content["min_precision"],
            content["mean_ap_weight"],
            content["ego_front_dist"],
            content["ego_rear_dist"],
            content["roi_dict"]["eval"],
            content["roi_dict"]["roi"],
        )

    @property
    def dist_fcn_callable(self):
        """Return the distance function corresponding to the dist_fcn string."""
        if self.dist_fcn == "center_distance":
            return center_distance
        elif self.dist_fcn == "iou_2d":
            return iou_2d_inverse
        else:
            raise Exception("Error: Unknown distance function %s!" % self.dist_fcn)

    def get_range_bin_list(self) -> List[Tuple[float, ...]]:
        """
        Get the range bins in the format of a list of min and max range.

        Returns
        -------
            A list of range bins, in which each element is a tuple of (min_range, max_range)
        """
        if self.roi_eval:
            range_bin_list = [
                tuple(b)
                for b in self._range_bins
                if b[0] >= self.roi["xrange"][0] and b[1] <= self.roi["xrange"][1]
            ]
            range_bin_list.append(tuple(self.roi["xrange"]))  # type: ignore
            self.overall_range_id = -1

        else:
            range_bin_list = [tuple(x) for x in self._range_bins]

        return range_bin_list

    def reset_bins(self):
        """Reset the range bins to include everything."""
        self._range_bins = [self._range_bins[0], self._range_bins[-1]]


class DetectionMetricData(MetricData):
    """Data class holds accumulated and interpolated data required to calculate the detection metrics."""

    nelem = 101

    def __init__(
        self,
        recall: npt.NDArray,
        precision: npt.NDArray,
        confidence: npt.NDArray,
        trans_err: npt.NDArray,
        vel_err: npt.NDArray,
        scale_err: npt.NDArray,
        iou_2d: npt.NDArray,
        orient_err: npt.NDArray,
        attr_err: npt.NDArray,
        pos_err_x: npt.NDArray,
        pos_err_y: npt.NDArray,
        pos_err_z: npt.NDArray,
        len_err: npt.NDArray,
        wid_err: npt.NDArray,
        ht_err: npt.NDArray,
        long_ext_acc: npt.NDArray,
        lat_ext_acc: npt.NDArray,
        zero_division: bool = False,
        gt_filtered: int = 0,
        gt_unfiltered: int = 0,
        pred_filtered: int = 0,
        pred_unfiltered: int = 0,
        samples: int = 0,
        range_str: str = "",
    ):
        # Assert lengths.
        assert len(recall) == self.nelem
        assert len(precision) == self.nelem
        assert len(confidence) == self.nelem
        assert len(trans_err) == self.nelem
        assert len(vel_err) == self.nelem
        assert len(scale_err) == self.nelem
        assert len(iou_2d) == self.nelem
        assert len(orient_err) == self.nelem
        assert len(attr_err) == self.nelem
        assert len(pos_err_x) == self.nelem
        assert len(pos_err_y) == self.nelem
        assert len(pos_err_z) == self.nelem
        assert len(len_err) == self.nelem
        assert len(wid_err) == self.nelem
        assert len(ht_err) == self.nelem
        assert len(long_ext_acc) == self.nelem
        assert len(lat_ext_acc) == self.nelem

        # Assert ordering.
        assert all(
            confidence == sorted(confidence, reverse=True)
        )  # Confidences should be descending.
        assert all(recall == sorted(recall))  # Recalls should be ascending.

        # Set attributes explicitly to help IDEs figure out what is going on.
        self.recall = recall
        self.precision = precision
        self.confidence = confidence
        self.trans_err = trans_err
        self.vel_err = vel_err
        self.scale_err = scale_err
        self.iou_2d = iou_2d
        self.orient_err = orient_err
        self.attr_err = attr_err
        self.pos_err_x = pos_err_x
        self.pos_err_y = pos_err_y
        self.pos_err_z = pos_err_z
        self.len_err = len_err
        self.wid_err = wid_err
        self.ht_err = ht_err
        self.long_ext_acc = long_ext_acc
        self.lat_ext_acc = lat_ext_acc
        self.zero_division = zero_division
        self.gt_filtered = gt_filtered
        self.gt_unfiltered = gt_unfiltered
        self.pred_filtered = pred_filtered
        self.pred_filtered = pred_unfiltered
        self.samples = samples
        self.range_str = range_str

    def __eq__(self, other):
        """Check if two instances of the class are equal."""
        eq = True
        for key in self.serialize():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    @property
    def max_recall_ind(self):
        """Returns index of max recall achieved."""
        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(self.confidence)[0]

        # If there are no matches, all the confidence values will be zero.
        max_recall_ind = 0 if len(non_zero) == 0 else non_zero[-1]

        return max_recall_ind

    @property
    def max_recall(self):
        """Returns max recall achieved."""
        return self.recall[self.max_recall_ind]

    def serialize(self):
        """Serialize instance into json-friendly format."""
        return {
            "recall": self.recall.tolist(),
            "precision": self.precision.tolist(),
            "confidence": self.confidence.tolist(),
            "trans_err": self.trans_err.tolist(),
            "vel_err": self.vel_err.tolist(),
            "scale_err": self.scale_err.tolist(),
            "iou_2d": self.iou_2d.tolist(),
            "orient_err": self.orient_err.tolist(),
            "attr_err": self.attr_err.tolist(),
            "pos_err_x": self.pos_err_x.tolist(),
            "pos_err_y": self.pos_err_y.tolist(),
            "pos_err_z": self.pos_err_z.tolist(),
            "len_err": self.len_err.tolist(),
            "wid_err": self.wid_err.tolist(),
            "ht_err": self.ht_err.tolist(),
            "long_ext_acc": self.long_ext_acc.tolist(),
            "lat_ext_acc": self.lat_ext_acc.tolist(),
        }

    @classmethod
    def deserialize(cls, content: dict):
        """Initialize from serialized content."""
        return cls(
            recall=np.array(content["recall"]),
            precision=np.array(content["precision"]),
            confidence=np.array(content["confidence"]),
            trans_err=np.array(content["trans_err"]),
            vel_err=np.array(content["vel_err"]),
            scale_err=np.array(content["scale_err"]),
            iou_2d=np.array(content["iou_2d"]),
            orient_err=np.array(content["orient_err"]),
            attr_err=np.array(content["attr_err"]),
            pos_err_x=np.array(content["pos_err_x"]),
            pos_err_y=np.array(content["pos_err_y"]),
            pos_err_z=np.array(content["pos_err_z"]),
            len_err=np.array(content["len_err"]),
            wid_err=np.array(content["wid_err"]),
            ht_err=np.array(content["ht_err"]),
            long_ext_acc=np.array(content["long_ext_acc"]),
            lat_ext_acc=np.array(content["lat_ext_acc"]),
        )

    @classmethod
    def no_predictions(cls):
        """Return a md instance corresponding to having no predictions."""
        return cls(
            recall=np.linspace(0, 1, cls.nelem),
            precision=np.zeros(cls.nelem),
            confidence=np.zeros(cls.nelem),
            trans_err=np.ones(cls.nelem),
            vel_err=np.ones(cls.nelem),
            scale_err=np.ones(cls.nelem),
            iou_2d=np.zeros(cls.nelem),
            orient_err=np.ones(cls.nelem),
            attr_err=np.ones(cls.nelem),
            pos_err_x=np.ones(cls.nelem),
            pos_err_y=np.ones(cls.nelem),
            pos_err_z=np.ones(cls.nelem),
            len_err=np.ones(cls.nelem),
            wid_err=np.ones(cls.nelem),
            ht_err=np.ones(cls.nelem),
            long_ext_acc=np.ones(cls.nelem),
            lat_ext_acc=np.ones(cls.nelem),
            zero_division=True,
        )

    @classmethod
    def random_md(cls):
        """Return an md instance corresponding to a random results."""
        np_rand = np.random.default_rng()
        return cls(
            recall=np.linspace(0, 1, cls.nelem),
            precision=np_rand.random(cls.nelem),
            confidence=np.linspace(0, 1, cls.nelem)[::-1],
            trans_err=np_rand.random(cls.nelem),
            vel_err=np_rand.random(cls.nelem),
            scale_err=np_rand.random(cls.nelem),
            iou_2d=np_rand.random(cls.nelem),
            orient_err=np_rand.random(cls.nelem),
            attr_err=np_rand.random(cls.nelem),
            pos_err_x=np_rand.random(cls.nelem),
            pos_err_y=np_rand.random(cls.nelem),
            pos_err_z=np_rand.random(cls.nelem),
            len_err=np_rand.random(cls.nelem),
            wid_err=np_rand.random(cls.nelem),
            ht_err=np_rand.random(cls.nelem),
            long_ext_acc=np_rand.random(cls.nelem),
            lat_ext_acc=np_rand.random(cls.nelem),
        )


class DetectionMetricDataList:
    """Class to store a set of DetectionMetricData in a dict indexed by (name, match-distance)."""

    def __init__(self):
        self.md = {}

    def __getitem__(self, key):
        """Return DetectionMetricData instance corresponding to 'key'."""
        return self.md[key]

    def __eq__(self, other):
        """Check if two instances of the class are equal."""
        eq = True
        for key in self.md:
            eq = eq and self[key] == other[key]
        return eq

    def get_class_data(
        self, detection_name: str
    ) -> List[Tuple[DetectionMetricData, float, Tuple]]:
        """Get all the MetricData entries for a certain detection_name."""
        return [
            (md, dist_th, bins)
            for (name, dist_th, bins), md in self.md.items()
            if name == detection_name
        ]

    def get_dist_data(
        self, dist_th: float
    ) -> List[Tuple[DetectionMetricData, str, Tuple]]:
        """Get all the MetricData entries for a certain match_distance."""
        return [
            (md, detection_name, bins)
            for (detection_name, dist, bins), md in self.md.items()
            if dist == dist_th
        ]

    def get_class_dist_data(
        self, detection_name: str, dist_th: float
    ) -> List[Tuple[DetectionMetricData, Tuple]]:
        """Get class distribution information."""
        return [
            (md, bins)
            for (name, dist, bins), md in self.md.items()
            if dist == dist_th and name == detection_name
        ]

    def set(
        self,
        detection_name: str,
        match_distance: float,
        bins: Tuple,
        data: DetectionMetricData,
    ):  # noqa: A003
        """Ses the MetricData entry for a certain detection_name and match_distance."""
        self.md[(detection_name, match_distance, bins)] = data

    def serialize(self) -> dict:
        """Serialize contents of the metric data dictionary."""
        return {
            key[0] + ":" + str(key[1]): value.serialize()
            for key, value in self.md.items()
        }

    @classmethod
    def deserialize(cls, content: dict):
        """Deserialize the contents of a given metric data dictionary."""
        mdl = cls()
        for key, md in content.items():
            name, distance, bins = key.split(":")
            mdl.set(name, float(distance), bins, DetectionMetricData.deserialize(md))
        return mdl


class DetectionMetrics:
    """Stores average precision and true positive metric results. Provides properties to summarize."""

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self._label_aps = dict()  # defaultdict(inner_default)
        self._label_aps_binned = dict()  # defaultdict(inner_default)
        self._label_tp_errors = dict()  # defaultdict(inner_default)
        self._label_tp_errors_binned = dict()  # defaultdict(inner_default)
        self.eval_time = None

    def add_label_ap(self, detection_name: str, dist_th: float, ap: float) -> None:
        """Populate the average precision for a given class."""
        self._label_aps[detection_name] = self._label_aps.get(detection_name, dict())
        self._label_aps[detection_name][dist_th] = self._label_aps[detection_name].get(
            dist_th, round(ap, 6)
        )

    def add_label_ap_binned(
        self, detection_name: str, dist_th: float, ap: float, range_bin: str
    ) -> None:
        """Populate the average precision for a given range and class."""
        self._label_aps_binned[range_bin] = self._label_aps_binned.get(
            range_bin, dict()
        )
        self._label_aps_binned[range_bin][detection_name] = self._label_aps_binned[
            range_bin
        ].get(detection_name, dict())
        self._label_aps_binned[range_bin][detection_name][dist_th] = (
            self._label_aps_binned[range_bin][detection_name].get(dist_th, round(ap, 6))
        )

    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        """Return the ap for a given class and distance threshold."""
        return self._label_aps[detection_name][dist_th]

    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        """Populate the true positive for a given class and metric."""
        self._label_tp_errors[detection_name] = self._label_aps.get(
            detection_name, dict()
        )
        self._label_tp_errors[detection_name][metric_name] = self._label_aps[
            detection_name
        ].get(metric_name, round(tp, 6))

    def add_label_tp_binned(
        self, detection_name: str, metric_name: str, tp: float, range_bin: str
    ):
        """Populate the true positive for a given range, class and metric."""
        self._label_tp_errors_binned[range_bin] = self._label_tp_errors_binned.get(
            range_bin, dict()
        )
        self._label_tp_errors_binned[range_bin][detection_name] = (
            self._label_tp_errors_binned[range_bin].get(detection_name, dict())
        )
        self._label_tp_errors_binned[range_bin][detection_name][metric_name] = (
            self._label_tp_errors_binned[range_bin][detection_name].get(
                metric_name, round(tp, 6)
            )
        )

    def add_label_tp_std_binned(
        self, detection_name: str, metric_name: str, tp: float, range_bin: str
    ):
        """Populate the true positive stdandard deviation for a given range, class and metric."""
        self._label_tp_errors_binned[range_bin] = self._label_tp_errors_binned.get(
            range_bin, dict()
        )
        self._label_tp_errors_binned[range_bin][detection_name] = (
            self._label_tp_errors_binned[range_bin].get(detection_name, dict())
        )
        self._label_tp_errors_binned[range_bin][detection_name][
            metric_name + "_std"
        ] = self._label_tp_errors_binned[range_bin][detection_name].get(
            metric_name + "_std", round(tp, 6)
        )

    def get_label_tp(self, detection_name: str, metric_name: str) -> float:
        """Return the tp for a give class and metric."""
        return self._label_tp_errors[detection_name][metric_name]

    def add_runtime(self, eval_time: float) -> None:
        """Store the runtime of the evaluation."""
        self.eval_time = eval_time

    @property
    def mean_dist_aps(self) -> Dict[str, float]:
        """Calculates the mean over distance thresholds for each label."""
        return {
            str(class_name): float(np.mean(list(d.values())))
            for class_name, d in self._label_aps.items()
        }

    @property
    def mean_ap(self) -> Union[int, np.float32]:
        """Calculates the mean AP by averaging over distance thresholds and classes."""
        mean_aps = [
            float(sum(d[dist_ths] for d in self._label_aps.values() if dist_ths in d))
            / len(self._label_aps)
            for dist_ths in self.cfg.dist_ths
            if any(dist_ths in d for d in self._label_aps.values())
        ]
        if not mean_aps:  # Check if the list is empty
            return 0
        return np.mean(mean_aps, dtype=np.float32)

    @property
    def tp_errors(self) -> Dict[str, float]:
        """Calculates the mean true positive error across all classes for each metric."""
        errors = {}
        for metric_name in TP_METRICS + TP_METRICS_SIZE_PLUS_SCALE:
            class_errors = []
            for detection_name in self.cfg.class_names:
                class_errors.append(self.get_label_tp(detection_name, metric_name))

            errors[metric_name] = float(np.nanmean(class_errors))

        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        """Return tp scores."""
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in TP_METRICS + TP_METRICS_SIZE_PLUS_SCALE:
            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            scores[metric_name] = score

        return scores

    @property
    def nd_score(self) -> float:
        """Compute the nuScenes detection score (NDS, weighted sum of the individual scores).

        Returns
        -------
        float:
            The NDS score
        """
        # Summarize.
        total = float(
            self.cfg.mean_ap_weight * self.mean_ap
            + np.sum(list(self.tp_scores.values()))
        )

        # Normalize.
        total = total / float(self.cfg.mean_ap_weight + len(self.tp_scores.keys()))

        return total

    def format_nested_dict_to_list(self, metrics: dict, indent: int = 0) -> List[str]:
        """
        Format a nested dictionary into a readable table-like string.

        Parameters
        ----------
        metrics (dict):
            A nested dictionary containing the metrics to format.
            The dictionary can contain other dictionaries as values for nested metrics.

        indent (int):
            The indentation level for nested categories. Default is 0 for the top level.

        Returns
        -------
        list:
            A list of strings, where each string is a formatted line of the table.
        """
        lines = []

        # Function to create a formatted row
        def create_row(key, value, indent, is_category=False):
            spacer = " " * indent
            if is_category:
                return f"{spacer}{key:<30} |"
            else:
                return f"{spacer}{key:<30} | {value}"

        for key, value in metrics.items():
            if isinstance(value, dict):
                # For nested dictionaries, add a category line and then process the nested dict
                lines.append(create_row(key, "", indent, is_category=True))
                nested_lines = self.format_nested_dict_to_list(value, indent + 4)
                lines.extend(nested_lines)
            else:
                # For regular key-value pairs, add a line with the key and value
                lines.append(create_row(key, value, indent))

        return lines

    def serialize(self):
        """Create a serialized metrics dictionary."""
        metrics_dict = {
            "label_aps": self._label_aps,
            "mean_dist_aps": self.mean_dist_aps,
            "aps": self._label_aps_binned,
            "mean_ap": self.mean_ap,
            "tps": self._label_tp_errors_binned,
            "tp_errors": self.tp_errors,
            "tp_scores": self.tp_scores,
            "nd_score": self.nd_score,
            "eval_time": self.eval_time,
            "cfg": self.cfg.serialize(),
        }

        def remove_gt_filtered(d, flatten_class_ap: bool = False):
            if not isinstance(d, dict):
                return d

            cleaned = {
                k: remove_gt_filtered(v, flatten_class_ap=flatten_class_ap)
                for k, v in d.items()
                if k != "gt_filtered"
            }

            if flatten_class_ap and cleaned:
                # Flatten {"6.0": value} -> value for display-only convenience.
                values = list(cleaned.values())
                if len(values) == 1 and not isinstance(values[0], dict):
                    return values[0]

            return cleaned

        metrics_dict_to_plot = {
            "mean_ap": self.mean_ap,
            "nd_score": self.nd_score,
            "eval_time": self.eval_time,
            "label_aps": remove_gt_filtered(self._label_aps, flatten_class_ap=True),
            "aps_binned": remove_gt_filtered(
                self._label_aps_binned, flatten_class_ap=True
            ),
        }
        print("=======")
        print("Evaluation Metrics:")
        print("\n".join(self.format_nested_dict_to_list(metrics_dict_to_plot)))
        print("=======")
        return metrics_dict

    @classmethod
    def deserialize(cls, content: dict):
        """Initialize from serialized dictionary."""
        cfg = DetectionConfig.deserialize(content["cfg"])

        metrics = cls(cfg=cfg)
        metrics.add_runtime(content["eval_time"])

        for detection_name, label_aps in content["label_aps"].items():
            for dist_th, ap in label_aps.items():
                metrics.add_label_ap(
                    detection_name=detection_name, dist_th=float(dist_th), ap=float(ap)
                )

        for detection_name, label_tps in content["label_tp_errors"].items():
            for metric_name, tp in label_tps.items():
                metrics.add_label_tp(
                    detection_name=detection_name, metric_name=metric_name, tp=float(tp)
                )

        return metrics

    def __eq__(self, other):
        """Check if two instances of the class are equal."""
        eq = True
        eq = eq and self._label_aps == other._label_aps
        eq = eq and self._label_tp_errors == other._label_tp_errors
        eq = eq and self.eval_time == other.eval_time
        eq = eq and self.cfg == other.cfg

        return eq
