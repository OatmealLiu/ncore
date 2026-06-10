import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes

from .eval_truckdrive_metric.eval_truckdrive_dataclasses import (
    DetectionBox,
    DetectionConfig,
    EvalBoxes,
    yaw_quaternion,
)
from .eval_truckdrive_metric.eval_truckdrive_evaluate import evaluate


@METRICS.register_module()
class TruckDriveMetric(BaseMetric):

    def __init__(
        self,
        ann_file: str,
        metric_config_path: str = "./eval_truckdrive_metric/truckdrive_metric_config_shortrange.json",
        max_elems: int = -1,
        label2cat: dict = {
            "DontCare": -1,
            "Bike": 0,
            "Passenger-Car": 1,
            "Person": 2,
            "RoadObstruction": 3,
            "SemiTruck-Cab": 4,
            "SemiTruck-Trailer": 5,
            "Vehicle": 6,
            "TrafficSign": -1,
            "EmergencyVehicle": -1,
        },
        gt_label_mapping: dict = {
            -1: -1,
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: -1,
            8: -1,
        },
        backend_args: Optional[dict] = None,
    ) -> None:
        super(TruckDriveMetric, self).__init__()
        self.ann_file = ann_file
        self.backend_args = backend_args
        self.max_elements = max_elems
        self.label2cat = label2cat
        self.gt_label_mapping = gt_label_mapping
        self.metric_config = json.load(open(metric_config_path, "r"))

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample["pred_instances_3d"]
            pred_2d = data_sample["pred_instances"]
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to("cpu")
            result["pred_instances_3d"] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to("cpu")
            result["pred_instances"] = pred_2d
            sample_idx = data_sample["sample_idx"]
            result["sample_idx"] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        data_infos = load(self.ann_file, backend_args=self.backend_args)["data_list"]
        if self.max_elements != -1:
            data_infos = data_infos[: self.max_elements]
        metric_dict = evaluate_truckdrive_metrics(
            data_infos,
            results,
            self.label2cat,
            self.gt_label_mapping,
            self.metric_config,
        )

        def _to_builtin(x):
            if isinstance(x, dict):
                return {str(k): _to_builtin(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_to_builtin(v) for v in x]
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating,)):
                return float(x)
            if isinstance(x, (np.bool_)):
                return bool(x)
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, torch.Tensor):
                return (
                    x.detach().cpu().tolist()
                    if x.numel() > 1
                    else x.detach().cpu().item()
                )
            return x

        metric_dict = _to_builtin(metric_dict)
        return metric_dict


def evaluate_truckdrive_metrics(
    data_infos,
    results,
    label2cat,
    gt_label_mapping,
    metric_config,
):
    metric_config = metric_config

    class_names_exclude = [
        # "Bike",
        # "Passenger-Car",
        # "Person",
        # "RoadObstruction",
        # "SemiTruck-Cab",
        # "SemiTruck-Trailer",
        # "Vehicle",
    ]

    label2cat = label2cat
    # label2cat = {'Bike': 0, 'Passenger-Car': 1, 'Person': 2, 'RoadObstruction': 3, 'SemiTruck-Cab': 4, 'SemiTruck-Trailer': 5, 'Vehicle': 6}

    label_class_mapping = {v: k for k, v in label2cat.items()}
    # label_class_mapping = {0:'Bike', 1: 'Passenger-Car', 2: 'Person', 3: 'RoadObstruction', 4: 'SemiTruck-Cab', 5: 'SemiTruck-Trailer', 6: 'Vehicle'}

    class_names_exclude = [k for k, v in label2cat.items() if v == -1]
    new_cat = {k: v for k, v in label2cat.items() if k not in class_names_exclude}
    cfg = DetectionConfig(
        class_range=new_cat,
        dist_fcn=metric_config["dist_fcn"],
        dist_ths=metric_config["dist_ths"],
        dist_th_tp=metric_config["dist_th_tp"],
        range_bins=metric_config["range_bins"],
        overall_range_id=metric_config["overall_range_id"],
        min_recall=metric_config["min_recall"],
        min_precision=metric_config["min_precision"],
        mean_ap_weight=metric_config["mean_ap_weight"],
        ego_front_dist=metric_config["ego_front_dist"],
        ego_rear_dist=metric_config["ego_rear_dist"],
        roi_eval=metric_config["roi_dict"]["eval"],
        roi=metric_config["roi_dict"]["roi"],
    )
    # 1) Gather GT from self.data_infos
    gt_annos = [info["instances"] for info in data_infos]  # list[list[dict]]
    # 2) Detections
    dt_annos = results

    # Convert to correct format
    gt_bboxes = EvalBoxes()
    pred_bboxes = EvalBoxes()
    for idx, (gt, dt) in enumerate(zip(gt_annos, dt_annos)):
        sample_idx = idx
        gt_lab = []
        gt_bbo = []
        for gt_single in gt:
            gt_lab.append(gt_single["bbox_label_3d"])
            gt_bbo.append(gt_single["bbox_3d"])
        gt_lab = np.array(gt_lab)
        gt_bbo = np.array(gt_bbo)
        scene_gts = dict(
            gt_bboxes_3d=LiDARInstance3DBoxes(
                gt_bbo, box_dim=7, origin=(0.0, 0.0, 0.0)
            ),
            gt_labels_3d=torch.Tensor(gt_lab),
        )
        scene_preds = dict(
            bboxes_3d=dt["pred_instances_3d"]["bboxes_3d"],
            labels_3d=dt["pred_instances_3d"]["labels_3d"],
            scores_3d=dt["pred_instances_3d"]["scores_3d"],
        )
        gt_bboxes.add_boxes(
            str(sample_idx),
            convert_lidar3d_to_detection_boxes(
                bboxes_3d=scene_gts["gt_bboxes_3d"],
                labels_3d=scene_gts["gt_labels_3d"],
                num_pts=-1,  # TODO parse from dataset
                sample_token=str(sample_idx),
                gt=True,
                class_names_exclude=class_names_exclude,
                label_class_mapping=label_class_mapping,
                gt_label_mapping=gt_label_mapping,
            ),
        )

        pred_bboxes.add_boxes(
            str(sample_idx),
            convert_lidar3d_to_detection_boxes(
                bboxes_3d=scene_preds["bboxes_3d"],
                labels_3d=scene_preds["labels_3d"],
                scores_3d=scene_preds["scores_3d"],
                num_pts=-1,
                sample_token=str(sample_idx),
                gt=False,
                class_names_exclude=class_names_exclude,
                label_class_mapping=label_class_mapping,
            ),
        )

    # Compute metrics
    metrics, _ = evaluate(
        gt_boxes=gt_bboxes,
        pred_boxes=pred_bboxes,
        cfg=cfg,
        verbose=True,
        gt_difficult=None,  # TODO: load difficult flags from dataset
    )
    return metrics.serialize()


def convert_lidar3d_to_detection_boxes(
    bboxes_3d: LiDARInstance3DBoxes,
    labels_3d: torch.Tensor,
    num_pts: torch.Tensor,
    sample_token: str,
    gt: bool,
    label_class_mapping: Dict,
    class_names_exclude: Dict,
    scores_3d: Optional[torch.Tensor] = None,
    gt_label_mapping: Optional[Dict] = None,
):
    """
    Convert LiDAR 3D boxes into detection boxes.

    This function iterates through each box in the provided LiDARInstance3DBoxes object,
    converting them into DetectionBox format with detailed attributes such as
    translation, size, rotation, velocity, etc.

    Parameters
    ----------
        bboxes_3d (LiDARInstance3DBoxes): 3D bounding boxes.
        labels_3d (torch.Tensor): Tensor of labels corresponding to each bounding box.
        scores_3d (Union[torch.Tensor, None]): Tensor of scores for each bounding box.
        num_pts (torch.Tensor): Number of points inside each bounding box.
        sample_token (str): Sample token associated with each detection.
        gt (bool): Flag indicating if the boxes are ground truth.

    Returns
    -------
        list: A list of DetectionBox objects created from the LiDAR boxes.
    """
    detection_boxes = []
    for box_idx in range(bboxes_3d.tensor.shape[0]):
        detection_score = -1.0 if gt else float(scores_3d[box_idx].cpu().numpy())
        translation = tuple([float(i) for i in bboxes_3d.center[box_idx].cpu().numpy()])
        size = tuple([float(i) for i in bboxes_3d.dims[box_idx].cpu().numpy()])
        yaw = float(bboxes_3d.yaw[box_idx].cpu().numpy())
        rotation = yaw_quaternion(yaw).elements
        # velocity_x = float(bboxes_3d.tensor[box_idx, 7].cpu().numpy())
        # velocity_y = float(bboxes_3d.tensor[box_idx, 8].cpu().numpy())
        label = int(labels_3d[box_idx]) if gt else int(labels_3d[box_idx].cpu().numpy())
        if gt:
            label = gt_label_mapping[label]
        detection_name = label_class_mapping[label]
        if detection_name in class_names_exclude:
            # print(f'excluding: {detection_name}')
            continue
        detection_box = DetectionBox(
            sample_token=sample_token,
            translation=translation,
            size=size,
            rotation=rotation,
            # velocity=(velocity_x, velocity_y),
            # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
            num_pts=num_pts,
            # The class name used in the detection challenge.
            detection_name=detection_name,
            # GT samples do not have a score.
            detection_score=detection_score,
            label=label,
        )
        detection_boxes.append(detection_box)
    return detection_boxes
