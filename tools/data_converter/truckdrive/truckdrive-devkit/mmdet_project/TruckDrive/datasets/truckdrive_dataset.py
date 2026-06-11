import sys
from typing import Callable, List, Union
import numpy as np

# Compatibility shim: pickle files created with numpy>=2.0 reference
# numpy._core, which does not exist in numpy<2.0.
if not hasattr(np, "_core"):
    import numpy.core as _numpy_core

    sys.modules["numpy._core"] = _numpy_core
    sys.modules.setdefault("numpy._core.multiarray", _numpy_core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", _numpy_core.numeric)

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmengine.fileio import load
from mmengine.dist.utils import get_dist_info


@DATASETS.register_module()
class TruckDriveDataset(Det3DDataset):
    r"""TruckDrive Dataset.

    This class serves as the API for experiments on the TruckDrive Dataset.

    Please refer to `TruckDrive Dataset <https://`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
                to convert to the FOV-based data type to support image-based
                detector.
            - 'fov_image_based': Only load the instances inside the default
                cam, and need to convert to the FOV-based data type to support
                image-based detector.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        with_velocity (bool): Whether to include velocity prediction
            into the experiments. Defaults to True.
        use_valid_flag (bool): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """

    METAINFO = {
        "classes": [
            "Bike",
            "Passenger-Car",
            "Person",
            "RoadObstruction",
            "SemiTruck-Cab",
            "SemiTruck-Trailer",
            "Vehicle",
            "TrafficSign",
            "EmergencyVehicle",
        ],
        "categories": {
            "Bike": 0,
            "Passenger-Car": 1,
            "Person": 2,
            "RoadObstruction": 3,
            "SemiTruck-Cab": 4,
            "SemiTruck-Trailer": 5,
            "Vehicle": 6,
            "TrafficSign": 7,
            "EmergencyVehicle": 8,
        },
    }
    CLASSES = (
        "DontCare",
        "Bike",
        "Passenger-Car",
        "Person",
        "RoadObstruction",
        "SemiTruck-Cab",
        "SemiTruck-Trailer",
        "Vehicle",
        "TrafficSign",  #  'DontCare'
        "EmergencyVehicle",  #  'DontCare'
    )

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        batch_size: int,
        pipeline: List[Union[dict, Callable]] = [],
        box_type_3d: str = "LiDAR",
        load_type: str = "frame_based",
        modality: dict = dict(
            use_camera=False,
            use_lidar=True,
        ),
        filter_empty_gt: bool = True,
        test_mode: bool = False,
        with_velocity: bool = True,
        use_valid_flag: bool = False,
        dontcare_labels: list = ["DontCare", "TrafficSign", "EmergencyVehicle"],
        max_elems: int = -1,
        **kwargs,
    ) -> None:
        self.batch_size = batch_size
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity
        self.max_elems = max_elems
        self.dontcare_labels = dontcare_labels

        # TODO: Redesign multi-view data process in the future
        assert load_type in ("frame_based", "mv_image_based", "fov_image_based")
        self.load_type = load_type

        assert box_type_3d.lower() in ("lidar", "camera")
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            modality=modality,
            pipeline=pipeline,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs,
        )

        self.update_label_mapping_and_counts()

    def update_label_mapping_and_counts(self) -> None:

        self.label_mapping = {}
        self.label2cat = {}
        i = 0
        for lbl in self.METAINFO["classes"]:
            if lbl in self.dontcare_labels:
                self.label_mapping[self.METAINFO["categories"][lbl]] = -1
                self.label2cat[lbl] = -1
            else:
                self.label_mapping[self.METAINFO["categories"][lbl]] = i
                self.label2cat[lbl] = i
                i += 1

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(
                f"The annotations loaded from annotation file "
                f"should be a dict, but got {type(annotations)}!"
            )
        if "data_list" not in annotations or "metainfo" not in annotations:
            raise ValueError("Annotation must have data_list and metainfo " "keys")
        metainfo = annotations["metainfo"]
        raw_data_list = annotations["data_list"]
        if self.max_elems != -1 and len(raw_data_list) > self.max_elems:
            raw_data_list = raw_data_list[: self.max_elems]

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError(
                            "data_info must be list of dict, but " f"got {type(item)}"
                        )
                data_list.extend(data_info)
            else:
                raise TypeError(
                    "data_info should be a dict or list of dict, "
                    f"but got {type(data_info)}"
                )

        _, world_size = get_dist_info()
        batch_total = self.batch_size * world_size
        modulo = len(data_list) % batch_total
        if modulo != 0:
            print("padding data_list to match batch_size * world_size")
            print(f"adding {batch_total - modulo} elements")
            data_list = data_list + [data_list[-1]] * (batch_total - modulo)

        return data_list

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is not None:
            ann_info = self._remove_dontcare(ann_info)

            if self.with_velocity:
                gt_bboxes_3d = ann_info["gt_bboxes_3d"]
                gt_velocities = ann_info["velocities"]
                nan_mask = np.isnan(gt_velocities[:, 0])
                gt_velocities[nan_mask] = [0.0, 0.0]
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocities], axis=-1)
                ann_info["gt_bboxes_3d"] = gt_bboxes_3d
        else:
            # empty instance
            ann_info = dict()
            if self.with_velocity:
                ann_info["gt_bboxes_3d"] = np.zeros((0, 9), dtype=np.float32)
            else:
                ann_info["gt_bboxes_3d"] = np.zeros((0, 7), dtype=np.float32)
            ann_info["gt_labels_3d"] = np.zeros(0, dtype=np.int64)

            if self.load_type in ["fov_image_based", "mv_image_based"]:
                ann_info["gt_bboxes"] = np.zeros((0, 4), dtype=np.float32)
                ann_info["gt_bboxes_labels"] = np.array(0, dtype=np.int64)
                ann_info["attr_labels"] = np.array(0, dtype=np.int64)
                ann_info["centers_2d"] = np.zeros((0, 2), dtype=np.float32)
                ann_info["depths"] = np.zeros((0), dtype=np.float32)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # TODO: Unify the coordinates
        if self.load_type in ["fov_image_based", "mv_image_based"]:
            gt_bboxes_3d = CameraInstance3DBoxes(
                ann_info["gt_bboxes_3d"],
                box_dim=ann_info["gt_bboxes_3d"].shape[-1],
                origin=(0.5, 0.5, 0.5),
            )
        else:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                ann_info["gt_bboxes_3d"],
                box_dim=ann_info["gt_bboxes_3d"].shape[-1],
                origin=(0.5, 0.5, 0.5),
            ).convert_to(self.box_mode_3d)

        ann_info["gt_bboxes_3d"] = gt_bboxes_3d

        return ann_info

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.load_type == "mv_image_based":
            data_list = []
            for idx, (cam_id, img_info) in enumerate(info["images"].items()):
                camera_info = dict()
                camera_info["images"] = dict()
                camera_info["images"][cam_id] = img_info
                if "cam_instances" in info and cam_id in info["cam_instances"]:
                    camera_info["instances"] = info["cam_instances"][cam_id]
                else:
                    camera_info["instances"] = []
                # TODO: check whether to change sample_idx for 6 cameras
                #  in one frame
                camera_info["sample_idx"] = info["sample_idx"] * 6 + idx
                camera_info["token"] = info["token"]
                camera_info["ego2global"] = info["ego2global"]

                if not self.test_mode:
                    # used in traing
                    camera_info["ann_info"] = self.parse_ann_info(camera_info)
                if self.test_mode and self.load_eval_anns:
                    camera_info["eval_ann_info"] = self.parse_ann_info(camera_info)
                data_list.append(camera_info)
            return data_list
        else:
            data_info = self.parent_parse_data_info(info)
            return data_info

    def parent_parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process the `instances` field to
        `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """

        if info["lidar2global"] is not None:
            ego_pose = np.array(info["lidar2global"])
            info["ego_pose"] = ego_pose
        if self.modality["use_lidar"]:
            info["num_pts_feats"] = info["lidar_points"]["num_pts_feats"]
            info["lidar_path"] = info["lidar_points"]["lidar_path"]
        if not self.test_mode:
            info["ann_info"] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info["eval_ann_info"] = self.parse_ann_info(info)

        return info
