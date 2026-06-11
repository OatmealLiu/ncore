import os
import mmcv
import copy
import cv2
import numpy as np
import torch

from mmcv.transforms.base import BaseTransform
from mmdet3d.structures.points import get_points_type
from mmdet3d.registry import TRANSFORMS
from PIL import Image


@TRANSFORMS.register_module()
class LoadMultiViewImageTruckDrive(BaseTransform):
    """Load multi channel images from a list of separate channel files.

    Expects results['image_paths'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_root,
        cameras_to_load=[
            ["forward_center_medium"],
            ["sideward_left_front_wide"],
            ["sideward_right_front_wide"],
            ["rearward_left_bottom_medium"],
            ["rearward_right_bottom_medium"],
        ],
        to_float32=True,
    ):
        self.cameras_to_load = cameras_to_load
        self.to_float32 = to_float32
        self.data_root = data_root

    def transform(self, results):
        filename_list, camname_list = self._get_filename_list(results)
        imgs, img_shapes, pad_shape = self._process_images(results, filename_list)
        results.update(
            {
                "filename": filename_list,
                "img": [imgs[..., i] for i in range(imgs.shape[-1])],
                "img_shape": img_shapes,
                "ori_shape": imgs.shape[:2],
                "pad_shape": pad_shape,
                "scale_factor": 1.0,
                "num_view": len(filename_list),
            }
        )
        self._process_lidar_and_camera_transforms(results, camname_list)
        return results

    def _get_filename_list(self, results):
        """Get filename list from results."""
        filename_list = []
        camname_list = []
        for cam_name in self.cameras_to_load:
            done = False
            for cn in cam_name:
                if cn in results["images"]:
                    filename_list.append(results["images"][cn]["img_path"])
                    camname_list.append(cn)
                    done = True
                    break
            if not done:
                # print(f"Warning: cameras {cam_name} not found in {results['images'].keys()}")
                filename_list.append(None)
                camname_list.append(list(results["images"].keys())[0])
        return filename_list, camname_list

    def _process_images(self, results, img_filename_list):
        imgs = []
        for name in img_filename_list:
            if name is not None:
                img = np.array(cv2.imread(os.path.join(self.data_root, name)))
                if img is None or np.array(img.shape).shape != (3,):
                    print(
                        f"\033[33mWarning: image {name} is corrupted. Using black image instead.\033[0m"
                    )
                    img = np.zeros((3, 3, 3), dtype=np.float32)
            else:
                print("missing cameras file: padding")
                img = np.zeros((3, 3, 3), dtype=np.float32)
            imgs.append(img)

        # handle the image with different shape
        img_shapes = np.stack([np.array(img.shape) for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        pad_shape = (
            img_shape_max[:2] if not np.all(img_shape_max == img_shape_min) else None
        )
        if pad_shape is not None:
            imgs = [mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs]
        imgs_array = np.stack(imgs, axis=-1)
        if self.to_float32:
            imgs_array = imgs_array.astype(np.float32)
        return imgs_array, img_shapes, pad_shape

    def _process_lidar_and_camera_transforms(self, results, camname_list):
        """Process lidar and camera transforms."""
        cam2img_list, lidar2cam_list, cam2lidar_list, lidar2img_list = [], [], [], []
        velo2cam_list, cam2velo_list, velo2img_list = [], [], []
        img_aug_matrix_list = []
        for cam_name in camname_list:
            cam_item = results["images"][cam_name]
            lidar2cam_list.append(np.asarray(cam_item["lidar2cam"]).astype(np.float32))
            velo2cam_list.append(np.asarray(cam_item["velo2cam"]).astype(np.float32))
            lidar2cam_array = np.asarray(cam_item["lidar2cam"]).astype(np.float32)
            velo2cam_array = np.asarray(cam_item["velo2cam"]).astype(np.float32)
            camera2lidar = np.linalg.inv(
                np.asarray(cam_item["lidar2cam"]).astype(np.float32)
            )
            cam2lidar_list.append(camera2lidar)
            camera2velo = np.linalg.inv(
                np.asarray(cam_item["velo2cam"]).astype(np.float32)
            )
            cam2velo_list.append(camera2velo)
            cam2img_array = np.array(cam_item["cam2img"]).astype(np.float32)
            cam2img_list.append(cam2img_array)
            lidar2img_list.append(cam2img_array @ lidar2cam_array)
            velo2img_list.append(cam2img_array @ velo2cam_array)
            img_aug_matrix_list.append(
                np.eye(4, dtype=np.float32)
            )  # no augmentation on image

        results.update(
            {
                "img_path": results["filename"],
                "cam2img": np.stack(cam2img_list, axis=0),
                "lidar2cam": np.stack(lidar2cam_list, axis=0),
                "cam2lidar": np.stack(cam2lidar_list, axis=0),
                "camera2ego": np.stack(cam2lidar_list, axis=0),
                "camera2lidar": np.stack(cam2lidar_list, axis=0),
                "velo2cam": np.stack(velo2cam_list, axis=0),
                "cam2velo": np.stack(cam2velo_list, axis=0),
                "lidar2img": np.stack(lidar2img_list, axis=0),
                "lidar2image": np.stack(lidar2img_list, axis=0),
                "velo2img": np.stack(velo2img_list, axis=0),
                "ori_cam2img": copy.deepcopy(results.get("cam2img", [])),
                "camera_intrinsics": np.stack(cam2img_list, axis=0),
                "img_aug_matrix": np.stack(img_aug_matrix_list, axis=0),
            }
        )

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(cameras_to_load={self.cameras_to_load})"
        return repr_str


@TRANSFORMS.register_module()
class LoadAEVAPointsFromBinTruckDrive(BaseTransform):

    def __init__(
        self,
        data_root,
        load_dim=6,
        use_dim=[0, 1, 2],
        time_dim=None,
    ):
        self.data_root = data_root
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.time_dim = time_dim
        self.coord_type = "LIDAR"

    def _load_points(self, pts_filename: str):
        if pts_filename is not None:
            scan = np.fromfile(os.path.join(self.data_root, pts_filename)).astype(
                np.float32
            )
            scan = scan.reshape((-1, self.load_dim))
            if scan.shape[0] < 20:
                to_pad = 20 - scan.shape[0]
                scan = np.concatenate(
                    [scan, np.zeros((to_pad, scan.shape[1]), dtype=scan.dtype)], axis=0
                )
                scan[-to_pad:, :2] = np.random.uniform(
                    -50.0, 50.0, size=(to_pad, 2)
                ).astype(np.float32)
                scan[-to_pad:, 2:3] = np.random.uniform(
                    -5.0, 5.0, size=(to_pad, 1)
                ).astype(np.float32)
        else:
            scan = np.zeros((20, self.load_dim), dtype=np.float32)
            scan[:, :2] = np.random.uniform(
                -50.0, 50.0, size=(scan.shape[0], 2)
            ).astype(np.float32)
            scan[:, 2:3] = np.random.uniform(-5.0, 5.0, size=(scan.shape[0], 1)).astype(
                np.float32
            )
        return scan.reshape((-1, self.load_dim))

    def transform(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        points = self._load_points(lidar_path)
        points = points[:, self.use_dim]
        if self.time_dim:
            points[:, self.time_dim] *= 1e-6

        points_class = get_points_type(self.coord_type)
        points = np.clip(points, a_min=None, a_max=60000)
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        results["points"] = points
        results["lidar_aug_matrix"] = np.eye(
            4, dtype=np.float32
        )  # no augmentation on lidar
        results["lidar2ego"] = np.eye(4, dtype=np.float32)  # no augmentation on lidar

        return results


@TRANSFORMS.register_module()
class FromAEVAtoVelodyneTruckDrive(BaseTransform):

    def __init__(self):
        pass

    def transform(self, results):
        lidar2velo = np.array(results["lidar_points"]["lidar2velo"])

        if results["lidar2global"] is not None:
            results["lidar2global"] = results["lidar2global"] @ np.linalg.inv(
                lidar2velo
            )
        if results["ego_pose"] is not None:
            results["ego_pose"] = results["ego_pose"] @ np.linalg.inv(lidar2velo)

        results["points"].tensor[:, :3] = (
            torch.cat(
                [
                    results["points"].tensor[:, :3],
                    torch.ones_like(results["points"].tensor[:, :1]),
                ],
                dim=-1,
            )
            @ lidar2velo.T
        )[:, :3]

        if "num_view" in results:
            results.update(
                {
                    "camera2lidar": results["cam2velo"],
                    "camera2ego": results["cam2velo"],
                    "lidar2camera": results["velo2cam"],
                    "lidar2image": results["velo2img"],
                    "lidar2cam": results["velo2cam"],
                    "cam2lidar": results["cam2velo"],
                    "lidar2img": results["velo2img"],
                }
            )

        return results


@TRANSFORMS.register_module()
class LoadOustersPointsFromBinTruckDrive(BaseTransform):

    def __init__(
        self,
        data_root,
        load_dim=6,
        use_dim=[0, 1, 2],
        pad_vel_dim=3,
        time_dim=None,
    ):
        self.data_root = data_root
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.coord_type = "LIDAR"
        self.pad_vel_dim = pad_vel_dim
        self.time_dim = time_dim

    def _load_points(self, pts_filename: str):
        if pts_filename is not None:
            scan = np.fromfile(
                os.path.join(self.data_root, pts_filename), dtype=np.float32
            )
            if scan.shape[0] < 20:
                to_pad = 20 - scan.shape[0]
                scan = np.concatenate(
                    [
                        scan,
                        np.zeros((to_pad, scan.shape[1]), dtype=scan.dtype),
                    ],
                    axis=0,
                )
                scan[-to_pad:, :2] = np.random.uniform(
                    -50.0, 50.0, size=(to_pad, 2)
                ).astype(np.float32)
                scan[-to_pad:, 2:3] = np.random.uniform(
                    -5.0, 5.0, size=(to_pad, 1)
                ).astype(np.float32)
        else:
            # print('missing ouster file: padding')
            scan = np.zeros((20, self.load_dim), dtype=np.float32)
        return scan.reshape((-1, self.load_dim))

    def transform(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        if results["short_range_lidar_points"] is not None:
            for ou in results["short_range_lidar_points"].keys():
                lidar_path = results["short_range_lidar_points"][ou][
                    "short_range_lidar_path"
                ]
                if lidar_path is None:
                    continue
                points = self._load_points(lidar_path)
                points = points[:, self.use_dim]
                points_class = get_points_type(self.coord_type)
                # pad velocity to match aeva points
                points[:, self.pad_vel_dim] = 0
                # add delta sync respect to aeva points
                if self.time_dim:
                    delta_t_sync = (
                        results["short_range_lidar_points"][ou]["timestamp"]
                        - results["lidar_points"]["timestamp"]
                    )
                    points[:, self.time_dim] += delta_t_sync
                    points[:, self.time_dim] *= 1e-6
                # transpose points from ouster frame to aeva frame
                ouster2lidar = np.array(
                    results["short_range_lidar_points"][ou]["ouster2lidar"]
                )
                points[:, :3] = (
                    np.concatenate(
                        [points[:, :3], np.ones((len(points), 1), dtype=np.float32)],
                        axis=-1,
                    )
                    @ ouster2lidar.T
                )[:, :3]
                points = np.clip(points, a_min=None, a_max=60000)
                points = points_class(
                    points, points_dim=points.shape[-1], attribute_dims=None
                )
                results["points"].tensor = torch.cat(
                    [results["points"].tensor, points.tensor], dim=0
                )

        return results


@TRANSFORMS.register_module()
class ScaleMultiViewImageTruckDrive(BaseTransform):

    def __init__(self, scales=(800, 448)):
        self.scales = np.array(scales)
        self.scales[0] = self.scales[0] + self.scales[1]
        self.scales[1] = self.scales[0] - self.scales[1]
        self.scales[0] = self.scales[0] - self.scales[1]

    def transform(self, results):
        img_shape = (
            results["img_shape"]
            if len(results["img_shape"][0]) == 1
            else results["img_shape"][0]
        )
        rand_scale = self.scales / np.array(img_shape[:2])
        y_size = int(img_shape[0] * rand_scale[0])
        x_size = int(img_shape[1] * rand_scale[1])
        scale_factor = np.eye(4, dtype=np.float32)
        scale_factor[0, 0] *= rand_scale[1]
        scale_factor[1, 1] *= rand_scale[0]
        results["img"] = [
            mmcv.imresize(img, (self.scales[1], self.scales[0]), return_scale=False)
            for img in results["img"]
        ]
        lidar2img = [scale_factor @ l2i for l2i in results["lidar2img"]]
        results["lidar2img"] = np.stack(lidar2img)
        results["lidar2image"] = results["lidar2img"]
        cam2img = [scale_factor @ l2i for l2i in results["cam2img"]]
        results["cam2img"] = np.stack(cam2img)

        results["img_shape"] = [img.shape for img in results["img"]]

        results["ori_cam2img"] = copy.deepcopy(results["cam2img"])
        results["camera_intrinsics"] = copy.deepcopy(results["cam2img"])
        velo2img = [scale_factor @ l2i for l2i in results["velo2img"]]
        results["velo2img"] = np.stack(velo2img)

        results["ori_shape"] = results["img_shape"][0][:2]
        results["pad_shape"] = results["img_shape"]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.scales}, "
        return repr_str


@TRANSFORMS.register_module()
class ImagesNumpyToPIL(BaseTransform):

    def __init__(self):
        pass

    def transform(self, results):
        results["img"] = [
            Image.fromarray(img.astype(np.uint8)) for img in results["img"]
        ]
        results["img_shape"] = results["img"][0].size
        results["ori_shape"] = results["img"][0].size
        results["pad_shape"] = results["img"][0].size
        return results
