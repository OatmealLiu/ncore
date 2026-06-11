
# MMDetection3D Dataloader and Config

The following folder contains the dataloader, utils and config for [MMDetection3d](https://github.com/open-mmlab/mmdetection3d)


## Environment

We use the following versions:
- mmcv: 2.1.0
- mmengine: 0.9.1
- mmdet: 3.3.0
- mmdet3d: 1.4.0

We follow https://mmdetection3d.readthedocs.io/en/v1.4.0/get_started.html to setup the environment


## Development

- Copy the entire folder [mmdet_project/TruckDrive](mmdet_project/TruckDrive) in `./mmdetection3d/projects/`
- Setup the MMDetection3D Repo and environment
- To train the short range model LiDAR only run: `bash tools/dist_train.sh projects/TruckDrive/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_truckdrive-3d.py 8`
