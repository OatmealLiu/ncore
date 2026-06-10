<div align="center">

# TruckDrive: Long-Range Autonomous Highway Driving Dataset

**Torc Robotics · Princeton University · CVPR 2026**

[![Paper](https://img.shields.io/badge/Paper-PDF-2ea44f?style=flat-square)](https://arxiv.org/pdf/2603.02413)
[![arXiv](https://img.shields.io/badge/arXiv-2603.02413-b31b1b?style=flat-square)](https://arxiv.org/abs/2603.02413)
[![Project Page](https://img.shields.io/badge/Project-Page-f0ad4e?style=flat-square)](http://torc-ai.github.io/TruckDrive)
[![Data](https://img.shields.io/badge/Data-Access_Portal-6f42c1?style=flat-square)](https://d3ehgyu1hepsur.cloudfront.net/?prefix=)

**Filippo Ghilotti, Edoardo Palladin, Samuel Brucker, Adam Sigal, Mario Bijelic, Felix Heide**

</div>

<div align="center">

TruckDrive is a long-range autonomous highway driving dataset designed for heavy-truck safety, perception, prediction, and planning research. It targets high-speed highway operation, where reliable scene understanding hundreds of meters ahead is required for anticipatory planning and safe braking.

This repository hosts the **TruckDrive Devkit** together with release documentation.

</div>

## License

This devkit is released under the **Apache License, Version 2.0**.
See [`LICENSE.txt`](./LICENSE.txt) for the full license text.

This devkit is distributed independently of the TruckDrive Dataset. The dataset is governed by separate license terms. See the [Dataset Repository](http://torc-ai.github.io/TruckDrive) for details.

## Downloading the TruckDrive Dataset
To download the dataset use the provided [bash script](download_truckdrive.sh).
For faster downloads, install `aria2` first:
```bash
sudo apt update
sudo apt install -y aria2
```
Download the full dataset:
```bash
bash download_truckdrive.sh \
  --out /PATH/TO/TruckDrive \
  --all-scenes \
  --all-modalities 
```
By default, the script uses aria2c automatically if it is installed, with --jobs 4 and --aria2-connections 8, otherwise it defaults to curl (slower).
To download only one scene:
```bash
bash download_truckdrive.sh \
  --out /PATH/TO/TruckDrive \
  --scene scene_28_1 \
  --all-modalities 
```
To download only specific folders, replace --all-modalities with options such as:
--camera --lidar --radar --poses --calibration --annotations --accumulated-gt-depth

## Highlights

- **475k** synchronized multimodal samples
- **165k** densely annotated frames
- up to **1,000 m** for 2D benchmark annotations
- up to **400 m** for 3D benchmark annotations
- **7** long-range FMCW LiDARs, **3** short-range LiDARs, **10** 4D radars, and **11-15** cameras
- supports perception, tracking, depth estimation, prediction, planning, and end-to-end driving

## Dataset Overview

TruckDrive targets long-range, highway-scale autonomous driving for semi-trucks and other heavy commercial vehicles. It is designed to stress perception and planning systems beyond the short-range assumptions common in urban autonomous-driving benchmarks.

| Item | Paper-reported description |
|---|---|
| Domain | Long-range highway and commercial-vehicle driving |
| Platform | Semi-truck-mounted multimodal sensor suite |
| Synchronized samples |  **475k** |
| Densely annotated frames |  **165k** |
| Unlabeled synchronized samples |  **310k** |
| Sequences |  **3,828** sequences recorded over 2 years |
| Sequence duration |  **15-25 seconds** |
| Average ego trajectory |  **500 m** per sequence |
| 2D annotation range | Up to **1,000 m** |
| 3D annotation range | Up to **400 m** |
| Primary operating regime | Highway-speed, long-range truck autonomy |

## Dataset Viewer

This repository includes an interactive dataset viewer for inspecting TruckDrive scenes in 3D and 2D. The viewer supports synchronized LiDAR, radar, camera images, 3D bounding boxes, lane lines, accumulated depth overlays and video export.

For installation instructions, expected dataset layout, coordinate-frame conventions and usage examples, see the dataset-viewer documentation:

[dataset_viewer/README.md](dataset_viewer/README.md)

## Generate training/validation/testing jsons/pkls

This repository includes a script to generate train/val/test jsons/pkls with syncronized sensors and annotations

For instructions, see the related documentation:

[generate_training_data/README.md](generate_training_data/README.md)

## MMDetection3D Code

This repository includes code to run experiments in MMDetection3D with TruckDrive.

For instructions, see the related documentation:

[mmdet_project/README.md](mmdet_project/README.md)

## Dependencies

This devkit requires the following third-party packages, which must be installed separately:

- **PyQt5** — GNU Lesser General Public License v3 (LGPLv3)
  https://www.riverbankcomputing.com/software/pyqt/

- **MMDetection** — Apache License, Version 2.0
  https://github.com/open-mmlab/mmdetection

See [`THIRD-PARTY-NOTICES.txt`](./THIRD-PARTY-NOTICES.txt) for full attribution and license details.

## Citation
If you find our dataset useful in your work, please cite us!

```bibtex
@inproceedings{ghilotti2026truckdrive,
  title     = {TruckDrive: Long-Range Autonomous Highway Driving Dataset},
  author    = {Ghilotti, Filippo and Palladin, Edoardo and Brucker, Samuel and Sigal, Adam and Bijelic, Mario and Heide, Felix},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
