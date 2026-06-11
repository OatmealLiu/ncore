```shell
mkdir accumulated_gt_depth annotations calibrations camera lidar poses radar

mv \
  forward_center_medium \
  forward_left_narrow \
  forward_right_narrow \
  rearward_left_bottom_medium \
  rearward_right_bottom_medium \
  sideward_left_back_wide \
  sideward_left_front_wide \
  sideward_right_back_wide \
  sideward_right_front_wide \
  accumulated_gt_depth/


mv bounding_boxes lane_lines annotations/

mv calib_*.json calibrations/

mv leopard camera/

mv aeva ouster lidar/

mv gt_trajectory.txt poses/

mv conti542 radar/
```