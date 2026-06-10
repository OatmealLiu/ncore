_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_truckdrive-3d_fullrange.py'
]
point_cloud_range = _base_.point_cloud_range
# xbound=[-54.0, 54.0, 0.3],
# ybound=[-54.0, 54.0, 0.3],
# zbound=[-10.0, 10.0, 20.0],
# dbound=[1.0, 60.0, 0.5],
xbound=[-160.0, 272.0, 0.6]
ybound=[-108.0, 108.0, 0.6]
zbound=[-20.0, 20.0, 40.0]
dbound=[1.0, 300.0, 2.5]

input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None
image_size=[256, 704]
image_size_reverse=[704, 256]

model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=False),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa: E501
        )),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=xbound,
        ybound=ybound,
        zbound=zbound,
        dbound=dbound,
        downsample=2),
    fusion_layer=dict(
        type='ConvFuser', in_channels=[80, 256], out_channels=256))

train_pipeline = [
    dict(
        type='LoadMultiViewImageTruckDrive',
        data_root=_base_.data_root,
        cameras_to_load=[
            ["forward_center_medium"],
            ["sideward_left_front_wide"],
            ["sideward_right_front_wide"],
            ["rearward_left_bottom_medium"],
            ["rearward_right_bottom_medium"],
        ],
        to_float32=True),
    dict(
        type='ScaleMultiViewImageTruckDrive',
        scales=image_size_reverse),
    dict(
        type='LoadAEVAPointsFromBinTruckDrive',
        data_root=_base_.data_root,
        load_dim=_base_.load_dim,
        use_dim=_base_.use_dim,
        time_dim=_base_.time_dim),
    dict(
        type='LoadOustersPointsFromBinTruckDrive',
        data_root=_base_.data_root,
        load_dim=_base_.short_range_load_dim,
        use_dim=_base_.short_range_use_dim,
        pad_vel_dim=_base_.short_range_pad_vel_dim,
        time_dim=_base_.short_range_time_dim),
    dict(
        type='FromAEVAtoVelodyneTruckDrive'),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=_base_.point_cloud_range),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=_base_.class_names),
    # Actually, 'GridMask' is not used here
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=6,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix'
        ])
]

test_pipeline = [
    dict(
        type='LoadMultiViewImageTruckDrive',
        data_root=_base_.data_root,
        cameras_to_load=[
            ["forward_center_medium"],
            ["sideward_left_front_wide"],
            ["sideward_right_front_wide"],
            ["rearward_left_bottom_medium"],
            ["rearward_right_bottom_medium"],
        ],
        to_float32=True),
    dict(
        type='ScaleMultiViewImageTruckDrive',
        scales=image_size_reverse),
    dict(
        type='LoadAEVAPointsFromBinTruckDrive',
        data_root=_base_.data_root,
        load_dim=_base_.load_dim,
        use_dim=_base_.use_dim,
        time_dim=_base_.time_dim),
    dict(
        type='LoadOustersPointsFromBinTruckDrive',
        data_root=_base_.data_root,
        load_dim=_base_.short_range_load_dim,
        use_dim=_base_.short_range_use_dim,
        pad_vel_dim=_base_.short_range_pad_vel_dim,
        time_dim=_base_.short_range_time_dim),
    dict(
        type='FromAEVAtoVelodyneTruckDrive'),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'num_views'
        ])
]

max_elems=-1
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        batch_size=2, 
        pipeline=train_pipeline,
        modality=input_modality))
val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        modality=input_modality))
test_dataloader = val_dataloader

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=6,
        end=6,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 1 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=2.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=2.4,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()

lr = 1e-4 # 0.0002 = 2 * 1e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(type='CheckpointHook', interval=1))

load_from = 'work_dirs/bevfusion_truckdrive_fullrange/epoch_20.pth'
del _base_.custom_hooks
