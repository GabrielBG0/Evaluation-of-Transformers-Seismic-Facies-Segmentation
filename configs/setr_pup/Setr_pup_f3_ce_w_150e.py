_base_ = [
    '../_base_/models/setr_pup.py', '../_base_/datasets/f3.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_150e.py'
]


num_classes=6
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

norm_cfg = dict(type='SyncBN', requires_grad=True)


class_weight = [0.0456399, 0.1064931, 0.02634881, 0.1825596, 0.4259724, 0.2129862]

loss_balanced_h=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight)

loss_balanced_aux_h=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=class_weight)


model = dict(
    pretrained=None,
    backbone=dict(
        img_size=(512, 512),
        drop_rate=0.0),
    decode_head=dict(num_classes=num_classes, loss_decode=loss_balanced_h,),

    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=num_classes,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=loss_balanced_aux_h),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=1,
            num_classes=num_classes,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=loss_balanced_h),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=2,
            num_classes=num_classes,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=loss_balanced_h),
    ],
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))


work_dir = '/parceirosbr/asml/gabriel.gutierrez/segmentation_unicamp/experiments/f3/traning/setr_pup/ce/150/'