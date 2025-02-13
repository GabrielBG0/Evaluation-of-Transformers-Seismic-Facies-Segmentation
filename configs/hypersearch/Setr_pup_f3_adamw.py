_base_ = [
    '../_base_/models/setr_pup.py', '../_base_/datasets/f3.py',
    '../_base_/default_runtime.py',
]


num_classes=6
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

norm_cfg = dict(type='SyncBN', requires_grad=True)

# optimizer
optimizer = dict(type='AdamW', lr=4.9647542094923015e-05, weight_decay=0.002025122106863325, paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.0)}))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1000)
checkpoint_config = dict()
evaluation = dict(interval=10, metric='mIoU', pre_eval=True, by_epoch=True, save_best='mIoU')

class_weight = [0.0456399, 0.1064931, 0.02634881, 0.1825596, 0.4259724, 0.2129862]

loss_balanced_h=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight)

loss_balanced_aux_h=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.3, class_weight=class_weight)


backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint = 'pretrain/vit_large_p16.pth'
model = dict(
    pretrained=None,
    backbone=dict(
        img_size=(512, 512),
        drop_rate=0.01697168888835995),
    decode_head=dict(num_classes=num_classes, loss_decode=loss_balanced_h, dropout_ratio=0.05485357250995974),

    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=num_classes,
            dropout_ratio=0.05485357250995974,
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
            dropout_ratio=0.05485357250995974,
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
            dropout_ratio=0.05485357250995974,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=3,
            align_corners=False,
            loss_decode=loss_balanced_h),
    ],
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

# change work dir
work_dir = './'