_base_ = [
    '../_base_/models/segmenter.py', '../_base_/datasets/f3.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1000e.py'
]

class_weight = [0.0456399, 0.1064931, 0.02634881, 0.1825596, 0.4259724, 0.2129862]
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
loss_balanced_h=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight)

model = dict(
    backbone=dict(
        type='VisionTransformer',
        img_size=crop_size,
        embed_dims=1024,
        num_layers=24,
        num_heads=16),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=1024,
        channels=1024,
        num_heads=16,
        embed_dims=1024,
        num_classes=6,
        loss_decode=loss_balanced_h),

    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

# change work dir
work_dir = './'