_base_ = [
    '../_base_/models/segformer.py', '../_base_/datasets/parihaka.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_150e.py'
]

class_weight = [0.0456399, 0.1064931, 0.02634881, 0.1825596, 0.4259724, 0.2129862]

loss_balanced_h=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)

model =  dict(
    backbone=dict(
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        num_classes=6,
        in_channels=[64, 128, 320, 512], 
        loss_decode=loss_balanced_h))


# change work dir
work_dir = './'