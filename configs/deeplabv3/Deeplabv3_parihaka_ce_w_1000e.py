_base_ = [
    '../_base_/models/deeplabv3.py', '../_base_/datasets/parihaka.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1000e.py'
]

class_weight = [0.0456399, 0.1064931, 0.02634881, 0.1825596, 0.4259724, 0.2129862]

loss_balanced_h=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)

loss_balanced_aux_h=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)

model = dict(
    decode_head=dict(
        num_classes=6, 
        loss_decode=loss_balanced_h), 
    auxiliary_head=dict(
        num_classes=6, 
        loss_decode=loss_balanced_aux_h), 
    )

# change work dir
work_dir = './'