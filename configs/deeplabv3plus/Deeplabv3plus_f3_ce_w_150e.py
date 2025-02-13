_base_ = [
    '../_base_/models/deeplabv3plus.py', '../_base_/datasets/f3.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_150e.py'
]

#checkpoint = 'pretrain/resnetv1c101_8xb32_in1k_20220214-434fe45f.pth'

class_weight = [0.0456399, 0.1064931, 0.02634881, 0.1825596, 0.4259724, 0.2129862]

loss_balanced_h=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weight)

loss_balanced_aux_h=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=class_weight)

model = dict(
    decode_head=dict(
        num_classes=6, 
        loss_decode=loss_balanced_h), 
    auxiliary_head=dict(
        num_classes=6, 
        loss_decode=loss_balanced_aux_h))

# change work dir
work_dir = './'