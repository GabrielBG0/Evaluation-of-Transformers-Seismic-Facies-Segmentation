# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001, paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.0)}))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=150)
checkpoint_config = dict()
evaluation = dict(interval=10, metric='mIoU', pre_eval=True, by_epoch=True, save_best='mIoU')