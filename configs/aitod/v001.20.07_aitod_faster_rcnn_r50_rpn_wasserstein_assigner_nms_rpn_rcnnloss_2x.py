"""


"""

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_aitod.py',
    '../_base_/datasets/aitod_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='WassersteinLoss', loss_weight=10.0))),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='wasserstein_nms', iou_threshold=0.85)),
        rpn=dict(
            assigner=dict(
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='wasserstein'))))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
