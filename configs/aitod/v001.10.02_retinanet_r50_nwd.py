"""
retinanet with normalized wasserstein


"""

_base_ = [
    '../_base_/models/retinanet_r50_fpn_aitod.py',
    '../_base_/datasets/aitod_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            iou_calculator=dict(type='BboxDistanceMetric'),
            assign_metric='wasserstein')),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='wasserstein_nms', iou_threshold=0.65),
        max_per_img=3000))


# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)