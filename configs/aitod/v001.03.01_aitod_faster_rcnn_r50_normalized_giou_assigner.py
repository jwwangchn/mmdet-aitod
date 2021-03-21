"""
Faster R-CNN with Normalized GIOU Assigner


"""

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_aitod.py',
    '../_base_/datasets/aitod_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='normalized_giou')),
            rcnn=dict(
                assigner=dict(
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='normalized_giou'))))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
