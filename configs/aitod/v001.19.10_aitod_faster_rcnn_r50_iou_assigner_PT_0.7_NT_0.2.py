"""
Faster R-CNN with Normalized Wasserstein Assigner



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
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.3)),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.3))))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
