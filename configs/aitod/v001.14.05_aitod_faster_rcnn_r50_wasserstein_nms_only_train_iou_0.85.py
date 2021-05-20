"""
Faster R-CNN with Wasserstein NMS (only train) iou_threshold = 0.85



"""

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_aitod.py',
    '../_base_/datasets/aitod_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    train_cfg=dict(
            rpn_proposal=dict(
                nms_pre=3000,
                max_per_img=3000,
                nms=dict(type='wasserstein_nms', iou_threshold=0.85))))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
