"""

"""

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_tinycoco.py',
    '../_base_/datasets/tinycoco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

fp16 = dict(loss_scale=512.)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
evaluation = dict(interval=12, metric='bbox')