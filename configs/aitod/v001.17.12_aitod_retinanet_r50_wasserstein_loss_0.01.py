"""

"""


_base_ = [
    '../_base_/models/retinanet_r50_fpn_aitod.py',
    '../_base_/datasets/aitod_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        loss_bbox=dict(type='WassersteinLoss', loss_weight=20.0)))

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)