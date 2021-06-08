"""


"""

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_dota.py',
    '../_base_/datasets/dota_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]


scale_constant = 20

model = dict(
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                iou_calculator=dict(type='BboxDistanceMetric', constant=scale_constant),
                assign_metric='wasserstein'))))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2)

fp16 = dict(loss_scale=512.)

optimizer = dict(type='SGD', lr=0.01*2.0/4.0, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=1)
evaluation = dict(interval=12, metric='bbox', proposal_nums=(300, 1000, 3000))