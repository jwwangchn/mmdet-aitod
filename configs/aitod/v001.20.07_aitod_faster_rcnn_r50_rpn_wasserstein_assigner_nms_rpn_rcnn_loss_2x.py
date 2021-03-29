"""

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.192
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.485
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.138
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.041
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.194
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.291
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.374
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.299
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.316
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.320
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.042
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.342
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.396
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.445
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.820
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.286
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.367
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.503
# Class-specific LRP-Optimal Thresholds # 
 [0.678 0.832 0.689 0.807 0.636 0.667 0.632 0.736]
2021-03-27 19:02:20,282 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.302 | bridge        | 0.138 | storage-tank | 0.318 |
| ship     | 0.339 | swimming-pool | 0.148 | vehicle      | 0.242 |
| person   | 0.089 | wind-mill     | 0.048 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

"""

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_aitod.py',
    '../_base_/datasets/aitod_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    rpn_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='WassersteinLoss', loss_weight=10.0)),
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

fp16 = dict(loss_scale=512.)

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
