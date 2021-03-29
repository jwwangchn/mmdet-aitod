"""
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.179
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.472
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.120
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.065
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.196
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.276
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.332
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.304
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.325
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.330
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.119
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.346
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.380
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.409
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.823
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.290
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.405
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.478
# Class-specific LRP-Optimal Thresholds # 
 [0.624 0.924 0.917 0.943 0.812 0.745 0.821 0.936]
2021-03-27 18:02:16,482 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.201 | bridge        | 0.152 | storage-tank | 0.268 |
| ship     | 0.456 | swimming-pool | 0.117 | vehicle      | 0.188 |
| person   | 0.070 | wind-mill     | 0.068 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+


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
        rcnn=dict(
            assigner=dict(
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='wasserstein')),
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
