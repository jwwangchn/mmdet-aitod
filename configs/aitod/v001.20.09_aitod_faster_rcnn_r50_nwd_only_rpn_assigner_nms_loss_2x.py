"""

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.196
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.495
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.141
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.033
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.199
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.292
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.375
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.300
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.319
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.322
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.034
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.341
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.398
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.443
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.815
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.285
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.360
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.500
# Class-specific LRP-Optimal Thresholds # 
 [0.733 0.801 0.7   0.797 0.752 0.665 0.573 0.758]
2021-03-28 07:48:42,720 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.306 | bridge        | 0.142 | storage-tank | 0.322 |
| ship     | 0.368 | swimming-pool | 0.143 | vehicle      | 0.246 |
| person   | 0.090 | wind-mill     | 0.048 | None         | None  |
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
    # roi_head=dict(
    #     bbox_head=dict(
    #         reg_decoded_bbox=True,
    #         loss_bbox=dict(type='WassersteinLoss', loss_weight=10.0))),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                gpu_assign_thr=512)),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='wasserstein_nms', iou_threshold=0.85)),
        rpn=dict(
            assigner=dict(
                gpu_assign_thr=512, 
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
