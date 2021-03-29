"""
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.190
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.486
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.132
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.028
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.195
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.288
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.382
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.296
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.314
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.317
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.030
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.337
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.397
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.450
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.819
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.286
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.362
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.505
# Class-specific LRP-Optimal Thresholds # 
 [0.753 0.767 0.705 0.79  0.738 0.674 0.615 0.806]
2021-03-28 11:36:05,409 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.289 | bridge        | 0.138 | storage-tank | 0.319 |
| ship     | 0.344 | swimming-pool | 0.147 | vehicle      | 0.241 |
| person   | 0.090 | wind-mill     | 0.049 | None         | None  |
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
        rcnn=dict(
            assigner=dict(
                gpu_assign_thr=1024)),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='wasserstein_nms', iou_threshold=0.85)),
        rpn=dict(
            assigner=dict(
                gpu_assign_thr=1024, 
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
