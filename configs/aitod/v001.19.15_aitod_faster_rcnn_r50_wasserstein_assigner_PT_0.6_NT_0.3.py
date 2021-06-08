"""
Faster R-CNN with Normalized Wasserstein Assigner
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.141
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.385
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.074
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.054
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.154
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.195
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.229
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.287
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.314
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.320
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.111
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.342
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.371
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.353
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.868
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.307
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.499
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.543
# Class-specific LRP-Optimal Thresholds # 
 [0.812 0.968 0.968 0.964 0.755 0.88  0.917 0.99 ]
2021-06-05 11:10:19,963 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.136 | bridge        | 0.112 | storage-tank | 0.233 |
| ship     | 0.300 | swimming-pool | 0.077 | vehicle      | 0.172 |
| person   | 0.063 | wind-mill     | 0.037 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-06-05 11:10:19,987 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.875 | bridge        | 0.898 | storage-tank | 0.788 |
| ship     | 0.720 | swimming-pool | 0.919 | vehicle      | 0.842 |
| person   | 0.936 | wind-mill     | 0.961 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

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
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='wasserstein')),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='wasserstein'))))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
