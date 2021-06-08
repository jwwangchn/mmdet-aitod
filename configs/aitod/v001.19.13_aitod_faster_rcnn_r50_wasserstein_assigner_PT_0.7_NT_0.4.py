"""
Faster R-CNN with Normalized Wasserstein Assigner


Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.114
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.301
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.061
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.006
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.117
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.192
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.209
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.187
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.201
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.203
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.005
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.182
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.351
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.334
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.893
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.315
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.527
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.675
# Class-specific LRP-Optimal Thresholds # 
 [0.69  0.646 0.87  0.855 0.152 0.832 0.812 0.053]
2021-06-05 02:53:24,228 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.115 | bridge        | 0.038 | storage-tank | 0.263 |
| ship     | 0.217 | swimming-pool | 0.047 | vehicle      | 0.177 |
| person   | 0.051 | wind-mill     | 0.001 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-06-05 02:53:24,229 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.899 | bridge        | 0.960 | storage-tank | 0.764 |
| ship     | 0.797 | swimming-pool | 0.948 | vehicle      | 0.839 |
| person   | 0.943 | wind-mill     | 0.996 | None         | None  |
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
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='wasserstein')),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='wasserstein'))))

fp16 = dict(loss_scale=512.)


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
