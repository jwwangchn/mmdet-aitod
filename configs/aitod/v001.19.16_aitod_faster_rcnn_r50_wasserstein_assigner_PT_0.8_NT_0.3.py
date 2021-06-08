"""
Faster R-CNN with Normalized Wasserstein Assigner


Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.103
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.297
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.044
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.013
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.119
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.157
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.180
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.223
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.244
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.250
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.012
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.277
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.320
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.291
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.899
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.319
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.572
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.614
# Class-specific LRP-Optimal Thresholds # 
 [0.488 0.764 0.946 0.956 0.076 0.913 0.904 0.216]
2021-06-05 12:29:56,032 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.083 | bridge        | 0.073 | storage-tank | 0.241 |
| ship     | 0.166 | swimming-pool | 0.038 | vehicle      | 0.155 |
| person   | 0.047 | wind-mill     | 0.022 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-06-05 12:29:56,033 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.918 | bridge        | 0.930 | storage-tank | 0.788 |
| ship     | 0.844 | swimming-pool | 0.940 | vehicle      | 0.856 |
| person   | 0.949 | wind-mill     | 0.971 | None         | None  |
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
                    pos_iou_thr=0.8,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='wasserstein')),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.8,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='wasserstein'))))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
