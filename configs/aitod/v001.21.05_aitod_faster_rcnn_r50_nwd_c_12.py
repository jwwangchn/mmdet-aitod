"""
Faster R-CNN with Normalized Wasserstein Assigner

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.158
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.419
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.087
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.059
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.172
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.220
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.248
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.283
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.305
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.310
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.116
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.330
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.350
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.354
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.854
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.306
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.454
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.529
# Class-specific LRP-Optimal Thresholds # 
 [0.693 0.932 0.928 0.912 0.729 0.792 0.829 0.94 ]
2021-06-04 22:46:17,113 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.129 | bridge        | 0.138 | storage-tank | 0.249 |
| ship     | 0.375 | swimming-pool | 0.069 | vehicle      | 0.190 |
| person   | 0.067 | wind-mill     | 0.043 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-06-04 22:46:17,115 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.884 | bridge        | 0.869 | storage-tank | 0.782 |
| ship     | 0.653 | swimming-pool | 0.925 | vehicle      | 0.833 |
| person   | 0.932 | wind-mill     | 0.951 | None         | None  |
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
                    iou_calculator=dict(
                        type='BboxDistanceMetric',
                        constant=12.0),
                    assign_metric='wasserstein')),
            rcnn=dict(
                assigner=dict(
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='wasserstein'))))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
