"""
Faster R-CNN with Normalized Wasserstein Assigner

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.137
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.363
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.077
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.022
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.134
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.226
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.262
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.246
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.261
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.262
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.027
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.260
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.369
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.376
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.871
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.314
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.496
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.558
# Class-specific LRP-Optimal Thresholds # 
 [0.674 0.809 0.852 0.836 0.553 0.794 0.758 0.871]
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.162 | bridge        | 0.039 | storage-tank | 0.269 |
| ship     | 0.281 | swimming-pool | 0.070 | vehicle      | 0.186 |
| person   | 0.060 | wind-mill     | 0.030 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.845 | bridge        | 0.960 | storage-tank | 0.763 |
| ship     | 0.744 | swimming-pool | 0.919 | vehicle      | 0.829 |
| person   | 0.935 | wind-mill     | 0.972 | None         | None  |
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
                        constant=8.0),
                    assign_metric='wasserstein')),
            rcnn=dict(
                assigner=dict(
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='wasserstein'))))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
