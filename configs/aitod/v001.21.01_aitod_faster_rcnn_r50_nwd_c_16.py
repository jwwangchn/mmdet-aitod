"""
Faster R-CNN with Normalized Wasserstein Assigner

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.154
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.410
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.087
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.068
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.171
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.209
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.220
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.281
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.305
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.311
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.137
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.327
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.356
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.335
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.858
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.308
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.479
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.528
# Class-specific LRP-Optimal Thresholds # 
 [0.662 0.905 0.922 0.916 0.723 0.784 0.788 0.891]
 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.115 | bridge        | 0.133 | storage-tank | 0.256 |
| ship     | 0.370 | swimming-pool | 0.059 | vehicle      | 0.188 |
| person   | 0.067 | wind-mill     | 0.048 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.897 | bridge        | 0.872 | storage-tank | 0.779 |
| ship     | 0.657 | swimming-pool | 0.940 | vehicle      | 0.839 |
| person   | 0.934 | wind-mill     | 0.950 | None         | None  |
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
                        constant=16.0),
                    assign_metric='wasserstein')),
            rcnn=dict(
                assigner=dict(
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='wasserstein'))))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
