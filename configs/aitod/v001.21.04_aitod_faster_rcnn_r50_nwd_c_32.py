"""
Faster R-CNN with Normalized Wasserstein Assigner

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.151
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.401
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.084
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.068
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.159
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.214
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.254
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.276
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.299
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.304
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.152
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.310
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.351
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.358
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.859
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.305
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.484
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.538
# Class-specific LRP-Optimal Thresholds # 
 [0.603 0.933 0.915 0.927 0.729 0.821 0.863 0.94 ]
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.136 | bridge        | 0.133 | storage-tank | 0.260 |
| ship     | 0.346 | swimming-pool | 0.056 | vehicle      | 0.185 |
| person   | 0.064 | wind-mill     | 0.032 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.868 | bridge        | 0.871 | storage-tank | 0.789 |
| ship     | 0.673 | swimming-pool | 0.938 | vehicle      | 0.837 |
| person   | 0.936 | wind-mill     | 0.963 | None         | None  |
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
                        constant=32.0),
                    assign_metric='wasserstein')),
            rcnn=dict(
                assigner=dict(
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='wasserstein'))))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
