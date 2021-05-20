"""
retinanet with normalized wasserstein

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.092
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.249
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.050
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.032
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.100
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.131
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.169
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.181
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.200
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.206
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.073
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.222
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.261
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.281
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.914
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.326
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.658
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.687
# Class-specific LRP-Optimal Thresholds # 
 [0.053 0.282 0.312 0.331 0.065 0.351 0.308 0.078]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.000 | bridge        | 0.069 | storage-tank | 0.178 |
| ship     | 0.286 | swimming-pool | 0.001 | vehicle      | 0.161 |
| person   | 0.043 | wind-mill     | 0.001 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.999 | bridge        | 0.929 | storage-tank | 0.844 |
| ship     | 0.733 | swimming-pool | 0.997 | vehicle      | 0.855 |
| person   | 0.954 | wind-mill     | 0.997 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

"""

_base_ = [
    '../_base_/models/retinanet_r50_fpn_aitod.py',
    '../_base_/datasets/aitod_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            iou_calculator=dict(type='BboxDistanceMetric'),
            assign_metric='wasserstein')))


# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)