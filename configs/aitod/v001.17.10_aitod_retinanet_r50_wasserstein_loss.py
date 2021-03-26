"""
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.022
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.092
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.008
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.006
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.027
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.043
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.115
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.084
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.107
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.123
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.041
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.142
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.126
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.187
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.971
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.356
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.831
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.814
# Class-specific LRP-Optimal Thresholds # 
 [0.05  0.263 0.3   0.393   nan 0.403 0.351 0.05 ]
2021-03-24 22:36:50,299 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.000 | bridge        | 0.004 | storage-tank | 0.025 |
| ship     | 0.111 | swimming-pool | 0.000 | vehicle      | 0.058 |
| person   | 0.010 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
"""


_base_ = [
    '../_base_/models/retinanet_r50_fpn_aitod.py',
    '../_base_/datasets/aitod_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        loss_bbox=dict(type='WassersteinLoss', loss_weight=2.0)))

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)