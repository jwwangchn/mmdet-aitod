"""
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.043
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.145
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.023
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.018
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.049
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.089
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.172
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.152
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.178
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.194
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.077
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.201
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.209
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.323
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.947
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.325
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.785
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.753
# Class-specific LRP-Optimal Thresholds # 
 [0.096 0.108 0.343 0.375 0.072 0.392 0.348 0.05 ]
2021-03-26 12:35:52,903 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.000 | bridge        | 0.029 | storage-tank | 0.055 |
| ship     | 0.198 | swimming-pool | 0.000 | vehicle      | 0.092 |
| person   | 0.021 | wind-mill     | 0.000 | None         | None  |
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
        loss_bbox=dict(type='WassersteinLoss', loss_weight=20.0)))

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)