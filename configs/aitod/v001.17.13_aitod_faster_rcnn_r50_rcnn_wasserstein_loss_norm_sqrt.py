"""
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.090
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.230
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.057
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.066
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.195
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.290
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.156
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.160
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.160
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.093
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.370
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.412
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.913
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.312
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.537
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.748
# Class-specific LRP-Optimal Thresholds # 
 [0.854 0.389 0.614 0.526 0.89  0.447 0.568 0.051]
2021-03-25 17:49:10,639 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.140 | bridge        | 0.020 | storage-tank | 0.195 |
| ship     | 0.180 | swimming-pool | 0.053 | vehicle      | 0.118 |
| person   | 0.029 | wind-mill     | 0.001 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
"""

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_aitod.py',
    '../_base_/datasets/aitod_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(
                type='WassersteinLoss', 
                loss_weight=10.0,
                mode='norm_sqrt',
                gamma=2))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                gpu_assign_thr=128)),
        rcnn=dict(
            assigner=dict(
                gpu_assign_thr=128))))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)