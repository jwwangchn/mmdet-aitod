"""
Faster R-CNN with Normalized Wasserstein Assigner

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.048
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.133
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.025
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.021
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.135
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.193
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.084
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.086
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.086
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.023
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.269
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.295
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.952
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.315
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.556
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.844
# Class-specific LRP-Optimal Thresholds # 
 [0.397 0.216 0.581 0.501 0.05  0.367 0.247   nan]
2021-06-07 13:33:38,141 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.093 | bridge        | 0.009 | storage-tank | 0.121 |
| ship     | 0.076 | swimming-pool | 0.010 | vehicle      | 0.053 |
| person   | 0.023 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-06-07 13:33:38,142 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.917 | bridge        | 0.984 | storage-tank | 0.884 |
| ship     | 0.925 | swimming-pool | 0.987 | vehicle      | 0.948 |
| person   | 0.974 | wind-mill     | 1.000 | None         | None  |
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
                    min_pos_iou=0.3)),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.8,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3))))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
