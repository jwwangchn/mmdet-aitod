"""
Faster R-CNN with Normalized Wasserstein Assigner

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.053
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.130
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.031
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.009
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.145
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.270
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.091
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.092
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.092
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.008
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.272
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.394
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.950
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.307
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.583
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.849
# Class-specific LRP-Optimal Thresholds # 
 [0.742 0.196 0.547 0.534 0.445 0.29  0.35    nan]
2021-06-05 19:50:57,332 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.141 | bridge        | 0.008 | storage-tank | 0.105 |
| ship     | 0.071 | swimming-pool | 0.047 | vehicle      | 0.033 |
| person   | 0.018 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-06-05 19:50:57,333 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.880 | bridge        | 0.982 | storage-tank | 0.905 |
| ship     | 0.933 | swimming-pool | 0.948 | vehicle      | 0.969 |
| person   | 0.981 | wind-mill     | 1.000 | None         | None  |
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
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4)),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4))))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
