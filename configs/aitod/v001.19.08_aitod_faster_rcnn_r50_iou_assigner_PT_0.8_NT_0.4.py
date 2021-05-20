"""
Faster R-CNN with Normalized Wasserstein Assigner

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.079
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.216
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.037
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.063
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.163
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.181
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.126
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.132
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.132
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.108
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.265
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.278
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.926
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.328
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.514
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.773
# Class-specific LRP-Optimal Thresholds # 
 [0.173 0.346 0.662 0.614 0.05  0.472 0.372 0.056]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.089 | bridge        | 0.061 | storage-tank | 0.179 |
| ship     | 0.150 | swimming-pool | 0.003 | vehicle      | 0.110 |
| person   | 0.038 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.919 | bridge        | 0.935 | storage-tank | 0.835 |
| ship     | 0.863 | swimming-pool | 0.996 | vehicle      | 0.900 |
| person   | 0.958 | wind-mill     | 1.000 | None         | None  |
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
                    neg_iou_thr=0.4,
                    min_pos_iou=0.2)),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.8,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.2))))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
