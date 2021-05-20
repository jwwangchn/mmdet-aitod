"""
Faster R-CNN with Normalized Wasserstein Assigner

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.074
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.208
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.034
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.001
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.057
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.156
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.177
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.128
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.134
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.134
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.110
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.270
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.280
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.929
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.315
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.554
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.760
# Class-specific LRP-Optimal Thresholds # 
 [0.12  0.399 0.791 0.805 0.05  0.683 0.578 0.05 ]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.077 | bridge        | 0.057 | storage-tank | 0.170 |
| ship     | 0.136 | swimming-pool | 0.009 | vehicle      | 0.101 |
| person   | 0.034 | wind-mill     | 0.007 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.929 | bridge        | 0.941 | storage-tank | 0.837 |
| ship     | 0.874 | swimming-pool | 0.988 | vehicle      | 0.907 |
| person   | 0.962 | wind-mill     | 0.991 | None         | None  |
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
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2)),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.8,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2))))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
