"""
Faster R-CNN with Wasserstein NMS (only train)

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.115
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.265
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.083
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.076
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.238
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.345
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.173
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.178
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.178
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.111
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.381
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.459
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.893
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.304
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.465
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.722
# Class-specific LRP-Optimal Thresholds # 
 [0.723 0.678 0.538 0.533 0.804 0.427 0.47  0.056]
 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.224 | bridge        | 0.029 | storage-tank | 0.204 |
| ship     | 0.201 | swimming-pool | 0.086 | vehicle      | 0.129 |
| person   | 0.043 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.803 | bridge        | 0.964 | storage-tank | 0.813 |
| ship     | 0.823 | swimming-pool | 0.910 | vehicle      | 0.878 |
| person   | 0.953 | wind-mill     | 1.000 | None         | None  |
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
            rpn_proposal=dict(
                nms_pre=3000,
                max_per_img=3000,
                nms=dict(type='giou_nms', iou_threshold=0.7))))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
