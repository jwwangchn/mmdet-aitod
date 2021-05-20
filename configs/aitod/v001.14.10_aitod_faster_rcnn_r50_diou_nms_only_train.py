"""
Faster R-CNN with Wasserstein NMS (only train)

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.112
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.268
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.076
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.076
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.236
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.331
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.170
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.174
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.174
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.108
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.377
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.457
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.895
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.304
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.450
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.720
# Class-specific LRP-Optimal Thresholds # 
 [0.74  0.63  0.522 0.599 0.773 0.441 0.435 0.051]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.222 | bridge        | 0.030 | storage-tank | 0.195 |
| ship     | 0.196 | swimming-pool | 0.084 | vehicle      | 0.126 |
| person   | 0.041 | wind-mill     | 0.001 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.800 | bridge        | 0.968 | storage-tank | 0.820 |
| ship     | 0.826 | swimming-pool | 0.910 | vehicle      | 0.879 |
| person   | 0.955 | wind-mill     | 0.999 | None         | None  |
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
                nms=dict(type='diou_nms', iou_threshold=0.7))))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
