"""
Faster R-CNN with Wasserstein NMS (only train) iou_threshold = 0.85

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.116
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.260
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.086
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.001
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.080
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.243
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.345
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.183
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.188
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.188
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.119
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.385
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.511
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.895
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.276
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.445
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.723
# Class-specific LRP-Optimal Thresholds # 
 [0.796 0.7   0.521 0.568 0.626 0.375 0.345   nan]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.209 | bridge        | 0.050 | storage-tank | 0.205 |
| ship     | 0.207 | swimming-pool | 0.082 | vehicle      | 0.131 |
| person   | 0.045 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.825 | bridge        | 0.947 | storage-tank | 0.815 |
| ship     | 0.822 | swimming-pool | 0.919 | vehicle      | 0.875 |
| person   | 0.954 | wind-mill     | 1.000 | None         | None  |
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
                nms=dict(type='wasserstein_nms', iou_threshold=0.85))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='wasserstein_nms', iou_threshold=0.85),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='wasserstein_nms', iou_threshold=0.65),
            max_per_img=3000)))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
