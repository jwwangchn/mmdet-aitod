"""
Faster R-CNN baseline

threshold = 0.65

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.105
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.245
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.077
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.075
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.229
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.321
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.181
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.181
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.106
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.384
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.520
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.904
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.286
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.467
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.739
# Class-specific LRP-Optimal Thresholds # 
 [0.9   0.636 0.613 0.579 0.848 0.449 0.451   nan]
2021-04-04 20:01:20,395 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.200 | bridge        | 0.028 | storage-tank | 0.189 |
| ship     | 0.190 | swimming-pool | 0.084 | vehicle      | 0.124 |
| person   | 0.039 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

threshold = 0.55

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.107
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.256
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.076
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.073
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.232
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.327
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.171
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.175
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.175
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.103
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.371
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.494
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.899
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.287
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.457
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.721
# Class-specific LRP-Optimal Thresholds # 
 [0.832 0.443 0.555 0.579 0.736 0.433 0.451   nan]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.212 | bridge        | 0.031 | storage-tank | 0.191 |
| ship     | 0.192 | swimming-pool | 0.086 | vehicle      | 0.124 |
| person   | 0.039 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

threshold = 0.45

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.108
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.259
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.076
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.072
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.231
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.330
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.166
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.169
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.169
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.100
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.361
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.471
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.896
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.287
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.436
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.718
# Class-specific LRP-Optimal Thresholds # 
 [0.781 0.443 0.506 0.579 0.736 0.402 0.377   nan]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.214 | bridge        | 0.033 | storage-tank | 0.192 |
| ship     | 0.193 | swimming-pool | 0.088 | vehicle      | 0.121 |
| person   | 0.039 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

threshold = 0.35

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.107
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.256
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.075
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.069
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.227
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.326
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.161
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.162
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.162
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.096
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.350
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.444
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.896
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.287
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.421
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.722
# Class-specific LRP-Optimal Thresholds # 
 [0.76  0.443 0.506 0.523 0.736 0.376 0.433   nan]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.214 | bridge        | 0.034 | storage-tank | 0.191 |
| ship     | 0.185 | swimming-pool | 0.088 | vehicle      | 0.114 |
| person   | 0.038 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+


Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.108
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.256
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.075
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.069
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.227
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.326
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.161
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.162
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.162
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.096
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.350
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.444
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.896
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.287
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.421
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.722
# Class-specific LRP-Optimal Thresholds # 
 [0.76  0.443 0.506 0.523 0.736 0.376 0.433   nan]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.214 | bridge        | 0.034 | storage-tank | 0.191 |
| ship     | 0.185 | swimming-pool | 0.088 | vehicle      | 0.114 |
| person   | 0.038 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.805 | bridge        | 0.960 | storage-tank | 0.821 |
| ship     | 0.832 | swimming-pool | 0.911 | vehicle      | 0.887 |
| person   | 0.955 | wind-mill     | 1.000 | None         | None  |
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
            num_classes=8)),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                gpu_assign_thr=1024)),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                gpu_assign_thr=1024))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='wasserstein_nms', iou_threshold=0.35),
            max_per_img=3000)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
