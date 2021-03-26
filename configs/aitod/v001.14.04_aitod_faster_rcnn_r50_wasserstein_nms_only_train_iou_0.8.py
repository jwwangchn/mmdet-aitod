"""
Faster R-CNN with Wasserstein NMS (only train) iou_threshold = 0.8

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.113
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.262
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.081
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.082
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.240
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.324
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.173
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.177
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.177
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.118
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.369
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.442
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.890
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.296
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.495
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.717
# Class-specific LRP-Optimal Thresholds # 
 [0.515 0.517 0.424 0.382 0.343 0.212 0.272 0.056]
2021-03-24 09:50:51,891 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.213 | bridge        | 0.043 | storage-tank | 0.200 |
| ship     | 0.222 | swimming-pool | 0.069 | vehicle      | 0.128 |
| person   | 0.046 | wind-mill     | 0.000 | None         | None  |
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
                nms=dict(type='wasserstein_nms', iou_threshold=0.8))))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
