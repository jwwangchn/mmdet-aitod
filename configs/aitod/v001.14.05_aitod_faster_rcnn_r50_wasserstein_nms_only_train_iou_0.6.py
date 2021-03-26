"""
Faster R-CNN with Wasserstein NMS (only train) iou_threshold = 0.6

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.092
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.232
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.059
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.001
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.060
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.188
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.336
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.157
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.162
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.162
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.093
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.363
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.448
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.910
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.293
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.478
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.746
# Class-specific LRP-Optimal Thresholds # 
 [0.615 0.612 0.593 0.426 0.745 0.383 0.329   nan]
2021-03-24 08:56:11,028 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.197 | bridge        | 0.021 | storage-tank | 0.168 |
| ship     | 0.150 | swimming-pool | 0.075 | vehicle      | 0.097 |
| person   | 0.040 | wind-mill     | 0.000 | None         | None  |
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
                nms=dict(type='wasserstein_nms', iou_threshold=0.6))))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
