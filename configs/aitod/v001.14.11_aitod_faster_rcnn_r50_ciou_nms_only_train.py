"""
Faster R-CNN with Wasserstein NMS (only train)

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.109
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.257
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.077
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.072
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.234
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.328
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.168
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.172
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.172
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.104
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.379
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.444
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.898
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.287
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.455
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.727
# Class-specific LRP-Optimal Thresholds # 
 [0.814 0.498 0.55  0.484 0.613 0.439 0.467   nan]
2021-04-04 01:46:28,392 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.213 | bridge        | 0.022 | storage-tank | 0.198 |
| ship     | 0.199 | swimming-pool | 0.076 | vehicle      | 0.126 |
| person   | 0.041 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.810 | bridge        | 0.974 | storage-tank | 0.818 |
| ship     | 0.826 | swimming-pool | 0.922 | vehicle      | 0.879 |
| person   | 0.956 | wind-mill     | 1.000 | None         | None  |
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
                nms=dict(type='ciou_nms', iou_threshold=0.7))))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
