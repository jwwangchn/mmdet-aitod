"""
Faster R-CNN with Normalized Wasserstein Assigner

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.088
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.216
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.055
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.049
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.203
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.302
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.143
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.146
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.146
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.063
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.362
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.444
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.917
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.298
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.498
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.766
# Class-specific LRP-Optimal Thresholds # 
 [0.822 0.552 0.696 0.749 0.752 0.593 0.663   nan]
2021-06-05 21:49:31,155 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.190 | bridge        | 0.014 | storage-tank | 0.164 |
| ship     | 0.146 | swimming-pool | 0.067 | vehicle      | 0.094 |
| person   | 0.030 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-06-05 21:49:31,155 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.839 | bridge        | 0.982 | storage-tank | 0.847 |
| ship     | 0.870 | swimming-pool | 0.925 | vehicle      | 0.910 |
| person   | 0.967 | wind-mill     | 1.000 | None         | None  |
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
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3)),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3))))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
