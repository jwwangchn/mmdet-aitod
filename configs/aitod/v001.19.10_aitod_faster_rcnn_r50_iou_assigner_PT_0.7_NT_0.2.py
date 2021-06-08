"""
Faster R-CNN with Normalized Wasserstein Assigner

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.082
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.233
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.038
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.067
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.171
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.202
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.156
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.162
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.162
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.131
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.322
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.322
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.920
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.323
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.544
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.714
# Class-specific LRP-Optimal Thresholds # 
 [0.539 0.691 0.854 0.883 0.1   0.817 0.757 0.053]
2021-06-05 19:04:40,294 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.087 | bridge        | 0.057 | storage-tank | 0.184 |
| ship     | 0.141 | swimming-pool | 0.038 | vehicle      | 0.104 |
| person   | 0.037 | wind-mill     | 0.010 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-06-05 19:04:40,294 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.928 | bridge        | 0.945 | storage-tank | 0.829 |
| ship     | 0.869 | swimming-pool | 0.947 | vehicle      | 0.903 |
| person   | 0.959 | wind-mill     | 0.976 | None         | None  |
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
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2)),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2))))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
