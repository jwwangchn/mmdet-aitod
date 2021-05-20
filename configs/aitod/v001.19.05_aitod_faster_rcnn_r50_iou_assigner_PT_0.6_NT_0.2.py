"""
Faster R-CNN with Normalized Wasserstein Assigner

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.113
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.293
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.060
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.092
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.217
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.277
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.201
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.208
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.208
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.171
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.386
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.415
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.897
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.313
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.505
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.670
# Class-specific LRP-Optimal Thresholds # 
 [0.901 0.801 0.871 0.884 0.774 0.842 0.801 0.248]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.181 | bridge        | 0.064 | storage-tank | 0.219 |
| ship     | 0.197 | swimming-pool | 0.051 | vehicle      | 0.139 |
| person   | 0.039 | wind-mill     | 0.018 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.863 | bridge        | 0.940 | storage-tank | 0.798 |
| ship     | 0.822 | swimming-pool | 0.946 | vehicle      | 0.872 |
| person   | 0.956 | wind-mill     | 0.976 | None         | None  |
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
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2)),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2))))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
