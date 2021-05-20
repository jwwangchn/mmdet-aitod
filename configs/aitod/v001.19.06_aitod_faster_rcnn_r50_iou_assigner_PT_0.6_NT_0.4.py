"""
Faster R-CNN with Normalized Wasserstein Assigner

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.073
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.173
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.048
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.025
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.187
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.315
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.117
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.118
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.118
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.029
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.321
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.443
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.932
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.292
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.543
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.813
# Class-specific LRP-Optimal Thresholds # 
 [0.796 0.366 0.603 0.574 0.572 0.262 0.339   nan]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.179 | bridge        | 0.007 | storage-tank | 0.134 |
| ship     | 0.115 | swimming-pool | 0.072 | vehicle      | 0.056 |
| person   | 0.022 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.847 | bridge        | 0.985 | storage-tank | 0.879 |
| ship     | 0.898 | swimming-pool | 0.924 | vehicle      | 0.947 |
| person   | 0.975 | wind-mill     | 1.000 | None         | None  |
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
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4)),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4))))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
