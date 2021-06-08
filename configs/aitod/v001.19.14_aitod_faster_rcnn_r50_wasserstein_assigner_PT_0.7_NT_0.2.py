"""
Faster R-CNN with Normalized Wasserstein Assigner
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.110
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.309
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.049
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.048
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.134
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.136
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.168
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.260
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.295
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.304
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.136
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.328
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.339
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.302
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.896
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.320
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.590
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.604
# Class-specific LRP-Optimal Thresholds # 
 [0.695 0.947 0.965 0.978 0.607 0.934 0.951 0.961]
2021-06-05 02:49:39,208 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.061 | bridge        | 0.067 | storage-tank | 0.244 |
| ship     | 0.211 | swimming-pool | 0.053 | vehicle      | 0.162 |
| person   | 0.061 | wind-mill     | 0.019 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-06-05 02:49:39,225 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.934 | bridge        | 0.937 | storage-tank | 0.780 |
| ship     | 0.798 | swimming-pool | 0.945 | vehicle      | 0.852 |
| person   | 0.944 | wind-mill     | 0.978 | None         | None  |
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
                    min_pos_iou=0.2,
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='wasserstein')),
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    assign_metric='wasserstein'))))

fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
