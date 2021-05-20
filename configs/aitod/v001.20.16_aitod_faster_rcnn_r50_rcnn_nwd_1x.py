"""

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.138
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.353
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.083
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.017
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.129
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.228
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.279
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.241
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.246
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.246
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.020
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.223
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.374
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.468
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.868
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.303
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.428
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.607
# Class-specific LRP-Optimal Thresholds # 
 [0.851 0.882 0.859 0.88  0.763 0.805 0.801 0.599]
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.171 | bridge        | 0.098 | storage-tank | 0.224 |
| ship     | 0.294 | swimming-pool | 0.085 | vehicle      | 0.158 |
| person   | 0.053 | wind-mill     | 0.023 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.856 | bridge        | 0.908 | storage-tank | 0.793 |
| ship     | 0.717 | swimming-pool | 0.909 | vehicle      | 0.844 |
| person   | 0.941 | wind-mill     | 0.972 | None         | None  |
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
            reg_decoded_bbox=True,
            loss_bbox=dict(type='WassersteinLoss', loss_weight=10.0))),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='wasserstein',
                gpu_assign_thr=512))),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='wasserstein_nms', iou_threshold=0.65),
            max_per_img=3000)))

fp16 = dict(loss_scale=512.)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
evaluation = dict(interval=12, metric='bbox')
