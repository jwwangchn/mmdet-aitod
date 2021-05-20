"""
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.333
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.555
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.350
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.012
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.096
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.295
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.389
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.403
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.419
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.423
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.012
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.169
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.411
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.477
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.709
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.202
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.253
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.464
# Class-specific LRP-Optimal Thresholds # 
 [0.735 0.708 0.876 0.553 0.588 0.585 0.558 0.921 0.623 0.627 0.695 0.745
 0.56  0.491 0.719 0.07    nan   nan]

+--------------------+-------+-------------------+-------+------------------+-------+
| category           | AP    | category          | AP    | category         | AP    |
+--------------------+-------+-------------------+-------+------------------+-------+
| plane              | 0.596 | baseball-diamond  | 0.389 | bridge           | 0.210 |
| ground-track-field | 0.394 | small-vehicle     | 0.300 | large-vehicle    | 0.516 |
| ship               | 0.578 | tennis-court      | 0.773 | basketball-court | 0.348 |
| storage-tank       | 0.413 | soccer-ball-field | 0.296 | roundabout       | 0.383 |
| harbor             | 0.305 | swimming-pool     | 0.263 | helicopter       | 0.229 |
| container-crane    | 0.002 | airport           | 0.000 | helipad          | 0.000 |
+--------------------+-------+-------------------+-------+------------------+-------+
2021-05-13 16:41:53,789 - mmdet - INFO - 
+--------------------+-------+-------------------+-------+------------------+-------+
| category           | oLRP  | category          | oLRP  | category         | oLRP  |
+--------------------+-------+-------------------+-------+------------------+-------+
| plane              | 0.452 | baseball-diamond  | 0.641 | bridge           | 0.821 |
| ground-track-field | 0.676 | small-vehicle     | 0.753 | large-vehicle    | 0.579 |
| ship               | 0.469 | tennis-court      | 0.281 | basketball-court | 0.733 |
| storage-tank       | 0.627 | soccer-ball-field | 0.777 | roundabout       | 0.670 |
| harbor             | 0.735 | swimming-pool     | 0.768 | helicopter       | 0.779 |
| container-crane    | 0.993 | airport           | 1.000 | helipad          | 1.000 |
+--------------------+-------+-------------------+-------+------------------+-------+
"""

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_dota.py',
    '../_base_/datasets/dota_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]


model = dict(
    rpn_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='WassersteinLoss', loss_weight=10.0)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                gpu_assign_thr=512)),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='wasserstein_nms', iou_threshold=0.85)),
        rpn=dict(
            assigner=dict(
                gpu_assign_thr=512, 
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='wasserstein'))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='wasserstein_nms', iou_threshold=0.85),
            min_bbox_size=0)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1)

fp16 = dict(loss_scale=512.)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=1)
evaluation = dict(interval=12, metric='bbox')