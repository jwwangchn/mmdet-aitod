"""
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.168
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.404
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.113
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.019
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.155
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.270
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.324
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.259
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.262
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.263
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.021
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.242
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.393
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.464
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.836
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.285
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.349
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.559
# Class-specific LRP-Optimal Thresholds # 
 [0.865 0.863 0.819 0.876 0.684 0.77  0.813 0.706]
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.219 | bridge        | 0.127 | storage-tank | 0.222 |
| ship     | 0.381 | swimming-pool | 0.122 | vehicle      | 0.165 |
| person   | 0.058 | wind-mill     | 0.053 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.796 | bridge        | 0.879 | storage-tank | 0.794 |
| ship     | 0.642 | swimming-pool | 0.870 | vehicle      | 0.838 |
| person   | 0.934 | wind-mill     | 0.937 | None         | None  |
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
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
