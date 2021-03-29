"""
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.081
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.217
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.043
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.057
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.180
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.252
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.145
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.149
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.149
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.079
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.350
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.387
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.923
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.312
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.512
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.761
# Class-specific LRP-Optimal Thresholds # 
 [0.837 0.704 0.804 0.484 0.735 0.472 0.616   nan]
2021-03-26 20:19:40,887 - mmdet - INFO - 
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.133 | bridge        | 0.021 | storage-tank | 0.161 |
| ship     | 0.166 | swimming-pool | 0.043 | vehicle      | 0.113 |
| person   | 0.025 | wind-mill     | 0.000 | None         | None  |
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
            loss_bbox=dict(
                type='WassersteinLoss', 
                loss_weight=10.0,
                mode='norm_sqrt',
                gamma=1))))
    # train_cfg=dict(
    #     rpn=dict(
    #         assigner=dict(
    #             gpu_assign_thr=128)),
    #     rcnn=dict(
    #         assigner=dict(
    #             gpu_assign_thr=128))))

fp16 = dict(loss_scale=512.)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)