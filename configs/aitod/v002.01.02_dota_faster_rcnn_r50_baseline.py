"""
Faster R-CNN baseline

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=3000 ] = 0.594
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=3000 ] = 0.373
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=3000 ] = 0.218
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=3000 ] = 0.390
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=3000 ] = 0.443
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.446
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.448
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=3000 ] = 0.448
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=3000 ] = 0.315
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=3000 ] = 0.479
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=3000 ] = 0.564

"""

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_dota.py',
    '../_base_/datasets/dota_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

fp16 = dict(loss_scale=512.)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=12, metric='bbox', proposal_nums=(300, 1000, 3000))