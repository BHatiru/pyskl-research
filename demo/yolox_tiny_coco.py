# Self-contained YOLOX-Tiny config for 80-class COCO inference.
# Person is class 0 in COCO — caller should use res[0] to get person boxes.
#
# Checkpoint (auto-downloaded):
#   https://download.openmmlab.com/mmdetection/v2.0/yolox/
#   yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth
#
# Compared to Faster-RCNN R50 FPN:
#   - Backbone: CSPDarknet (depth=0.33, width=0.375) vs ResNet-50
#   - Single-stage (no RPN/ROI) vs two-stage
#   - ~5× fewer FLOPs, ~5× faster on GPU and >>5× faster on CPU

img_scale = (416, 416)

model = dict(
    type="YOLOX",
    input_size=img_scale,
    random_size_range=(10, 20),
    random_size_interval=10,
    backbone=dict(type="CSPDarknet", deepen_factor=0.33, widen_factor=0.375),
    neck=dict(
        type="YOLOXPAFPN",
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
    ),
    bbox_head=dict(
        type="YOLOXHead",
        num_classes=80,
        in_channels=96,
        feat_channels=96,
    ),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)

# ---------------------------------------------------------------------------
# Inference data pipeline (matches official mmdet YOLOX config)
# ---------------------------------------------------------------------------
test_pipeline = [
    dict(type="LoadImageFromWebcam"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(
                type="Pad",
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0)),
            ),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(test=dict(pipeline=test_pipeline))
