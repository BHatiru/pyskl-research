#!/usr/bin/env bash
# ==========================================================================
#  run_demo2.sh — Real-time skeleton-based action recognition demo
#  Repository : pyskl
#  Model      : STGCN++ (Joint modality, NTU-120 XSub, HRNet 2D skeletons)
#  Device     : GPU (CUDA) — falls back to CPU automatically if no GPU
# ==========================================================================

set -euo pipefail
PYSKL_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PYSKL_ROOT"

# ---------- 0. Config ----------
CONDA_ENV="pyskl"
DEVICE="cuda:0"

ACTION_CONFIG="configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py"
ACTION_CKPT="http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth"
LABEL_MAP="tools/data/label_map/nturgbd_120.txt"

DET_CONFIG="demo/faster_rcnn_r50_fpn_1x_coco-person.py"
DET_CKPT="https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
POSE_CONFIG="demo/hrnet_w32_coco_256x192.py"
POSE_CKPT="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"

# ---------- 1. Environment setup (run once) ----------
# conda env create -f pyskl.yaml          # Python 3.7
#   — OR —
# conda env create -f pyskl_310.yaml      # Python 3.10
# conda activate $CONDA_ENV
# pip install -e .
#
# Verify:
#   python -c "import pyskl; print('pyskl OK')"
#   python -c "import mmdet; print('mmdet OK')"
#   python -c "import mmpose; print('mmpose OK')"
#   python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# ---------- 2. Stream mode (default) ----------
python demo/demo_realtime.py \
    --mode stream \
    --config "$ACTION_CONFIG" \
    --checkpoint "$ACTION_CKPT" \
    --det-config "$DET_CONFIG" \
    --det-checkpoint "$DET_CKPT" \
    --pose-config "$POSE_CONFIG" \
    --pose-checkpoint "$POSE_CKPT" \
    --label-map "$LABEL_MAP" \
    --device "$DEVICE" \
    --clip-len 30 \
    --clip-stride 15 \
    --skip-frames 2 \
    --short-side 480
