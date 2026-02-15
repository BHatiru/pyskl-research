# ONNX Pipeline Demo – Architecture & Benchmarks

## Overview

A **pure-ONNX** real-time skeleton-based action recognition pipeline that
requires **only** `numpy`, `opencv-python`, and `onnxruntime(-gpu)` at runtime.
No mmcv, mmpose, mmdet, or pyskl imports needed for inference.

## Pipeline

```
Webcam frame
  │
  ▼
┌──────────────────────┐
│  YOLOX-Tiny (ONNX)   │  Person detection, 416×416, built-in NMS
│  19.3 MB             │  HumanArt+COCO trained, AP 47.7
└──────────┬───────────┘
           │ person bboxes
           ▼
┌──────────────────────┐
│  RTMPose-m (ONNX)    │  Top-down 17-kp COCO pose, 192×256
│  51.8 MB             │  SimCC output, body7 trained, AP 75.3
└──────────┬───────────┘
           │ keypoints (17×2) + scores (17)
           ▼
┌──────────────────────┐
│  PreNormalize2D      │  Pixel coords → [-1,1] normalization
│  (numpy, in-script)  │  Zero out low-confidence (<0.01) joints
└──────────┬───────────┘
           │ sliding window buffer
           ▼
┌──────────────────────┐
│  STGCN++ (ONNX)      │  Skeleton action recognition
│  5.4 MB              │  Input: (1,2,100,17,3), Output: 120 NTU classes
│  NTU120 xsub hrnet   │  Uniform temporal sampling
└──────────┬───────────┘
           │ top-5 action labels + probabilities
           ▼
    ┌──────────────┐
    │  Visualization│  Skeleton overlay, HUD, FPS counter
    └──────────────┘
```

## Models

| Model      | File                               | Size    | Input          | Provider |
| ---------- | ---------------------------------- | ------- | -------------- | -------- |
| YOLOX-Tiny | `yolox_tiny.onnx`                  | 19.3 MB | (1,3,416,416)  | CUDA     |
| RTMPose-m  | `rtmpose_m.onnx`                   | 51.8 MB | (1,3,256,192)  | CUDA     |
| STGCN++    | `stgcnpp_ntu120_xsub_hrnet_j.onnx` | 5.4 MB  | (1,2,100,17,3) | CPU      |

## Benchmark Results (RTX 3070 Ti)

### ONNX Pipeline (det/pose=GPU, recog=CPU)

| Stage                      | Time (ms) | FPS      |
| -------------------------- | --------- | -------- |
| Detection (YOLOX-Tiny)     | 12.3      | 81.5     |
| Pose (RTMPose-m, 1 person) | 4.7       | 214.7    |
| Recognition (STGCN++)      | 22.5      | 44.5     |
| **TOTAL**                  | **39.4**  | **25.4** |

### ONNX Pipeline (all CPU)

| Stage           | Time (ms) | FPS      |
| --------------- | --------- | -------- |
| Detection       | 27.4      | 36.5     |
| Pose (1 person) | 14.3      | 70.0     |
| Recognition     | 22.7      | 44.0     |
| **TOTAL**       | **64.4**  | **15.5** |

### Comparison with Old mmcv Pipeline

| Pipeline                         | Det        | Pose       | Device  | FPS      | Speedup               |
| -------------------------------- | ---------- | ---------- | ------- | -------- | --------------------- |
| mmcv: Faster-RCNN + HRNet-w32    | 71ms       | 89ms       | GPU     | 6.2      | 1.0×                  |
| mmcv: YOLOX-Tiny + ViPNAS-MbV3   | 19ms       | 26ms       | GPU     | 21.5     | 3.5×                  |
| mmcv: Faster-RCNN + HRNet-w32    | 2316ms     | 243ms      | CPU     | 0.4      | —                     |
| **ONNX: YOLOX-Tiny + RTMPose-m** | **12.3ms** | **4.7ms**  | **GPU** | **25.4** | **4.1×**              |
| **ONNX: YOLOX-Tiny + RTMPose-m** | **27.4ms** | **14.3ms** | **CPU** | **15.5** | **39×** (vs mmcv CPU) |

### Key Improvements

- **No framework overhead**: Pure ONNX runtime, no mmcv/mmpose/mmdet initialization
- **RTMPose-m**: State-of-the-art pose with SimCC — 5.5× faster than ViPNAS on GPU
- **Built-in NMS in YOLOX**: Eliminates post-processing Python loop
- **CPU-viable**: 15.5 FPS on CPU alone (old pipeline: 0.4 FPS)
- **Minimal dependencies**: `pip install numpy opencv-python onnxruntime-gpu nvidia-cudnn-cu12`

## Usage

```bash
# GPU mode (recommended)
python demo/demo_onnx.py --device cuda --short-side 480

# CPU mode
python demo/demo_onnx.py --device cpu

# Video file
python demo/demo_onnx.py --clip path/to/video.mp4

# Benchmark
python demo/demo_onnx.py --benchmark --device cuda --benchmark-iters 200

# Skip detection every other frame for more speed
python demo/demo_onnx.py --device cuda --det-every 2
```

## Dependencies

```
numpy
opencv-python
onnxruntime-gpu>=1.19.0    # or onnxruntime for CPU-only
nvidia-cudnn-cu12>=9.1.0   # only for GPU mode
```

## Files

```
demo/
  demo_onnx.py              – Main ONNX pipeline demo script
  run_onnx_demo.bat         – Windows launcher
  onnx_models/
    yolox_tiny.onnx          – Person detector
    rtmpose_m.onnx           – Pose estimator
    stgcnpp_ntu120_xsub_hrnet_j.onnx  – Action recognizer
tools/
  export_stgcnpp_onnx.py    – Script to export STGCN++ from PyTorch to ONNX
```
