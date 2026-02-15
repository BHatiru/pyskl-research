# Demo 2 — Real-time STGCN++ Action Recognition: Design Notes

## What this demo does

Opens a webcam (or video file), streams frames, and periodically runs a full
**skeleton-based action recognition** pipeline on a sliding window of captured
frames. The predicted action label and top-5 probabilities are overlaid on the
live feed in real time.

---

## Architecture at a glance

```
  Webcam / Video
       │
       ▼  (main thread – smooth 30 fps display)
  ┌──────────────┐
  │ Frame Buffer │  ← circular deque, keeps last ~90 frames
  └──────┬───────┘
         │  every `clip_stride` new frames  (background thread)
         ▼
  ┌──────────────────────────────────┐
  │  1. Person Detection             │  Faster-RCNN R50 (mmdet)
  │  2. Pose Estimation              │  HRNet-w32 (mmpose, COCO 17-kp)
  │  3. Skeleton Tracking            │  Naive inter-frame tracker
  │  4. Action Recognition           │  STGCN++ (pyskl)
  └──────────────┬───────────────────┘
                 │
                 ▼
         Action label + top-5 scores → drawn on the live display
```

All models run on **GPU** by default. If no CUDA device is found the script
falls back to CPU automatically (expect ~10× slower inference).

---

## Model details

| Component         | Architecture                | Source      | Weights                                                                                                                                                                       |
| ----------------- | --------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Person detector   | Faster-RCNN, ResNet-50-FPN  | mmdet 2.23  | [coco-person](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth) |
| Pose estimator    | HRNet-w32                   | mmpose 0.24 | [coco-256×192](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth)                                                             |
| Action recogniser | **STGCN++** (RecognizerGCN) | pyskl       | [NTU-120 XSub HRNet Joint](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth)                                                         |

### Why STGCN++?

- **Lightweight GCN backbone** — inference on a 100-frame skeleton clip is
  <10 ms on GPU (the bottleneck is detection + pose, not recognition).
- Trained on **NTU RGB+D 120** (120 daily-life action classes) with **2D HRNet
  skeletons** — the exact same skeleton format the demo pipeline produces from
  webcam frames.
- Official pretrained checkpoint provided by the pyskl Model Zoo (84.4 % Top-1
  on NTU-120 XSub).

### Skeleton format used

**2D HRNet (COCO 17-keypoint)**  
Each frame produces `[M, 17, 3]` — M persons, 17 joints, (x, y, score).  
The STGCN++ model config specifies `graph_cfg=dict(layout='coco')` which
matches this layout.

The alternative would be Kinect 3D skeletons (25 joints, layout `'nturgb+d'`),
but those require a Kinect sensor and a different checkpoint.

---

## Two demo modes

### Stream mode (default)

The webcam feed is displayed continuously. A **background thread** grabs the
latest `clip_len` frames (default 30) every `clip_stride` new frames (default 15) and runs the full det → pose → recognition pipeline. Because inference is
off the main thread, the display never freezes.

Typical throughput on a mid-range laptop GPU (e.g. RTX 3060 Mobile):

| Stage                                     | Time per window     |
| ----------------------------------------- | ------------------- |
| Detection (30 frames, skip=2 → 15 frames) | ~1.5 s              |
| Pose estimation (15 frames)               | ~0.8 s              |
| Recognition (STGCN++)                     | ~0.01 s             |
| **Total**                                 | **~2.3 s → 0.4 Hz** |

The label refreshes roughly **every 2-3 seconds** while the camera feed stays
at full frame rate.

### Clip mode

Press **r** to record a 3-second clip, which is then processed in one shot.
Useful for a controlled demo where you want to perform an action, stop, and
show the result.

---

## Dataset & label map

The label map file `tools/data/label_map/nturgbd_120.txt` contains 120 action
names (one per line, 0-indexed), for example:

```
0  drink water
1  eat meal/snack
2  brushing teeth
9  clapping
10 reading
47 punching/slapping other person
...
```

These are the actions defined in **NTU RGB+D 120** (a large-scale indoor action
recognition benchmark with 114k video clips).

---

## Files created

| File                    | Purpose                                                  |
| ----------------------- | -------------------------------------------------------- |
| `demo/demo_realtime.py` | Main real-time inference script (stream + clip modes)    |
| `run_demo2.bat`         | One-click Windows launcher with all arguments pre-filled |
| `run_demo2.sh`          | Equivalent bash launcher for Linux/macOS                 |
| `demo2_notes.md`        | This file — design notes for the advisor meeting         |

---

## Quick-start commands

```bash
# GPU — streaming webcam
python demo/demo_realtime.py --device cuda:0

# GPU — process bundled sample video (loops)
python demo/demo_realtime.py --video demo/ntu_sample.avi --device cuda:0

# GPU — clip mode (press 'r' to record 3 s)
python demo/demo_realtime.py --mode clip --device cuda:0

# CPU fallback (slow — expect ~0.05 Hz)
python demo/demo_realtime.py --device cpu --skip-frames 3 --clip-len 15

# Original offline demo for comparison (writes output mp4)
python demo/demo_skeleton.py demo/ntu_sample.avi demo/demo_output.mp4 \
    --config configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py \
    --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
```

---

## Potential improvements (out of scope for now)

- Replace Faster-RCNN + HRNet with a lighter detector/pose stack (e.g. YOLO +
  MoveNet) to push inference closer to real-time on CPU.
- Use TorchScript / ONNX export for the GCN to reduce recognition latency.
- Multi-person action label display (currently shows top-1 over all tracked
  persons).
