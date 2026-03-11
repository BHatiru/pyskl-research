---
marp: true
theme: default
paginate: true
style: |
  section { font-family: 'Segoe UI', Arial, sans-serif; }
  h1 { color: #1a73e8; }
  h2 { color: #2d3748; }
  table { font-size: 0.82em; }
  code { font-size: 0.85em; }
---

<!-- _class: lead -->
# Demo 1 — Federated Medical Action Recognition

**Privacy-Preserving Fall Detection via Skeleton-Based FL**

---

## Pipeline Overview

```
 FL Training (Colab T4)                  Real-Time Inference (CPU)
┌──────────────────────────┐            ┌──────────────────────────┐
│  NTU RGB+D 60 dataset    │            │  Webcam / Video input    │
│  ↓ filter 10 classes     │            │  ↓                       │
│  5 hospital clients      │   ONNX     │  YOLOX → RTMPose → STGCN │
│  (non-IID, Dirichlet)    │ ────────►  │  ↓                       │
│  STGCN++ (443K params)   │  export    │  10 medical actions      │
│  50 rounds FedAvg + BN   │            │  15-20 FPS, CPU only     │
└──────────────────────────┘            └──────────────────────────┘
  Patient data never leaves              Only skeleton keypoints —
  the hospital (FL)                      no face/identity on camera
```

---

## 10 Focused Classes

| # | Medical Actions (7) | # | Normal Context (3) |
|---|---|---|---|
| 0 | **Falling** | 7 | Standing up |
| 1 | Staggering | 8 | Sitting down |
| 2 | Touch head (headache) | 9 | Walking towards |
| 3 | Touch chest (heart pain) | | |
| 4 | Touch back (backache) | | |
| 5 | Touch neck (neckache) | | |
| 6 | Nausea / vomiting | | |

Filtered from NTU-60's 56K samples → focused medical subset

---

## FL Training Setup

| Component | Detail |
|---|---|
| **Data** | NTU RGB+D 60 — HRNet 2D skeletons, 10-class subset |
| **Non-IID split** | Dirichlet α=0.5 across 5 clients |
| **Warm-start** | 5 epochs centralized pretraining |
| **FL rounds** | 50 rounds FedAvg + BN calibration |
| **LR schedule** | 0.01 → 0.001 (R20) → 0.0001 (R35) |
| **Model** | STGCN++ — 6 stages, 443K params |
| **ONNX export** | ~2 MB, verified against PyTorch |

Privacy: raw skeleton data stays on each client — only model weights are aggregated

---

## Deployment Stack

| Model | Role | Size |
|---|---|---|
| `yolox_tiny.onnx` | Person detection | 19 MB |
| `rtmpose_m.onnx` | 17-keypoint pose | 52 MB |
| **`stgcnpp_medical_federated.onnx`** | **FL medical recognition** | **~2 MB** |

**Input:** `(1, 2, 100, 17, 3)` → **Output:** `(1, 10)` action logits

Runtime: `numpy` + `opencv` + `onnxruntime` — no PyTorch, no GPU needed

---

## Training Outputs

The Colab notebook generates:

- **Non-IID data distribution heatmap** — 5 clients × 10 medical classes
- **4-panel FL analysis** — timeline, client heatmap, loss curves, summary
- **Confusion matrix** with per-class accuracy bars
- **Fall detection metrics** — recall, precision, F1, missed-fall analysis

---

<!-- _class: lead -->
# Live Demo

---

## Demo Commands

**Our medical FL model** (10 classes, fall detection):
```
python demo/demo_onnx.py --device cpu --threads 4 --fast \
    --short-side 640 \
    --recog-model stgcnpp_medical_federated.onnx \
    --label-map demo1_fed_skeleton/label_map_medical.txt
```

**Pre-recorded video analysis** (auto-splits into 3s windows):
```
python demo/demo_onnx.py --mode clip --clip test/video.mp4 \
    --device cpu --threads 4 \
    --recog-model stgcnpp_medical_federated.onnx \
    --label-map demo1_fed_skeleton/label_map_medical.txt
```

**Pretrained baseline** (120 classes, comparison):
```
python demo/demo_onnx.py --device cpu --threads 4 --fast --short-side 640
```
