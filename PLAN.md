# Skeleton HAR — Federated Emergency Detection System

## System Overview

Two federated subsystems under one emergency detection umbrella:

1. **Skeleton HAR** (our focus) — camera-based action recognition via STGCN++
2. **Vitals Anomaly** (teammates handle) — wearable sensor anomaly detection via 1D-CNN

## Current State (April 11, 2026)

### What's Done
- Phase 1 baselines: Centralized 89.5%, FedAvg 77.2% (10 classes, 50R, 5C)
- PFL methods scan: FedProx **73.4%** best (10R quick scan) > FedBN 71.9% > FedAvg 70.8%
- Multimodal v1: skeleton-only 79.3% > late fusion 77.3% (synthetic IMU hurt)
- Vitals temporal model: centralized 87.2% on CPU
- Combined demo script working (`demo_combined/demo_emergency_system.py`)
- Expanded from 10 → **15 classes** with **3-tier severity system**

### What's Next — Priority Order
1. **Run `train_colab_sweep.ipynb`** on lab PC / Colab T4 (~45 min)
   - Centralized baselines (2D + 3D)
   - FedProx sweep: 5, 10, 20, 50 clients
   - 3D skeleton (NTU-25) experiment
   - ONNX export of best model
2. **Link trained ONNX to live demo** (`demo/demo_onnx.py`)
   - Plug `stgcnpp_medical_2d.onnx` + `label_map_medical_15.txt`
   - Add tier-based alert overlay (colors from `tier_map.json`)
3. **Sunday presentation demo** — live webcam + simulated vitals

## 15-Class 3-Tier Alert System

| Tier | Color | Idx | Classes | Demo Action |
|------|-------|-----|---------|-------------|
| 🔴 EMERGENCY | #FF0000 | 0-2 | falling, staggering, nausea/vomiting | Immediate nurse alert |
| 🟡 PAIN | #FFA500 | 3-6 | touch head/chest/back/neck | Nurse notify |
| 🔵 SYMPTOM | #4488FF | 7 | sneeze/cough | Log & monitor |
| 🟢 NORMAL | #44DD44 | 8-14 | stand, sit, walk×2, drink, eat, phone | No action |

### NTU-60 → Our Label Mapping
```python
MEDICAL_CLASS_MAP = {
    42: 0,  41: 1,  47: 2,   # EMERGENCY
    43: 3,  44: 4,  45: 5,  46: 6,  # PAIN
    40: 7,                    # SYMPTOM
    8: 8,  7: 9,  58: 10,  59: 11,  0: 12,  1: 13,  27: 14,  # NORMAL
}
```

## Model Architecture

**STGCN++** (lightweight, self-contained — no mmcv dependency):
- base_channels=64, 6 stages, inflate at [3,5], downsample at [3,5]
- ~111K params (COCO-17) / ~160K params (NTU-25)
- Input: `(N, 2, 100, V, 3)` where V=17 (2D) or V=25 (3D)
- Supports both COCO-17 and NTU-25 graphs

## FL Configuration

- **Method**: FedProx (μ=0.01)
- **Optimizer**: SGD lr=0.01, momentum=0.9, weight_decay=5e-4
- **Setup**: batch_size=64, 1 local epoch, Dirichlet α=0.5 non-IID split
- **Warmup**: 3 epochs centralized before FL rounds
- **BN calibration** after each aggregation round

## Key Files

| File | Purpose |
|------|---------|
| `demo1_fed_skeleton/train_colab_sweep.ipynb` | **Main training notebook** — run this first |
| `demo1_fed_skeleton/label_map_15.txt` | 15-class label file |
| `demo1_fed_skeleton/models/stgcn.py` | Self-contained STGCN++ (COCO-17 only, sweep notebook has dual-graph version) |
| `demo/demo_onnx.py` | Live webcam demo (YOLOX → RTMPose → STGCN++ ONNX) |
| `demo/onnx_models/` | Pre-existing ONNX models (yolox_tiny, rtmpose_m) |
| `federated_v2/train_vitals_colab.ipynb` | Vitals training (teammates) |
| `demo_combined/demo_emergency_system.py` | Combined system demo |
| `experiments/log.md` | Full experiment history with all results |

## Data (downloaded by notebook)

- **2D**: `ntu60_hrnet.pkl` (~1.1 GB) — COCO-17 joints from HRNet
- **3D**: `ntu60_3danno.pkl` (~0.6 GB) — NTU-25 joints from Kinect depth
- Source: `https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/`

## Demo Pipeline (inference)

```
Webcam → YOLOX-Tiny (19MB) → RTMPose-m (52MB) → STGCN++ ONNX (5MB) → 15-class prediction → tier alert overlay
```

All models run via ONNX Runtime — no PyTorch needed at inference.

## Prior Results Reference

| Experiment | Method | Accuracy | Notes |
|-----------|--------|----------|-------|
| Phase 1 (10cls, 50R) | Centralized | 89.5% | Upper bound |
| Phase 1 (10cls, 50R) | FedAvg 5C | 77.2% | ~12pp gap |
| Quick scan (10cls, 10R) | FedProx 5C | 73.4% | Best FL method |
| Quick scan (10cls, 10R) | FedBN 5C | 71.9% | |
| Quick scan (10cls, 10R) | FedAvg 5C | 70.8% | |
| Vitals | Centralized | 87.2% | Temporal 1D-CNN |

*Note: These were with 10 classes. The 15-class sweep will establish new baselines.*
