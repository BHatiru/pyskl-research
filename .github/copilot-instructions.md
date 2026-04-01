# Project: pyskl-research

## What This Is

A research workspace extending **PYSKL** (skeleton-based action recognition toolkit) with **federated learning** for privacy-preserving medical action recognition (fall detection, health monitoring). Built on PyTorch, Flower FL framework, and the STGCN++ family of graph convolutional networks.

## Active Research

- **Federated learning for medical action recognition** — privacy-preserving fall detection across simulated hospital clients
- **New model architectures / methods** — improving GCN-based skeleton recognition

## Codebase Structure

| Directory | Purpose |
|-----------|---------|
| `pyskl/` | Core library: models, datasets, APIs, utils (mmcv-based registry system) |
| `configs/` | Training configs per algorithm (aagcn, ctrgcn, dgstgcn, msg3d, posec3d, stgcn, stgcn++) |
| `demo/` | Real-time demos: webcam, video, ONNX pipeline, gesture recognition |
| `demo1_fed_skeleton/` | **Primary research area** — federated learning medical HAR |
| `demo1_fed_skeleton/fl/` | Flower FL infrastructure: client.py, strategy_fsar.py |
| `demo1_fed_skeleton/models/` | Self-contained STGCN++ (no mmcv dependency) |
| `tools/` | Training/testing scripts, ONNX export |
| `experiments/` | Experiment tracking log and reports |

## Key Technical Conventions

- **Input tensor format:** `(N, M, T, V, C)` — batch, persons (2), temporal frames (100), joints (17 COCO), channels (3: x, y, score)
- **Skeleton format:** COCO-17 (17 joints), 2D coordinates + confidence score
- **Normalization:** Skeleton coords normalized to [-1, 1]; low-confidence joints zeroed
- **Temporal sampling:** Uniform sampling to fixed clip_len=100 frames; zero-pad to 2 persons
- **Model registry:** pyskl uses mmcv's `@BACKBONES.register_module()` pattern for core models
- **Self-contained FL model:** `demo1_fed_skeleton/models/stgcn.py` is standalone (no mmcv dep), ~285K params

## FL Setup Details

- **Framework:** Flower (flwr) with simulation mode
- **Strategies:** FedAvg, FedBN (BatchNorm local), Clustered FedBN
- **Clients:** 5 simulated hospital clients, Dirichlet-α=0.5 (non-IID)
- **Medical classes (10):** falling, staggering, touch head/chest/back/neck, nausea, standing up, sitting down, walking
- **Dataset source:** NTU RGB+D 60 filtered to medical subset
- **Privacy model:** Raw skeleton stays on-device; only aggregated weights traverse network

## Config System

Pyskl configs are Python files (not YAML) following mmcv conventions:
```python
model = dict(type='RecognizerGCN', backbone=dict(type='STGCN', ...), cls_head=dict(...))
dataset_type = 'PoseDataset'
train_pipeline = [dict(type='PreNormalize3D'), dict(type='GenSkeFeat', ...), ...]
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
```

## Commands

- **Install:** `pip install -e .` or `python setup.py develop`
- **Train (pyskl):** `python tools/train.py <config_path> [--gpus N]`
- **Train (FL):** `python demo1_fed_skeleton/train_federated.py --data-dir data/fed_skeleton --mode fedbn --num-rounds 30`
- **Export ONNX:** `python demo1_fed_skeleton/export_onnx.py --checkpoint outputs/global_model.pt --output outputs/model.onnx`
- **Test:** `python tools/test.py <config_path> <checkpoint_path>`

## Research Documentation

- [demo1_fed_skeleton/demo1_notes.md](../demo1_fed_skeleton/demo1_notes.md) — FL setup details and architecture
- [demo1_fed_skeleton/presentation.md](../demo1_fed_skeleton/presentation.md) — Research goals and results
- [demo/onnx_pipeline_notes.md](../demo/onnx_pipeline_notes.md) — ONNX deployment pipeline
- [experiments/log.md](../experiments/log.md) — Running experiment log
