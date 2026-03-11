# Demo 1 — Federated Learning Skeleton-based HAR in Flower

## Overview

This demo implements **federated skeleton-based human-activity recognition**
using the [Flower](https://flower.dev) framework, aligned with our existing
ECG Flower pipeline.  A lightweight **STGCN++** model is trained on
**COCO-17 2D skeleton sequences** across multiple simulated clients under
non-IID data splits.

```
demo1_fed_skeleton/
├── prepare_clients.py        # data partitioning → per-client NPZ files
├── train_federated.py        # main FL training loop (Flower simulation)
├── export_onnx.py            # export final model → ONNX (Demo 2 compat.)
├── models/
│   └── stgcn.py              # self-contained STGCN++ classifier
├── fl/
│   ├── client.py             # Flower NumPyClient wrapper
│   └── strategy_fsar.py      # FedAvg / FedBN / Clustered strategy
├── data/fed_skeleton/        # generated client shards (gitignored)
└── outputs/                  # checkpoints + logs (gitignored)
```

---

## Quick Start

### 0. Install dependencies

```bash
pip install torch numpy flwr[simulation]
# optional: onnx onnxsim onnxruntime  (for export / verification)
```

### 1. Prepare client data (synthetic)

```bash
cd demo1_fed_skeleton

# 5 clients, Dirichlet α=0.3  (lower α → more heterogeneous)
python prepare_clients.py --synthetic --num-classes 10 \
    --samples-per-class 200 --num-clients 5 \
    --split dirichlet --alpha 0.3

# Alternative: subject-based split
python prepare_clients.py --synthetic --num-classes 10 \
    --samples-per-class 200 --num-clients 5 \
    --split subject
```

Each client gets a `client_<cid>.npz` with keys `x (N,2,100,17,3)` and
`y (N,)`.  A `test.npz` is created for global evaluation.

### 2. Train — FedAvg baseline

```bash
python train_federated.py --data-dir data/fed_skeleton \
    --num-rounds 30 --mode fedavg --lr 0.01
```

### 3. Train — FedBN (BatchNorm kept local)

```bash
python train_federated.py --data-dir data/fed_skeleton \
    --num-rounds 30 --mode fedbn --lr 0.01
```

### 4. Train — Clustered FedBN (FSAR-inspired)

```bash
python train_federated.py --data-dir data/fed_skeleton \
    --num-rounds 30 --mode cluster \
    --num-clusters 2 --cluster-every 5
```

### 5. Export to ONNX

```bash
python export_onnx.py --checkpoint outputs/global_model.pt \
    --output outputs/model.onnx --verify
```

The exported ONNX model accepts `(N, 2, 100, 17, 3)` — pyskl convention
`(N, M, T, V, C)` — and produces `(N, num_classes)` logits, drop-in compatible
with Demo 2.

---

## Data Format

| Axis | Meaning | Value |
|------|---------|-------|
| N    | samples | varies |
| M    | persons | 2 (zero-padded if single person) |
| T    | temporal frames | 100 (fixed) |
| V    | joints (COCO-17) | 17 |
| C    | channels | 3 (x, y, score) |

For real data, supply an NPZ with keys `x`, `y`, and optionally `subjects`
(integer subject IDs for the subject-based split).

---

## Model Architecture

The classifier mirrors pyskl's **STGCN++** design but is self-contained
(no mmcv / pyskl dependency):

| Component | Detail |
|-----------|--------|
| Graph | COCO-17 spatial adjacency (3 subsets: self, inward, outward) |
| GCN unit | Learnable adaptive adjacency (`PA`) + residual |
| TCN unit | Multi-scale temporal conv (dilations 1-2-3-4 + maxpool + 1×1) |
| Blocks | 6 ST-GCN blocks (channels 64→64→64→128→128→256) |
| Head | AdaptiveAvgPool2d → mean over persons → Linear |
| Params | ~285K (lightweight; pyskl full = ~1.4M with 10 blocks) |

---

## Federated Learning Modes

### FedAvg (baseline)
All parameters are aggregated via weighted averaging each round.
Simplest baseline; may struggle under strong non-IID.

### FedBN (FSAR-inspired)
BatchNorm parameters (`running_mean`, `running_var`, `weight`, `bias` of
all BN layers **including** `data_bn`) are kept local — never sent to or
received from the server.  Only convolutional / linear / adaptive-adjacency
parameters are aggregated.

**Rationale**: BN statistics capture client-specific data distributions.
Keeping them local lets each client adapt to its own skeleton scale /
annotation style while still sharing structural features globally.

### Clustered FedBN
Extends FedBN with periodic **client clustering** (every N rounds).

1. Each client sends a **descriptor** (flattened last-FC weights) with its
   fit response.
2. The server runs KMeans on all descriptors → assigns each client to a
   cluster.
3. Each cluster maintains its own aggregated model.  Clients receive their
   cluster's model at the start of each round.

This helps when clients naturally group (e.g., different camera angles or
body proportions) — each group converges to a tailored model.

---

## Logged Metrics (per round)

| Metric | Description |
|--------|-------------|
| `global_acc` | Accuracy on global test set (federated eval average) |
| `mean_client_acc` | Mean of per-client accuracies |
| `worst_client_acc` | Minimum per-client accuracy (fairness indicator) |
| `global_loss` | Weighted-average loss across clients |

Saved to `outputs/fl_history.json`.

---

## ONNX Compatibility with Demo 2

The exported ONNX model has the same tensor signature as pyskl's
`tools/export_stgcnpp_onnx.py`:

```
Input:  "skeleton"  float32  (N, C, T, V, M)   — C=2, T=100, V=17, M=1
Output: "logits"    float32  (N, num_classes)
```

> **Note**: Demo 2 expects C=3 (x, y, score) from HRNet. When bridging
> Demo 1 to Demo 2, either pad the score channel to 1.0 in the real-time
> pipeline or retrain Demo 1 with `--in-channels 3`.

---

## File Reference

| File | Purpose |
|------|---------|
| `prepare_clients.py` | Dataset preparation & non-IID partitioning |
| `models/stgcn.py` | Standalone STGCN++ classifier (PyTorch) |
| `fl/client.py` | Flower `NumPyClient` — `get_parameters`, `fit`, `evaluate` |
| `fl/strategy_fsar.py` | Custom Flower `Strategy` — FedAvg / FedBN / Cluster |
| `train_federated.py` | Orchestrates Flower simulation, logging, checkpointing |
| `export_onnx.py` | Export `.pt` → `.onnx` with verification |
