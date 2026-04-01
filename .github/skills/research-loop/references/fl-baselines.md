# FL Baselines

Current best results for federated skeleton-based medical action recognition.
Update this file after each significant experiment.

---

## 2D Baselines (COCO-17, 17 joints)

### Centralized STGCN++ (no FL)

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | TBD | Single model trained on all data |
| Fall Detection F1 | TBD | |
| Mean Class Accuracy | TBD | |

### FedAvg (5 clients, Dirichlet α=0.5)

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | TBD | 50 rounds, lr=0.01 |
| Fall Detection F1 | TBD | |
| Worst Client Accuracy | TBD | |
| Mean Client Accuracy | TBD | |

### FedBN (5 clients, Dirichlet α=0.5)

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | TBD | 50 rounds, lr=0.01, BN kept local |
| Fall Detection F1 | TBD | |
| Worst Client Accuracy | TBD | |
| Mean Client Accuracy | TBD | |

### Clustered FedBN

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | TBD | 50 rounds, 2 clusters, re-cluster every 5 rounds |
| Fall Detection F1 | TBD | |
| Worst Client Accuracy | TBD | |

---

## 3D Baselines (NTU Kinect, 25 joints)

### Centralized STGCN++ (no FL)

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | TBD | NTU 3D data, 25-joint layout |
| Fall Detection F1 | TBD | |
| Mean Class Accuracy | TBD | |

### FedAvg (5 clients, Dirichlet α=0.5)

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | TBD | |
| Fall Detection F1 | TBD | |
| Worst Client Accuracy | TBD | |

### FedBN (5 clients, Dirichlet α=0.5)

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | TBD | |
| Fall Detection F1 | TBD | |
| Worst Client Accuracy | TBD | |

---

## Advanced FL Methods (Phase 3)

| Method | 2D Acc | 3D Acc | Fall F1 | Worst Client | Notes |
|--------|--------|--------|---------|--------------|-------|
| FedAvg | TBD | TBD | TBD | TBD | Baseline |
| FedBN | TBD | TBD | TBD | TBD | BN local |
| FedProx | TBD | TBD | TBD | TBD | Proximal term |
| SCAFFOLD | TBD | TBD | TBD | TBD | Variance reduction |
| (others) | TBD | TBD | TBD | TBD | From paper survey |

---

## Model Info

- **Architecture**: STGCN++ (6 blocks, ~443K params)
- **2D Input**: `(N, 2, 100, 17, 3)` — COCO-17 skeleton
- **3D Input**: `(N, 2, 100, 25, 3)` — NTU Kinect 25-joint skeleton
- **Classes (10)**: falling, staggering, touch head, touch chest, touch back, touch neck, nausea, standing up, sitting down, walking
- **Dataset**: NTU RGB+D 60 medical subset
- **FL Framework**: Flower (simulation mode)

Fill in TBD values after running experiments.
