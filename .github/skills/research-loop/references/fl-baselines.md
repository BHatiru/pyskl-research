# FL Baselines

Current best results for federated skeleton-based medical action recognition.
Update this file after each significant experiment.

---

## 2D Baselines (COCO-17, 17 joints)

### Centralized STGCN++ (no FL)

| Metric | Value | Notes |
|--------|-------|-------|
| Best Accuracy | **0.8949** | 20 epochs, SGD lr=0.01, MultiStepLR [10,15] |
| Final Accuracy | 0.8807 | |
| Training Time | 1516s | Colab T4 GPU |
| Fall Detection F1 | TBD | Need to re-run with confusion matrix |

### FedAvg (5 clients, Dirichlet α=0.5)

| Metric | Value | Notes |
|--------|-------|-------|
| Best Accuracy | **0.7719** | Peak at R20, then degrades |
| Final Accuracy | 0.7195 | R50 |
| Worst Client Acc (R50) | 0.5991 | |
| Mean Client Acc (R50) | 0.7629 | |
| Training Time | 4551s | 50 rounds, Colab T4 |
| Fall Detection F1 | TBD | |

### FedBN (5 clients, Dirichlet α=0.5)

| Metric | Value | Notes |
|--------|-------|-------|
| Best Accuracy | **0.7636** | Slightly worse than FedAvg |
| Final Accuracy | 0.7264 | R50 |
| Worst Client Acc (R50) | 0.5926 | |
| Mean Client Acc (R50) | 0.7529 | |
| Training Time | 4546s | |
| Fall Detection F1 | TBD | |

### Clustered FedBN

| Metric | Value | Notes |
|--------|-------|-------|
| Best Accuracy | **0.7683** | Peak at R30 |
| Final Accuracy | 0.7461 | Best final among FL methods |
| Worst Client Acc (R50) | 0.6503 | Best fairness |
| Mean Client Acc (R50) | 0.7422 | |
| Training Time | 4550s | |
| Clusters | [0,0,0,0,1] | Collapsed after R10 — client 4 isolated |
| Fall Detection F1 | TBD | |

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
