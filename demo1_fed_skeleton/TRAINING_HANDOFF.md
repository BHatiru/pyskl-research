# Training Handoff — RTX 5090 (32 GB VRAM)

> **Goal:** Train STGCN++ on the **3D NTU RGB+D 60** skeleton dataset with more epochs to significantly improve accuracy beyond current baselines. Export the best model to ONNX for deployment in the real-time emergency dashboard.

---

## Current State of Results

All results below use the **medical action subset** from NTU RGB+D 60.

### 10-Class Models (Original)

| Method | Skeleton | Best Acc | Final Acc | Notes |
|--------|----------|----------|-----------|-------|
| Centralized | 2D (17j) | **89.5%** | 88.1% | 20 epochs, upper bound |
| Centralized | 3D (25j) | ~76.5%* | — | Fewer epochs, underfit |
| FedAvg | 2D (17j) | 77.2% | 71.9% | 50 rounds, peaks R20 then degrades |
| FedBN | 2D (17j) | 76.4% | 72.6% | 50 rounds, BN local |
| Clustered FedBN | 2D (17j) | 76.8% | 74.6% | 50 rounds, cluster collapsed |
| **FedProx** | 2D (17j) | **73.4%** | 73.4% | **10 rounds only** — still improving |
| FedProx | 3D (25j) | 72.3% | — | 10 clients, 10 rounds |
| Ditto | 2D (17j) | 71.3% | 70.5% | 10 rounds, 2.2× compute |
| FedRep | 2D (17j) | 64.1% | 64.1% | 10 rounds, insufficient |

### 15-Class Models (Expanded Medical Set — Currently Deployed)

| Method | Skeleton | Best Acc | Notes |
|--------|----------|----------|-------|
| Centralized | 2D (17j, COCO) | 71.4% | `cent_2d_best.pt` → deployed as ONNX |
| Centralized | 3D (25j, NTU) | 76.5% | `cent_3d_best.pt` → ONNX exists but not pipeline-compatible |
| FedProx | 3D (25j), 10 clients | 72.3% | `fedprox_3d_10c_best.pt` |

### Key Insight
- **3D data (25 NTU joints) should outperform 2D (17 COCO joints)** — it has depth information + more joints
- The 3D centralized model already beats 2D (76.5% vs 71.4%) despite fewer training epochs
- **The 3D models are undertrained** — they need significantly more epochs with the 5090's compute power

---

## What Needs to Be Done

### Priority 1: Proper 3D Centralized Training
- Train centralized STGCN++ on 3D NTU-25 skeleton data for **50+ epochs** (not just 20)
- Use learning rate scheduling: step decay (×0.1 at epoch 30, ×0.01 at epoch 45) or cosine annealing
- Target: **>85% accuracy** (approaching the 89.5% ceiling of 2D centralized with better joint data)

### Priority 2: Full FedProx Run on 3D
- FedProx is the best FL method (won quick-scan at 10 rounds)
- Run for **50+ rounds** with proper LR schedule
- Fix the degradation-after-R20 issue (likely LR too high late in training)
- Test with `μ=0.01` (current) and `μ=0.001` (lighter proximal term)

### Priority 3: Client Scaling Experiments
- Compare: 5, 10, 20, 50 clients (checkpoints exist for 2D; redo for 3D)
- Measure worst-client accuracy (fairness metric)

### Priority 4: Export & Deploy
- Export best 3D model to ONNX
- **Important:** The dashboard currently uses 2D COCO-17 joints (from RTMPose). To use 3D NTU-25 model in the real-time pipeline, you'd need a pose estimator that outputs NTU-25 format, OR retrain with 2D inputs mapped to COCO-17.
- Alternatively: train a strong 2D model with more epochs as the deployment target, and keep 3D models for research comparison

---

## 15 Medical Action Classes

```
Index | NTU ID | Action
------|--------|--------
  0   | A043   | Falling
  1   | A044   | Staggering (touch head/body)
  2   | A049   | Nausea or vomiting
  3   | A039   | Touch head (headache)
  4   | A040   | Touch chest (heart pain)
  5   | A041   | Touch back (backache)
  6   | A042   | Touch neck (neckache)
  7   | A006   | Sneeze/cough (pick up/put on)
  8   | A008   | Standing up
  9   | A009   | Sitting down
 10   | A054   | Walking towards
 11   | A055   | Walking apart
 12   | A001   | Drinking
 13   | A002   | Eating
 14   | A017   | Phone call
```

### 4-Tier Emergency System

| Tier | Classes | Dashboard Behavior |
|------|---------|-------------------|
| **EMERGENCY** | Falling (0), Nausea (2) | Red alert, alarm sound |
| **PAIN** | Staggering (1), TouchHead (3), TouchChest (4), TouchBack (5), TouchNeck (6) | Orange warning |
| **SYMPTOM** | Sneeze (7) | Yellow notice |
| **NORMAL** | StandingUp (8), SittingDown (9), WalkTowards (10), WalkApart (11), Drinking (12), Eating (13), Phone (14) | Green/neutral |

---

## Dataset

### Source
- **NTU RGB+D 60** — 56,880 action samples, 60 classes, 40 subjects
- Filtered to the 15 medical/safety-related classes above
- After filtering: **~6,695 train / ~2,749 test** samples (10-class split; 15-class may differ)

### Skeleton Formats

| Format | Joints | Channels | Source |
|--------|--------|----------|--------|
| **3D NTU-25** | 25 (Kinect) | 3 (x, y, z) | Original NTU RGB+D dataset |
| **2D COCO-17** | 17 (COCO) | 3 (x, y, score) | HRNet/RTMPose 2D estimation |

### Data Location
- Pre-sharded client data: `demo1_fed_skeleton/data/fed_skeleton/client_*.npz`
- Test set: `demo1_fed_skeleton/data/fed_skeleton/test.npz`
- Raw NTU sequences: `ntu_action_dataset/sequences.npy` (9462, 64, 25, 3) — **if available**
- Each NPZ contains `x` (N, M=2, T=100, V, C) and `y` (N,) arrays

### Data Preparation
```bash
# Re-shard data for 3D with different number of clients
python demo1_fed_skeleton/prepare_clients.py \
    --source <path_to_ntu_data.pkl> \
    --output-dir demo1_fed_skeleton/data/fed_skeleton \
    --num-clients 5 \
    --alpha 0.5 \
    --seed 42
```

---

## Model Architecture

**STGCN++ (Self-contained, no mmcv dependency)**

| Property | Value |
|----------|-------|
| File | `demo1_fed_skeleton/models/stgcn.py` |
| Graph | Configurable — COCO-17 (2D) or NTU-25 (3D) |
| GCN | UnitGCN with learnable adaptive adjacency |
| TCN | Multi-scale temporal conv (dilations 1,2,3,4 + maxpool + 1×1) |
| Stages | 6 ST-GCN blocks: 64→64→64→128→128→256 |
| inflate/down | Stages [3, 5] — channel doubles + temporal stride=2 |
| Head | AdaptiveAvgPool2d → mean over M persons → Dropout(0.3) → Linear(256, num_classes) |
| Input | `(N, M=2, T=100, V=17or25, C=3)` |
| Params | ~443K (with dropout=0.3) |

---

## Training Commands

### Centralized Training (via Colab notebook)
The primary notebook for medical 15-class training:
```
demo1_fed_skeleton/train_colab_medical.ipynb
```
Upload to Colab, set runtime to GPU (5090 if local), and run all cells.

### Federated Training (CLI)
```bash
# FedProx on 3D data, 50 rounds, GPU
python demo1_fed_skeleton/train_federated.py \
    --data-dir demo1_fed_skeleton/data/fed_skeleton \
    --mode fedprox \
    --num-rounds 50 \
    --epochs-per-round 3 \
    --batch-size 64 \
    --lr 0.01 \
    --num-classes 15 \
    --in-channels 3 \
    --num-person 2 \
    --base-channels 64 \
    --num-stages 6 \
    --device cuda \
    --output-dir demo1_fed_skeleton/outputs
```

**Note:** `train_federated.py` currently supports modes: `fedavg`, `fedbn`, `cluster`. FedProx was implemented in the Colab notebooks (`train_colab_methods.ipynb`) but may need to be ported to the CLI script. Check if `--mode fedprox` is available; if not, add it using the FedProx implementation from the notebook (SGD + proximal term `μ/2 * ||w - w_global||²`).

### Key Hyperparameters to Tune

| Param | Current | Try |
|-------|---------|-----|
| `epochs` (centralized) | 20 | **50-100** |
| `rounds` (FL) | 10-50 | **50-100** |
| `epochs-per-round` | 1 | **2-5** |
| `lr` | 0.01 | 0.01 with cosine/step schedule |
| `batch-size` | 64 | **128-256** (5090 has 32GB) |
| `dropout` | 0.3 | 0.3 (keep) |
| `weight-decay` | 1e-4 | 1e-4 (keep) |
| FedProx `μ` | 0.01 | 0.01, 0.001 |

### LR Schedule (Critical — fixes R20 degradation)
The Phase 1 baselines used step decay (×0.1 @R20, ×0.01 @R35) but accuracy still degraded after R20. Try:
- **Cosine annealing** from 0.01 → 1e-5 over all rounds
- **Warmup + cosine**: 5 warmup rounds from 0.001 → 0.01, then cosine decay
- **Reduce on plateau**: halve LR if val acc doesn't improve for 5 rounds

---

## ONNX Export

After training, export the best checkpoint:

```bash
# Export 15-class 3D model
python demo1_fed_skeleton/export_onnx.py \
    --checkpoint demo1_fed_skeleton/outputs/cent_3d_best.pt \
    --output demo1_fed_skeleton/outputs/stgcnpp_medical_3d.onnx \
    --num-classes 15 \
    --in-channels 3 \
    --num-person 2 \
    --base-channels 64 \
    --num-stages 6 \
    --simplify \
    --verify

# Export 15-class 2D model (for dashboard deployment)
python demo1_fed_skeleton/export_onnx.py \
    --checkpoint demo1_fed_skeleton/outputs/cent_2d_best.pt \
    --output demo/onnx_models/stgcnpp_medical_15.onnx \
    --num-classes 15 \
    --in-channels 3 \
    --num-person 2 \
    --base-channels 64 \
    --num-stages 6 \
    --simplify \
    --verify
```

---

## Existing Checkpoints

| File | Description |
|------|-------------|
| `outputs/cent_2d_best.pt` | Centralized 2D, 15cls, currently deployed |
| `outputs/cent_3d_best.pt` | Centralized 3D, 15cls, undertrained |
| `outputs/fedprox_2d_5c_best.pt` | FedProx 2D, 5 clients |
| `outputs/fedprox_2d_10c_best.pt` | FedProx 2D, 10 clients |
| `outputs/fedprox_2d_20c_best.pt` | FedProx 2D, 20 clients |
| `outputs/fedprox_2d_50c_best.pt` | FedProx 2D, 50 clients |
| `outputs/fedprox_3d_10c_best.pt` | FedProx 3D, 10 clients, undertrained |
| `outputs/stgcnpp_medical_2d.onnx` | ONNX 2D model (deployed) |
| `outputs/stgcnpp_medical_3d.onnx` | ONNX 3D model (not deployed) |

---

## RTX 5090 Setup

### Environment
```bash
# Clone repo
git clone <repo_url>
cd pyskl-research

# Create conda env
conda env create -f pyskl_310.yaml
# OR
conda create -n pyskl python=3.10
conda activate pyskl
pip install -e .
pip install -r requirements.txt

# Key packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install flwr  # Flower FL framework
pip install onnx onnxruntime-gpu onnxsim
```

### GPU Expectations
- **RTX 5090 (32 GB):** Should handle batch_size=256+ easily for ~443K param model
- Training 100 epochs centralized should take **<30 minutes** on this hardware
- 50 FL rounds with 5 clients × 3 epochs/round = **~1 hour**
- Can run multiple experiments in parallel (model is tiny)

### Verify GPU
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

---

## Dashboard Integration

The real-time dashboard lives in `demo/`:
- **OpenCV version:** `demo/demo_dashboard.py`
- **Web version:** `demo/demo_web_dashboard.py` + `demo/web_dashboard.html`
- **Tier mapping:** `demo/tier_map_15.json`
- **ONNX models:** `demo/onnx_models/`

The dashboard uses **2D COCO-17 joints** (from YOLOX-Tiny → RTMPose-m). To deploy a new model:
1. Export checkpoint to ONNX with `export_onnx.py`
2. Copy to `demo/onnx_models/stgcnpp_medical_15.onnx`
3. The `MedicalRecognizer` class in `demo/demo_onnx.py` handles loading + inference

**If training a 2D model:** drop-in replacement, just copy ONNX file.
**If training a 3D model:** the pipeline can't use it directly (RTMPose outputs COCO-17, not NTU-25). Either:
- Train a good 2D model for deployment, keep 3D for research
- Map NTU-25 → COCO-17 at export time (lossy)
- Switch pose estimator to one that outputs 25 joints (complex)

---

## Experiment Tracking

Log results to `experiments/log.md` following this format:
```markdown
## Experiment Name — YYYY-MM-DD

**Config:** [key hyperparameters]
**Hardware:** RTX 5090, 32GB VRAM
**Data:** [dataset details]

| Method | Best Acc | Final Acc | Epochs/Rounds | Time |
|--------|----------|-----------|---------------|------|
| ... | ... | ... | ... | ... |

**Observations:** [what worked, what didn't]
**Next:** [follow-up experiments]
```

Also save per-experiment JSON with full histories to `demo1_fed_skeleton/outputs/`.

---

## TL;DR Action Plan

1. **Set up environment** on RTX 5090 machine
2. **Train centralized 3D** for 50-100 epochs with LR scheduling → target >85%
3. **Train centralized 2D** for 50-100 epochs → target >90% (this is the deployment model)
4. **Run FedProx 3D** for 50 rounds with cosine LR → close the FL gap
5. **Export best models** to ONNX
6. **Copy 2D ONNX** to `demo/onnx_models/stgcnpp_medical_15.onnx` for dashboard
7. **Log everything** to `experiments/log.md`
