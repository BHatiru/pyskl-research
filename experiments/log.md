# Experiment Log

Running record of all experiments for the pyskl-research federated learning project.
Entries are appended chronologically. Each entry follows the template in `.github/skills/research-loop/references/experiment-log-template.md`.

---

## Research Roadmap — 2026-03-24

### Phase 1: Establish Baselines
- [ ] Centralized STGCN++ on 2D medical subset (no FL)
- [ ] FedAvg baseline (5 clients, α=0.5, 50 rounds)
- [ ] FedBN baseline (same, BN kept local)
- [ ] Clustered FedBN baseline (2 clusters, re-cluster every 5 rounds)
- [ ] Record all metrics in `fl-baselines.md`

### Phase 2: 2D → 3D Skeleton Transition
- [ ] Adapt FL model for NTU RGB+D 3D Kinect data (25 joints instead of 17)
- [ ] Update `demo1_fed_skeleton/models/stgcn.py` graph for NTU 25-joint layout
- [ ] Update `prepare_clients.py` for 3D data loading
- [ ] Re-run all baselines (centralized, FedAvg, FedBN) on 3D data
- [ ] Compare 2D vs 3D — does 3D improve medical action recognition?

### Phase 3: Advanced FL Methods
- [ ] Literature review via scout: survey top FL methods from recent papers
- [ ] Implement FedProx (adds proximal term to client loss)
- [ ] Implement SCAFFOLD (variance reduction via control variates)
- [ ] Implement additional methods from paper comparisons or user-provided papers
- [ ] Compare all methods on both 2D and 3D data

### Phase 4: Multimodal Extension (future)
- [ ] Find aligned skeleton + IMU/inertial sensor datasets
- [ ] Design multimodal fusion architecture
- [ ] Federated multimodal training

---

## Project Setup — 2026-03-24

**Status:** Agent system initialized. Infrastructure ready (data shards, Flower strategy, ONNX export).

**Current state:**
- 2D STGCN++ trained with FedAvg (50 rounds) — outputs exist in `demo1_fed_skeleton/outputs/`
- ONNX pipeline working at 15-20 FPS on CPU
- 3D pipeline exists in pyskl configs but not used in FL yet
- FL modes implemented: FedAvg, FedBN, Clustered FedBN

**Next:** Run Phase 1 baselines systematically and record all metrics.

---
