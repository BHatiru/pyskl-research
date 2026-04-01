---
description: "Design an ablation study to isolate the effect of specific model components or hyperparameters."
agent: "experimenter"
argument-hint: "Describe the component(s) to ablate and the baseline to compare against"
---

Design a controlled ablation study for the specified component(s).

## Requirements

1. **Ablation grid**: Table of runs varying ONE factor at a time
2. **Baseline run**: Clearly defined reference configuration
3. **For each variant**: Config changes, expected effect, training command or Colab cell
4. **Metrics to track**: Overall accuracy, fall F1, worst-client accuracy, training time
5. **Analysis plan**: How to determine which component matters most
6. **Colab cells**: Ready-to-paste cells for each run in the ablation

Follow the existing config patterns in `configs/stgcn++/` and `demo1_fed_skeleton/train_federated.py`.
Keep training time per run under 1 hour on a Colab T4 GPU.
Use fixed random seeds (42) across all runs for fair comparison.
