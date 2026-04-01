---
description: "Generate a ready-to-run Colab training cell from experiment specs. Use when you need a complete, paste-ready notebook cell for training."
agent: "experimenter"
argument-hint: "Describe the training setup: model, data, hyperparams, FL mode"
---

Generate a complete, ready-to-paste Colab notebook cell for the specified training configuration.

## Cell Requirements

1. **Markdown header cell** explaining what this training run does
2. **Config cell** with all hyperparameters as named variables at the top:
   - Model params (num_classes, in_channels, etc.)
   - Training params (lr, epochs, batch_size, optimizer)
   - FL params (num_rounds, mode, num_clients, alpha)
   - Random seed (default: 42)
3. **Training cell** with:
   - Device setup and GPU check
   - Data loading with shape assertions: `assert x.shape[1:] == (2, 100, 17, 3)`
   - Model instantiation
   - Training loop with per-epoch/round metric logging
   - `%%time` magic for wall-clock tracking
4. **Evaluation cell** with:
   - Test set evaluation
   - Per-class metrics (classification_report)
   - Confusion matrix computation
   - Results summary dict printed at the end

Follow the patterns in `demo1_fed_skeleton/train_colab_medical.ipynb`.
Skeleton format: `(N, M, T, V, C)` = `(batch, 2, 100, 17, 3)`.
Medical classes (10): falling, staggering, touch head/chest/back/neck, nausea, standing up, sitting down, walking.
