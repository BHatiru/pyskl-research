---
applyTo: "**/*.ipynb"
description: "Conventions for research notebooks — reproducibility, experiment tracking, Colab compatibility"
---

# Research Notebook Conventions

## Structure
- First cell: environment setup (GPU check, pip installs, drive mount for Colab)
- Second cell: imports and config constants (hyperparams as named variables, not magic numbers)
- Clear markdown headers between logical sections (Data, Model, Training, Evaluation, Analysis)
- Final cell: summary of results and next steps

## Reproducibility
- Set random seeds (torch, numpy, random) in a dedicated cell near the top
- Log all hyperparameters in a dictionary or dataclass before training starts
- Print dataset statistics (samples per class, train/test split sizes) before training
- Save model checkpoints with descriptive names including key hyperparams

## Experiment Tracking
- Every training run should print: date, config dict, final metrics (loss, accuracy, per-class F1)
- For FL experiments: log per-round metrics (global_acc, mean_client_acc, worst_client_acc)
- When comparing experiments, create a summary table in a markdown cell
- Save results to `experiments/log.md` in the workspace when significant

## Colab Patterns
- Check GPU availability: `torch.cuda.is_available()` and print device name
- Mount Drive only when needed: `from google.colab import drive; drive.mount('/content/drive')`
- Install packages with `!pip install -q <package>` (quiet flag to reduce noise)
- Use `%%time` magic for training cells to track wall-clock time

## Data Format (this project)
- Skeleton tensors: `(N, M, T, V, C)` = (batch, 2 persons, 100 frames, 17 COCO joints, 3 channels)
- Channels: (x, y, confidence_score) normalized to [-1, 1]
- Labels: integer class index (0-9 for medical subset)
- Always validate tensor shapes after loading: `assert x.shape[1:] == (2, 100, 17, 3)`
