---
description: "Generate a structured experiment report from training results. Provide metrics, logs, or point to output files."
agent: "analyst"
argument-hint: "Paste training metrics or describe where results are stored"
---

Analyze the provided experiment results and generate a structured report.

## Required Output

1. **Summary card** (experiment name, date, hypothesis, pass/fail)
2. **Metrics comparison table** vs baseline (from `experiments/log.md`)
3. **Per-class breakdown** (precision, recall, F1 for each of the 10 medical classes)
4. **Confusion matrix insights** (top misclassifications)
5. **FL fairness** (per-client accuracy, worst-client gap)
6. **Visualization code** (matplotlib cells for training curves, confusion matrix, per-class F1)
7. **Top 3 actionable next steps** ranked by expected impact
8. **Append to experiment log** (`experiments/log.md`)

Prioritize **fall detection recall** and **worst-client fairness** as the most important metrics for this medical application.
