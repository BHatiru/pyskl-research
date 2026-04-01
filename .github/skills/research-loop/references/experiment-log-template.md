# Experiment Log Entry Template

Use this format when logging results to `experiments/log.md`.

```markdown
## Experiment: <short-name> — <YYYY-MM-DD>

**Hypothesis:** <testable statement>

**Setup:**
- Model: <architecture and params>
- FL mode: <fedavg/fedbn/cluster/centralized>
- Rounds: <N> | Epochs/round: <N> | LR: <X>
- Clients: <N> | Alpha: <X> | Seed: <X>
- Changes from baseline: <what's different>

**Results:**

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Overall Accuracy | X% | +Y% |
| Fall Detection F1 | X.XX | +Y.YY |
| Worst Client Acc | X% | +Y% |
| Mean Client Acc | X% | +Y% |
| Training Time | Xm | |

**Per-Class F1:**
| Class | F1 | Notes |
|-------|-----|-------|
| falling | X.XX | |
| staggering | X.XX | |
| ... | ... | |

**Conclusion:** <PASS/FAIL/PARTIAL> — <1-2 sentence summary>

**Next Step:** <what to try based on these results>
```
