---
name: research-loop
description: "Full research cycle for ML experiments. Use when: starting a new research iteration, reviewing experiment results and planning next steps, running the complete research-plan-implement-analyze cycle. Invoke with /research-loop."
argument-hint: "Describe the research goal or provide results to analyze"
---

# Research Loop Skill

Orchestrates a full research cycle: review current state → identify opportunities → propose experiment → generate artifacts → define success criteria.

## When to Use
- Starting a new research direction
- After getting experiment results and needing to plan the next iteration
- When you want the full research-plan-implement-analyze cycle

## Procedure

### Step 1: Review Current State
1. Read [experiments/log.md](../../../experiments/log.md) for all prior results
2. Read [demo1_fed_skeleton/demo1_notes.md](../../../demo1_fed_skeleton/demo1_notes.md) for current setup details
3. Identify the current best metrics and known failure modes
4. Summarize: "We are at X% accuracy, Y fall F1, worst client at Z%"

### Step 2: Identify Improvement Opportunities
Use the **scout** agent to search for methods addressing identified weaknesses:
- If fall/stagger confusion is high → search for discriminative skeleton features
- If worst-client accuracy is low → search for personalization/fairness methods in FL
- If overall accuracy plateaus → search for architectural improvements (attention, multi-scale)
- If training is unstable → search for FL optimization methods (FedProx, SCAFFOLD)

Reference the [current baselines](references/fl-baselines.md) for comparison.

### Step 3: Formulate Experiment
Use the **experimenter** agent to produce:
- Experiment card (hypothesis, variable, control, success criteria)
- Training configs or notebook cells
- Ablation grid if testing multiple variants

### Step 4: Implementation
Use the **implementer** agent for any code changes needed:
- New model components → `demo1_fed_skeleton/models/`
- New FL strategies → `demo1_fed_skeleton/fl/`
- Data processing → `demo1_fed_skeleton/prepare_clients.py`

### Step 5: Generate Artifacts
Produce ready-to-use outputs:
- Colab notebook cells (paste into `train_colab_medical.ipynb`)
- Updated training commands
- Success criteria checklist

### Step 6: Checkpoint
Present the complete plan to the user for approval before execution.

### Step 7: Post-Experiment Analysis
After results are available, use the **analyst** agent to:
- Parse and compare metrics
- Generate visualizations
- Update [experiments/log.md](../../../experiments/log.md)
- Propose next iteration

## References
- [FL Baselines](references/fl-baselines.md) — Current best metrics to compare against
- [Experiment Log Template](references/experiment-log-template.md) — Format for logging results
