---
applyTo: "pyskl/models/**/*.py"
description: "Conventions for pyskl model code — registry pattern, GCN architecture, config-driven design"
---

# Pyskl Model Conventions

## Registry Pattern
- All backbone classes must be decorated with `@BACKBONES.register_module()`
- Head classes use `@HEADS.register_module()`
- Import builder: `from ..builder import BACKBONES` (or `HEADS`)
- Config references models by `type` string: `dict(type='STGCN', ...)`

## Architecture Patterns
- GCN blocks follow the pattern: `unit_gcn` (spatial) → `unit_tcn` (temporal) → residual + ReLU
- Typical channel progression: 64→64→64→128→128→256 (6 blocks for lightweight) or 10 blocks for full
- Graph adjacency: constructed from layout (e.g., `coco`, `nturgb+d`) with mode `spatial`
- Adaptive adjacency: learnable `PA` parameter added to static adjacency matrix
- Multi-scale TCN: parallel dilated convolutions (dilations 1,2,3,4) + maxpool branch + 1×1 conv

## Self-contained FL Model
- `demo1_fed_skeleton/models/stgcn.py` is standalone — no mmcv/pyskl imports
- Uses `build_model(num_classes, in_channels, ...)` factory function
- Graph defined inline (COCO-17 adjacency)
- This is the model used in FL training — keep it independent

## Config Conventions (Python configs, not YAML)
```python
model = dict(
    type='RecognizerGCN',
    backbone=dict(type='STGCN', gcn_adaptive='init', gcn_with_res=True, tcn_type='mstcn',
                  graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=120, in_channels=256))
```
- Pipeline: PreNormalize3D → GenSkeFeat → UniformSample(clip_len=100) → FormatGCNInput(num_person=2)
- Optimizer: SGD with lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True
- LR schedule: CosineAnnealing to 0

## When Adding New Models
1. Create module file in appropriate subpackage under `pyskl/models/`
2. Register with `@BACKBONES.register_module()` or appropriate registry
3. Add import in `__init__.py` of the subpackage
4. Create config file(s) in `configs/<model_name>/` following naming: `<model>_<dataset>_<split>_<skeleton>/`
5. For FL use: also create standalone version in `demo1_fed_skeleton/models/` without mmcv deps
