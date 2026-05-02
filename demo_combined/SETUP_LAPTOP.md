# Laptop Setup — Medical HAR Dashboard

This dashboard does **not** require the full PYSKL/MMCV stack. It only needs Python, PyTorch, Streamlit, Plotly, the lightweight STGCN++ code, and the trained demo artifacts.

## 1. Clone and switch branch

```powershell
git clone https://github.com/BHatiru/pyskl-research.git
cd pyskl-research
git checkout experiments
```

If the repo already exists:

```powershell
cd pyskl-research
git fetch origin
git checkout experiments
git pull origin experiments
```

## 2. Create a virtual environment

Use Python 3.10 if available.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Install dashboard packages

CPU/default install:

```powershell
pip install -r demo_combined\requirements_dashboard.txt
```

For an NVIDIA laptop where CUDA PyTorch is desired, install PyTorch from the official command first, then install the rest:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy streamlit plotly
```

## 4. Required local artifacts

The dashboard needs these files in addition to the git-tracked code/results:

```text
demo1_fed_skeleton\outputs\cent_2d_15cls_best.pt      (~1.9 MB)
demo1_fed_skeleton\outputs\cent_3d_15cls_best.pt      (~2.0 MB)
demo1_fed_skeleton\data\fed_medical_2d\centralized\test.npz  (~37 MB)
demo1_fed_skeleton\data\fed_medical_3d\centralized\test.npz  (~70 MB)
```

These are local training/data artifacts and may not be present after a fresh clone. Fastest path: copy them from the lab PC into the same relative paths on the laptop.

If the lab PC provides `demo_combined\dashboard_artifacts_for_laptop.zip`, copy that zip to the laptop repo root and extract it there:

```powershell
Expand-Archive -Path demo_combined\dashboard_artifacts_for_laptop.zip -DestinationPath . -Force
```

The archive preserves the required repository-relative paths, so no manual file placement is needed after extraction.

Verify them with:

```powershell
$files = @(
  'demo1_fed_skeleton\outputs\cent_2d_15cls_best.pt',
  'demo1_fed_skeleton\outputs\cent_3d_15cls_best.pt',
  'demo1_fed_skeleton\data\fed_medical_2d\centralized\test.npz',
  'demo1_fed_skeleton\data\fed_medical_3d\centralized\test.npz'
)
foreach ($f in $files) { if (Test-Path $f) { "OK  $f" } else { "MISS $f" } }
```

If missing and copying is not possible, regenerate data with:

```powershell
python demo1_fed_skeleton\prepare_medical_data.py --num-clients 5 --alpha 0.5
```

Then train or copy checkpoints. Training from scratch is not needed for a demo if checkpoints are copied.

## 5. Run smoke-test inference

```powershell
python demo_combined\inference.py --mode 2d --device cpu --n 3
python demo_combined\inference.py --mode 3d --device cpu --n 3
```

Use `--device cuda` only if CUDA PyTorch is installed and working.

## 6. Launch dashboard

```powershell
python -m streamlit run demo_combined\dashboard_app.py --server.port 8501
```

Or on Windows after creating `.venv`:

```powershell
.\demo_combined\run_dashboard.bat
```

Open:

```text
http://localhost:8501
```

## Common fixes

- **No `.venv`**: create it using section 2; the committed repo does not include virtual environments.
- **Missing `.pt` or `.npz` files**: copy artifacts from the lab PC or regenerate data; see section 4.
- **CUDA errors**: run with CPU from the sidebar or use `--device cpu` in inference tests.
- **Plotly/Streamlit import errors**: rerun `pip install -r demo_combined\requirements_dashboard.txt` inside the activated `.venv`.
