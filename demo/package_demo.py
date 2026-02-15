"""Package the ONNX demo into a portable folder that can run on any machine.

Usage (run from repo root on your PC):
    conda activate pyskl
    python demo/package_demo.py --output D:/onnx_demo_portable

Then copy the output folder to your laptop and run:
    cd onnx_demo_portable
    setup.bat          (creates conda env + installs deps)
    run_demo.bat       (launches the demo)
"""

import argparse
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

FILES_TO_COPY = [
    # Main demo script
    ("demo/demo_onnx.py", "demo_onnx.py"),
    # ONNX models
    ("demo/onnx_models/yolox_tiny.onnx", "onnx_models/yolox_tiny.onnx"),
    ("demo/onnx_models/rtmpose_m.onnx", "onnx_models/rtmpose_m.onnx"),
    (
        "demo/onnx_models/stgcnpp_ntu120_xsub_hrnet_j.onnx",
        "onnx_models/stgcnpp_ntu120_xsub_hrnet_j.onnx",
    ),
    # Label map
    ("tools/data/label_map/nturgbd_120.txt", "label_map/nturgbd_120.txt"),
    # Requirements
    ("demo/requirements_onnx.txt", "requirements.txt"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for portable package"
    )
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists():
        print(f"Output directory already exists: {out}")
        resp = input("Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return
        shutil.rmtree(out)

    out.mkdir(parents=True)

    # Copy files
    print("Packaging files...")
    total_size = 0
    for src_rel, dst_rel in FILES_TO_COPY:
        src = REPO_ROOT / src_rel
        dst = out / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {src_rel} -> {dst_rel}  ({src.stat().st_size / 1e6:.1f} MB)")
        shutil.copy2(src, dst)
        total_size += src.stat().st_size

    # Patch demo_onnx.py default paths (since folder structure is different)
    demo_py = out / "demo_onnx.py"
    text = demo_py.read_text(encoding="utf-8")
    text = text.replace('default="demo/onnx_models"', 'default="onnx_models"')
    text = text.replace(
        'default="tools/data/label_map/nturgbd_120.txt"',
        'default="label_map/nturgbd_120.txt"',
    )
    demo_py.write_text(text, encoding="utf-8")

    # Write setup script (Windows)
    setup_bat = out / "setup.bat"
    setup_bat.write_text(
        r"""@echo off
echo ==========================================
echo   ONNX Action Recognition Demo - Setup
echo ==========================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda not found. Please install Miniconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Create environment
echo Creating conda environment 'onnx_demo' with Python 3.10...
call conda create -n onnx_demo python=3.10 -y
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment.
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
call conda activate onnx_demo
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo   Setup complete!
echo   Run: run_demo.bat
echo ==========================================
pause
""",
        encoding="utf-8",
    )

    # Write setup script (Linux/Mac)
    setup_sh = out / "setup.sh"
    setup_sh.write_text(
        """#!/bin/bash
echo "=========================================="
echo "  ONNX Action Recognition Demo - Setup"
echo "=========================================="
echo

# Check conda
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Creating conda environment 'onnx_demo' with Python 3.10..."
conda create -n onnx_demo python=3.10 -y

echo
echo "Installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate onnx_demo
pip install -r requirements.txt

echo
echo "=========================================="
echo "  Setup complete!"
echo "  Run: bash run_demo.sh"
echo "=========================================="
""",
        encoding="utf-8",
    )

    # Write run scripts
    run_bat = out / "run_demo.bat"
    run_bat.write_text(
        r"""@echo off
echo Starting ONNX Action Recognition Demo (CPU)...
echo Press Q in the video window to quit.
echo.
call conda activate onnx_demo
python demo_onnx.py --device cpu --short-side 480 --window-frames 30 --det-score-thr 0.5 %*
pause
""",
        encoding="utf-8",
    )

    run_gpu_bat = out / "run_demo_gpu.bat"
    run_gpu_bat.write_text(
        r"""@echo off
echo Starting ONNX Action Recognition Demo (GPU)...
echo Press Q in the video window to quit.
echo.
echo NOTE: GPU mode requires onnxruntime-gpu and CUDA/cuDNN.
echo If it fails, use run_demo.bat for CPU mode instead.
echo.
call conda activate onnx_demo
pip install onnxruntime-gpu nvidia-cudnn-cu12 2>nul
python demo_onnx.py --device cuda --recog-device cpu --short-side 480 --window-frames 30 --det-score-thr 0.5 %*
pause
""",
        encoding="utf-8",
    )

    run_sh = out / "run_demo.sh"
    run_sh.write_text(
        """#!/bin/bash
echo "Starting ONNX Action Recognition Demo (CPU)..."
echo "Press Q in the video window to quit."
echo
eval "$(conda shell.bash hook)"
conda activate onnx_demo
python demo_onnx.py --device cpu --short-side 480 --window-frames 30 --det-score-thr 0.5 "$@"
""",
        encoding="utf-8",
    )

    # Write README
    readme = out / "README.md"
    readme.write_text(
        f"""# Real-Time Skeleton Action Recognition (ONNX Demo)

Pure-ONNX pipeline for real-time skeleton-based action recognition.
**No mmcv, mmpose, mmdet, or pyskl needed** — just numpy, opencv, and onnxruntime.

## Quick Start

### 1. Setup (one-time)

**Windows:**
```
setup.bat
```

**Linux/Mac:**
```bash
bash setup.sh
```

This creates a conda environment `onnx_demo` with Python 3.10 and installs dependencies.

### 2. Run

**Windows (CPU):**
```
run_demo.bat
```

**Windows (GPU — requires NVIDIA GPU + CUDA 12):**
```
run_demo_gpu.bat
```

**Linux/Mac:**
```bash
bash run_demo.sh
```

### 3. Custom options

```bash
conda activate onnx_demo

# Webcam (default)
python demo_onnx.py --device cpu

# Video file
python demo_onnx.py --device cpu --clip path/to/video.mp4

# Lower resolution = faster
python demo_onnx.py --device cpu --short-side 320

# Skip detection every other frame
python demo_onnx.py --device cpu --det-every 2

# Benchmark mode
python demo_onnx.py --device cpu --benchmark
```

## Pipeline

```
Webcam → YOLOX-Tiny (detect persons) → RTMPose-m (17 keypoints)
       → STGCN++ (recognize action from skeleton) → 120 NTU RGB+D classes
```

## Models ({total_size / 1e6:.0f} MB total)

| Model | File | Size | Purpose |
|-------|------|------|---------|
| YOLOX-Tiny | yolox_tiny.onnx | 19 MB | Person detection |
| RTMPose-m | rtmpose_m.onnx | 52 MB | Pose estimation (17 COCO keypoints) |
| STGCN++ | stgcnpp_ntu120_xsub_hrnet_j.onnx | 5 MB | Action recognition (120 classes) |

## Performance

| Device | FPS | Notes |
|--------|-----|-------|
| CPU only | ~15-18 | Works on any laptop |
| GPU (RTX 3070 Ti) | ~25 | Needs CUDA 12 + cuDNN 9 |

## Requirements

- Python 3.8+ (3.10 recommended)
- numpy, opencv-python, onnxruntime
- Webcam (or video file with --clip)
""",
        encoding="utf-8",
    )

    print(f"\n{'='*50}")
    print(f"  Package created: {out}")
    print(f"  Total size: {total_size / 1e6:.0f} MB")
    print(f"  Files: {len(FILES_TO_COPY) + 6}")
    print(f"{'='*50}")
    print(f"\nCopy this folder to your laptop and run:")
    print(f"  cd {out.name}")
    print(f"  setup.bat")
    print(f"  run_demo.bat")


if __name__ == "__main__":
    main()
