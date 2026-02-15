"""Download models and set up environment for the demos.

Supports two pipelines:

  ── ONNX pipeline (lightweight, ~76 MB models, CPU-friendly) ──
    conda create -n onnx_demo python=3.10 -y && conda activate onnx_demo
    pip install -r demo/requirements_onnx.txt
    python demo/setup_onnx_demo.py
    python demo/demo_onnx.py --device cpu

  ── Full mmcv pipeline (heavier, needs PyTorch + CUDA) ──
    conda create -n pyskl python=3.8 -y && conda activate pyskl
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113
    pip install -r demo/requirements_full.txt
    pip install -e .
    python demo/setup_onnx_demo.py --full
    python demo/demo_realtime.py --mode stream

Usage:
    python demo/setup_onnx_demo.py          # ONNX models only (default)
    python demo/setup_onnx_demo.py --full    # ONNX + mmcv pipeline models
    python demo/setup_onnx_demo.py --check   # Check which models are present
"""

import argparse
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
MODEL_DIR = SCRIPT_DIR / "onnx_models"

MODELS = {
    "yolox_tiny.onnx": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        "inner_glob": "**/end2end.onnx",
        "size_mb": 19,
        "desc": "YOLOX-Tiny person detector (HumanArt+COCO, 416x416)",
        "pipeline": "onnx",
    },
    "rtmpose_m.onnx": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        "inner_glob": "**/*.onnx",
        "size_mb": 52,
        "desc": "RTMPose-m pose estimator (body7, SimCC, 256x192)",
        "pipeline": "onnx",
    },
    "stgcnpp_ntu120_xsub_hrnet_j.onnx": {
        "url": None,  # Exported locally — see instructions below
        "size_mb": 5,
        "desc": "STGCN++ action recognizer (NTU120, ONNX format)",
        "pipeline": "onnx",
    },
}

# ── Full-pipeline model checkpoints (PyTorch, auto-downloaded by mmdet/mmpose) ──
FULL_MODELS = {
    "faster_rcnn_r50_fpn_1x_coco-person.pth": {
        "url": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        "dest_dir": MODEL_DIR,
        "size_mb": 160,
        "desc": "Faster-RCNN R50 person detector (COCO, for demo_realtime.py)",
    },
    "hrnet_w32_coco_256x192.pth": {
        "url": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
        "dest_dir": MODEL_DIR,
        "size_mb": 110,
        "desc": "HRNet-w32 pose estimator (COCO, 256x192)",
    },
    "stgcnpp_ntu120_xsub_hrnet_j.pth": {
        "url": "http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth",
        "dest_dir": MODEL_DIR,
        "size_mb": 6,
        "desc": "STGCN++ action recognizer (NTU120, PyTorch checkpoint)",
    },
}

STGCNPP_EXPORT_CMD = (
    "python tools/export_stgcnpp_onnx.py "
    '--config "configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py" '
    "--checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth "
    "--output demo/onnx_models/stgcnpp_ntu120_xsub_hrnet_j.onnx"
)


def _progress_hook(block_num, block_size, total_size):
    """Simple download progress indicator."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r    {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)


def download_file(url, dest):
    """Download a single file with progress."""
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print()  # newline after progress
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False


def download_and_extract(name, info):
    """Download a zipped ONNX model and extract the .onnx file."""
    dest = MODEL_DIR / name
    if dest.exists():
        print(f"  ✓ {name} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    url = info.get("url")
    if url is None:
        return False  # needs special handling

    print(f"  ↓ Downloading {name} (~{info['size_mb']} MB)...")
    zip_path = MODEL_DIR / f"{name}.zip"

    if not download_file(url, zip_path):
        return False

    # Extract the .onnx file from the zip
    print(f"  ↳ Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        onnx_files = [f for f in zf.namelist() if f.endswith(".onnx")]
        if not onnx_files:
            print(f"  ✗ No .onnx file found in zip!")
            zip_path.unlink()
            return False

        # Extract the first .onnx file
        onnx_name = onnx_files[0]
        with zf.open(onnx_name) as src, open(dest, "wb") as dst:
            dst.write(src.read())

    zip_path.unlink()
    print(f"  ✓ {name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def download_checkpoint(name, info):
    """Download a PyTorch checkpoint directly (no zip)."""
    dest_dir = info.get("dest_dir", MODEL_DIR)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name

    if dest.exists():
        print(f"  ✓ {name} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    print(f"  ↓ Downloading {name} (~{info['size_mb']} MB)...")
    if not download_file(info["url"], dest):
        return False

    print(f"  ✓ {name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def export_stgcnpp():
    """Export STGCN++ to ONNX (requires pyskl + mmcv + torch)."""
    dest = MODEL_DIR / "stgcnpp_ntu120_xsub_hrnet_j.onnx"
    if dest.exists():
        print(f"  ✓ stgcnpp already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    print("  → STGCN++ needs to be exported from PyTorch.")
    print("    This requires the full pyskl environment (torch, mmcv, etc).")
    print(f"    Run:\n      {STGCNPP_EXPORT_CMD}")

    # Try to run it automatically
    try:
        import torch
        import mmcv

        print("  → Dependencies found, exporting automatically...")
        ret = os.system(STGCNPP_EXPORT_CMD)
        return ret == 0
    except ImportError:
        print("  → torch/mmcv not available. Export on a machine with pyskl installed,")
        print("    then copy stgcnpp_ntu120_xsub_hrnet_j.onnx to demo/onnx_models/")
        return False


def check_models():
    """Print status of all models."""
    print("\n  ── ONNX pipeline models ──")
    for name, info in MODELS.items():
        dest = MODEL_DIR / name
        if dest.exists():
            print(f"  ✓ {name:45s} {dest.stat().st_size / 1e6:6.1f} MB")
        else:
            print(f"  ✗ {name:45s} (missing)")

    print("\n  ── Full pipeline models (mmcv) ──")
    for name, info in FULL_MODELS.items():
        dest_dir = Path(info.get("dest_dir", MODEL_DIR))
        dest = dest_dir / name
        if dest.exists():
            print(f"  ✓ {name:45s} {dest.stat().st_size / 1e6:6.1f} MB")
        else:
            print(f"  ✗ {name:45s} (missing)")


def setup_onnx(skip_stgcn_export=False):
    """Download/extract ONNX pipeline models."""
    MODEL_DIR.mkdir(exist_ok=True)
    all_ok = True
    for name, info in MODELS.items():
        print(f"\n[{info['desc']}]")
        if info["url"]:
            ok = download_and_extract(name, info)
        else:
            ok = export_stgcnpp()
        if not ok:
            all_ok = False
    return all_ok


def setup_full():
    """Download PyTorch checkpoints for the mmcv-based pipeline."""
    MODEL_DIR.mkdir(exist_ok=True)
    all_ok = True
    for name, info in FULL_MODELS.items():
        print(f"\n[{info['desc']}]")
        ok = download_checkpoint(name, info)
        if not ok:
            all_ok = False

    # Check pyskl is installed
    print("\n[pyskl package]")
    try:
        import pyskl

        print(f"  ✓ pyskl {pyskl.__version__} is installed")
    except ImportError:
        print("  ✗ pyskl not installed — run: pip install -e .")
        all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download models for real-time action recognition demos"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also download PyTorch checkpoints for demo_realtime.py",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check which models are present, don't download",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Only download full-pipeline models (skip ONNX)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Real-time Action Recognition Demo — Model Setup")
    print("=" * 60)

    if args.check:
        check_models()
        print()
        return

    onnx_ok = True
    full_ok = True

    if not args.full_only:
        print("\n┌─────────────────────────────────┐")
        print("│  ONNX Pipeline (demo_onnx.py)   │")
        print("└─────────────────────────────────┘")
        onnx_ok = setup_onnx()

    if args.full or args.full_only:
        print("\n┌──────────────────────────────────────┐")
        print("│  Full Pipeline (demo_realtime.py)    │")
        print("└──────────────────────────────────────┘")
        full_ok = setup_full()

    # ── Summary ──
    print("\n" + "=" * 60)
    if not args.full_only:
        if onnx_ok:
            print("  ✓ ONNX pipeline ready:")
            print("      python demo/demo_onnx.py --device cpu")
        else:
            print("  ✗ ONNX pipeline — some models missing (see above)")

    if args.full or args.full_only:
        if full_ok:
            print("  ✓ Full pipeline ready:")
            print("      python demo/demo_realtime.py --mode stream")
        else:
            print("  ✗ Full pipeline — some models/packages missing (see above)")

    if not (args.full or args.full_only):
        print("\n  Tip: run with --full to also set up the mmcv-based demo")

    print("=" * 60)


if __name__ == "__main__":
    main()
