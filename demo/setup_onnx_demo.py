"""Download ONNX models and set up environment for the demo.

Run this once after cloning/pulling the repo on a new machine:

    conda create -n onnx_demo python=3.10 -y
    conda activate onnx_demo
    pip install -r demo/requirements_onnx.txt
    python demo/setup_onnx_demo.py
    python demo/demo_onnx.py --device cpu
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "onnx_models"

MODELS = {
    "yolox_tiny.onnx": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        "inner_glob": "**/end2end.onnx",
        "size_mb": 19,
        "desc": "YOLOX-Tiny person detector (HumanArt+COCO, 416x416)",
    },
    "rtmpose_m.onnx": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        "inner_glob": "**/*.onnx",
        "size_mb": 52,
        "desc": "RTMPose-m pose estimator (body7, SimCC, 256x192)",
    },
    "stgcnpp_ntu120_xsub_hrnet_j.onnx": {
        "url": None,  # Exported locally — see instructions below
        "size_mb": 5,
        "desc": "STGCN++ action recognizer (NTU120, 120 classes)",
    },
}

STGCNPP_EXPORT_CMD = (
    "python tools/export_stgcnpp_onnx.py "
    '--config "configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py" '
    "--checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth "
    "--output demo/onnx_models/stgcnpp_ntu120_xsub_hrnet_j.onnx"
)


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

    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False

    # Extract the .onnx file from the zip
    print(f"  ↳ Extracting...")
    import glob

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


def main():
    print("=" * 55)
    print("  ONNX Demo Setup — Download Models")
    print("=" * 55)
    print()

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

    print("\n" + "=" * 55)
    if all_ok:
        print("  All models ready! Run the demo:")
        print("    python demo/demo_onnx.py --device cpu")
    else:
        print("  Some models missing — see instructions above.")
    print("=" * 55)


if __name__ == "__main__":
    main()
