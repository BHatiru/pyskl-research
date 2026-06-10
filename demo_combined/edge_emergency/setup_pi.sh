#!/usr/bin/env bash
# One-time setup for the Smart-Care Edge node on a Raspberry Pi 5 (64-bit OS).
# Installs the minimal runtime: numpy + opencv + onnxruntime. No torch/mmcv.
set -e

echo ">> Smart-Care Edge - Raspberry Pi setup"

# System packages: OpenCV runtime libs + Python venv tooling.
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0 \
                        libatlas-base-dev

# Project-local virtual environment.
cd "$(dirname "$0")/../.."
if [ ! -d ".venv-edge" ]; then
  python3 -m venv .venv-edge
fi
source .venv-edge/bin/activate
python -m pip install --upgrade pip

# Runtime deps. onnxruntime ships aarch64 wheels for Pi 5 (ARMv8).
# ai-edge-litert runs the MoveNet TFLite pose backend (~4x faster front-end on
# CPU than YOLOX+RTMPose, validated to preserve fall detection).
pip install numpy "opencv-python-headless<5" onnxruntime ai-edge-litert

echo ""
echo ">> Done. Verify the required model files are present:"
for f in demo/onnx_models/movenet_thunder_int8.tflite \
         demo/onnx_models/stgcnpp_medical15_cent2d.onnx \
         demo/onnx_models/yolox_tiny.onnx \
         demo/onnx_models/rtmpose_m.onnx \
         tools/data/label_map/medical_15.txt; do
  if [ -f "$f" ]; then echo "   OK   $f"; else echo "   MISS $f  <-- copy this from the lab PC"; fi
done

echo ""
echo ">> Run the demo:  PY=.venv-edge/bin/python ./demo_combined/edge_emergency/run_pi.sh"
echo ">> Then open      http://<pi-ip>:8000/  on the laptop."
