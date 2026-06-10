#!/usr/bin/env bash
# Smart-Care Edge Emergency Node - Raspberry Pi 5 launcher.
# Runs the node against the Pi camera and serves the dashboard on :8000.
set -e
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8

PY="${PY:-python3}"

# Pi-tuned defaults. MoveNet-Thunder single-stage front-end is ~4x faster than
# YOLOX+RTMPose on CPU and was validated to preserve fall detection (URFD).
# NOTE: --num-person MUST be 2 — the cent2d STGCN++ ONNX has a fixed M=2 input
# (MoveNet emits 1 person; the recognizer zero-pads the 2nd slot).
# To fall back to the proven two-stage path: pass --pose-backend rtmpose.
# Override any flag on the command line, e.g. --short-side 224.
exec "$PY" demo_combined/edge_emergency/edge_node.py \
  --camera 0 \
  --pose-backend movenet \
  --short-side 256 \
  --recog-every 6 \
  --num-person 2 \
  --threads 4 \
  --port 8000 \
  "$@"
