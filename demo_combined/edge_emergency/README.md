# Smart-Care · Edge Emergency Detection

A live, self-contained demo: a **Raspberry Pi 5** watches a scene with its camera,
recognises medical actions on-device with a skeleton model, and raises **fall /
staggering emergency alerts** — streamed to a polished dashboard on a laptop over
the LAN. No cloud, no raw video leaves the device.

```
  Pi 5 camera ──► YOLOX-tiny (detect) ──► RTMPose (pose) ──► STGCN++ (action)
                                                                 │
                          severity + temporal-debounce alert engine
                                                                 │
        stdlib HTTP server  ── /stream.mjpg (video)  /events (SSE)  /  (dashboard)
                                                                 │
                                         Laptop browser  ◄────────┘
                                  http://<pi-ip>:8000/   (zero install)
```

The whole thing runs on **numpy + opencv + onnxruntime** only — no PyTorch, no
mmcv, no FastAPI. The Pi serves the dashboard itself, so the laptop just opens a
browser; nothing to install on the laptop and it survives a flaky network.

## What it shows

- **Live skeleton-overlaid video** with the current action + confidence.
- **Severity gauge** (NORMAL → LOW → MEDIUM → HIGH → CRITICAL).
- **Flashing emergency banner + audible alarm** on `falling` (CRITICAL) and
  `staggering` / `nausea` (HIGH), with a debounce so a single noisy frame never
  false-alarms in front of funders.
- **Top-5 class confidences**, throughput/latency telemetry, an emergency log,
  and the narrative badges (skeleton-only · federated-trained · on-device).

Model: **STGCN++ centralized 2D**, exported to ONNX
(`demo/onnx_models/stgcnpp_medical15_cent2d.onnx`). On the held-out test set:
**90.6% overall, falling 97.1% recall / 100% precision (0 false positives),
staggering 98.2%.**

## Quick test on a laptop (no Pi needed)

The pipeline needs `onnxruntime` + `opencv-python` in the project venv:

```powershell
uv pip install --python .venv\Scripts\python.exe onnxruntime opencv-python
```

Headless self-test on the bundled sample video (verifies models + alert logic):

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe demo_combined\edge_emergency\edge_node.py `
  --selftest 150 --video demo\ntu_sample.avi
```

Live with the laptop webcam + dashboard:

```powershell
demo_combined\edge_emergency\run_laptop.bat
# then open http://localhost:8000/
```

Or replay a video file into the dashboard (great for rehearsing a fall clip):

```powershell
.venv\Scripts\python.exe demo_combined\edge_emergency\edge_node.py `
  --video path\to\fall_clip.mp4 --loop --port 8000
```

## Deploy on the Raspberry Pi 5

1. Copy the repo (or at least `demo_combined/edge_emergency/`, `demo/demo_onnx.py`,
   `demo/onnx_models/*.onnx`, and `tools/data/label_map/medical_15.txt`) to the Pi.
2. One-time setup (installs numpy/opencv/onnxruntime into `.venv-edge`):
   ```bash
   bash demo_combined/edge_emergency/setup_pi.sh
   ```
3. Run it (Pi-tuned defaults: short-side 256, det-every 3, recog-every 6, 1 person):
   ```bash
   PY=.venv-edge/bin/python ./demo_combined/edge_emergency/run_pi.sh
   ```
4. On the laptop, open `http://<pi-ip>:8000/`. The node prints its LAN IP on start.

## Tuning for real-time on the Pi

Detection + pose dominate cost; recognition is cheap. Trade accuracy for speed:

| Flag | Effect | Demo default (Pi) |
|------|--------|-------------------|
| `--short-side` | downscale frames before detect/pose (biggest lever) | `256` |
| `--det-every N` | run person detection every N frames | `3` |
| `--recog-every N` | run STGCN++ every N frames (still ~2-3 Hz) | `6` |
| `--num-person` | person slots; `1` is fastest for single-subject | `1` |
| `--threads` | ONNX intra-op threads (Pi 5 has 4 cores) | `4` |
| `--jpeg-quality` | MJPEG quality vs. bandwidth | `80` |

Alert sensitivity:

| Flag | Meaning | Default |
|------|---------|---------|
| `--alert-conf` | min top-1 confidence to count as an emergency vote | `0.55` |
| `--alert-window` / `--alert-hits` | confirm if ≥hits of last window are emergencies | `5` / `3` |
| `--alert-hold` | seconds an alert stays latched after last sighting | `8.0` |

## Endpoints

| Route | Purpose |
|-------|---------|
| `/` | the dashboard (this folder's `dashboard.html`) |
| `/stream.mjpg` | annotated MJPEG video stream |
| `/events` | Server-Sent-Events feed of detections/alerts |
| `/state` | latest state as JSON (poll fallback) |
| `/healthz` | liveness check |

## Next steps (not yet done)

- **int8 quantization** of the ONNX models (`onnxruntime.quantization`) for a
  2-3× speedup on the Pi CPU — biggest remaining latency win.
- Optionally swap RTMPose-m → a lighter pose model (RTMPose-s/t) if pose is the
  bottleneck on the Pi (see `memory/reference-edge-pipeline-research.md`).
- Stage and record a controlled fall clip for a reliable rehearsed demo run.
