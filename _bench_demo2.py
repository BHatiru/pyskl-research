"""Benchmark Demo 2 inference on ntu_sample.avi (non-interactive, no GUI).

Runs the full det→pose→recognition pipeline on a batch of frames extracted
from the sample video and prints timing breakdowns.

Usage:
    conda run -n pyskl python _bench_demo2.py            # default (Faster-RCNN + HRNet)
    conda run -n pyskl python _bench_demo2.py --fast      # fast preset (YOLOX-tiny + VIPNas)
"""
import argparse
import time
import cv2
import numpy as np
import torch
import warnings
import sys
import os

# Suppress noisy warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# ── Extract frames from sample video ─────────────────────────────────────────
def extract_frames(video_path, max_frames=30, short_side=320):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        sys.exit(1)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if min(h, w) > short_side:
            scale = short_side / min(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        frames.append(frame)
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path} "
          f"(resized to short_side={short_side})")
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="demo/ntu_sample.avi")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--skip-frames", type=int, default=None)
    p.add_argument("--short-side", type=int, default=None)
    args = p.parse_args()

    # Build the args namespace that ActionRecognitionEngine expects
    from types import SimpleNamespace
    demo_args = SimpleNamespace(
        mode="clip",
        video=args.video,
        camera=0,
        config="configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py",
        checkpoint=("http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/"
                     "stgcnpp_ntu120_xsub_hrnet/j.pth"),
        detector="faster-rcnn",
        det_config=None,
        det_checkpoint=None,
        pose_model="hrnet-w32",
        pose_config=None,
        pose_checkpoint=None,
        det_score_thr=0.9,
        label_map="tools/data/label_map/nturgbd_120.txt",
        device="cpu",
        clip_len=args.max_frames,
        clip_stride=15,
        record_seconds=3.0,
        short_side=320,
        skip_frames=1,
        capture_fps=15,
        fast=False,
        fp16=False,
        no_fp16=True,
        no_vis=False,
        warmup=False,
    )

    # Apply --fast preset
    if args.fast:
        demo_args.detector = "yolox-tiny"
        demo_args.pose_model = "vipnas-mbv3"
        demo_args.skip_frames = 2
        demo_args.short_side = 256
        demo_args.det_score_thr = 0.5
        demo_args.fast = True

    # Manual overrides
    if args.skip_frames is not None:
        demo_args.skip_frames = args.skip_frames
    if args.short_side is not None:
        demo_args.short_side = args.short_side

    mode_label = "FAST" if demo_args.fast else "DEFAULT"
    print(f"\n{'='*60}")
    print(f"  Demo 2 Benchmark — {mode_label} mode")
    print(f"  Detector:   {demo_args.detector}")
    print(f"  Pose model: {demo_args.pose_model}")
    print(f"  Skip frames: {demo_args.skip_frames}")
    print(f"  Short side: {demo_args.short_side}")
    print(f"  Device: {demo_args.device}")
    print(f"{'='*60}\n")

    # Extract frames
    frames = extract_frames(args.video, args.max_frames, demo_args.short_side)
    if demo_args.skip_frames > 1:
        frames = frames[::demo_args.skip_frames]
        print(f"After skip_frames={demo_args.skip_frames}: {len(frames)} frames")

    # Import and build engine
    print("\nLoading models...")
    t_load = time.time()
    from demo.demo_realtime import ActionRecognitionEngine
    engine = ActionRecognitionEngine(demo_args)
    dt_load = time.time() - t_load
    print(f"Model loading: {dt_load:.1f}s\n")

    # Run inference
    print("Running inference...")
    label, top5, dt, vis_frames = engine.infer(frames, skip_vis=demo_args.no_vis)

    print(f"\n{'='*60}")
    print(f"  RESULT: {label}")
    for i, (name, score) in enumerate(top5[:5]):
        print(f"    {i+1}. {name}: {score:.4f}")
    print(f"\n  Total inference time: {dt:.2f}s")
    print(f"  Effective FPS: {len(frames)/dt:.2f}")
    print(f"  Model load time: {dt_load:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
