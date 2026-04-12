"""Headless inference test — runs full ONNX pipeline on a video file.

Usage:
    python demo/test_inference.py --clip demo/ntu_sample.avi
    python demo/test_inference.py --clip demo/ntu_sample.avi --output demo/test_output.mp4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Import pipeline components from demo_onnx
sys.path.insert(0, str(Path(__file__).parent))
from demo_onnx import (
    YOLOXDetector,
    RTMPoseEstimator,
    STGCNRecognizer,
    draw_skeleton,
    COCO_SKELETON,
)


def hex_to_bgr(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


TIER_COLORS = {
    "EMERGENCY": (0, 0, 255),
    "PAIN": (0, 165, 255),
    "SYMPTOM": (255, 136, 68),
    "NORMAL": (68, 221, 68),
}


def draw_tier_hud(frame, top5, tier_map, class_to_tier, tier_colors):
    """Draw tier-aware HUD on frame."""
    h, w = frame.shape[:2]
    if not top5:
        return frame

    idx, name, prob = top5[0]
    tier = class_to_tier.get(str(idx), "NORMAL")
    color = tier_colors.get(tier, (200, 200, 200))

    # Banner
    banner_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    cv2.putText(frame, f"[{tier}]", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    action_text = f"{name}  {prob:.0%}"
    ts = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    tx = max(10, (w - ts[0]) // 2)
    cv2.putText(frame, action_text, (tx, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Emergency flash
    if tier == "EMERGENCY":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

    # Top-5 sidebar
    y_start = banner_h + 30
    for i, (ci, cn, cp) in enumerate(top5[:5]):
        y = y_start + i * 28
        t = class_to_tier.get(str(ci), "NORMAL")
        c = tier_colors.get(t, (200, 200, 200))
        bar_w = int(cp * 250)
        cv2.rectangle(frame, (10, y - 14), (10 + bar_w, y + 8), c, -1)
        cv2.putText(frame, f"{cn} {cp:.0%}", (15, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return frame


def main():
    p = argparse.ArgumentParser(description="Headless ONNX inference test")
    p.add_argument("--clip", required=True, help="Video file path")
    p.add_argument("--output", default=None, help="Output video path")
    p.add_argument("--model-dir", default="demo/onnx_models")
    p.add_argument("--det-model", default="yolox_tiny.onnx")
    p.add_argument("--pose-model", default="rtmpose_m.onnx")
    p.add_argument("--recog-model", default="stgcnpp_medical_2d.onnx")
    p.add_argument("--label-map", default="demo1_fed_skeleton/outputs/label_map_medical_15.txt")
    p.add_argument("--tier-map", default="demo1_fed_skeleton/outputs/tier_map.json")
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    p.add_argument("--clip-len", type=int, default=100)
    p.add_argument("--window-seconds", type=float, default=3.0)
    args = p.parse_args()

    # Load tier map
    with open(args.tier_map) as f:
        tier_cfg = json.load(f)
    class_to_tier = tier_cfg["class_to_tier"]
    tier_colors_hex = tier_cfg["tier_colors"]
    tier_colors = {k: hex_to_bgr(v) for k, v in tier_colors_hex.items()}

    # Load models
    model_dir = Path(args.model_dir)
    print("Loading models...")
    detector = YOLOXDetector(str(model_dir / args.det_model), device=args.device)
    pose_est = RTMPoseEstimator(str(model_dir / args.pose_model), device=args.device)
    recognizer = STGCNRecognizer(
        str(model_dir / args.recog_model), args.label_map,
        device="cpu", clip_len=args.clip_len, num_person=2,
    )

    # Open video
    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.clip}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.clip} ({w}x{h} @ {fps:.0f}fps, {n_frames} frames, {n_frames/fps:.1f}s)")

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Read {len(frames)} frames")

    # Process in windows
    window_frames = int(args.window_seconds * fps)
    all_results = []

    for start in range(0, len(frames), window_frames):
        end = min(start + window_frames, len(frames))
        subclip = frames[start:end]
        if len(subclip) < 4:
            break

        t_s, t_e = start / fps, end / fps
        print(f"\n--- Window [{t_s:.1f}s - {t_e:.1f}s] ({len(subclip)} frames) ---")

        t0 = time.perf_counter()
        kpts_buf, scores_buf = [], []

        for f in subclip:
            bboxes = detector(f)
            if len(bboxes) > 0:
                kpts, scores = pose_est(f, bboxes)
            else:
                kpts = np.zeros((0, 17, 2))
                scores = np.zeros((0, 17))
            kpts_buf.append(kpts)
            scores_buf.append(scores)

        results = recognizer(kpts_buf, scores_buf, img_shape=(h, w))
        elapsed = (time.perf_counter() - t0) * 1000

        tier = class_to_tier.get(str(results[0][0]), "NORMAL")
        print(f"  Prediction: [{tier}] {results[0][1]} ({results[0][2]:.1%})")
        print(f"  Top-5:")
        for i, (ci, cn, cp) in enumerate(results[:5]):
            t = class_to_tier.get(str(ci), "NORMAL")
            print(f"    {i+1}. [{t:10s}] {cn:<30s} {cp:.1%}")
        print(f"  Time: {elapsed:.0f}ms ({elapsed/len(subclip):.1f}ms/frame)")

        all_results.append((start, end, results))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for start, end, results in all_results:
        t = class_to_tier.get(str(results[0][0]), "NORMAL")
        print(f"  [{start/fps:.1f}s - {end/fps:.1f}s]  [{t}] {results[0][1]} ({results[0][2]:.1%})")

    # Save annotated video
    if args.output:
        print(f"\nWriting annotated video to {args.output}...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

        for start, end, results in all_results:
            for frame in frames[start:end]:
                vis = frame.copy()
                # Run detection + pose for skeleton overlay
                bboxes = detector(vis)
                if len(bboxes) > 0:
                    kpts, scores = pose_est(vis, bboxes)
                    draw_skeleton(vis, kpts, scores, kpt_thr=0.3)
                    for bb in bboxes:
                        x1, y1, x2, y2 = [int(v) for v in bb]
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
                draw_tier_hud(vis, results, tier_cfg, class_to_tier, tier_colors)
                writer.write(vis)

        writer.release()
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
