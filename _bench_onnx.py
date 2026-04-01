"""Headless benchmark of demo_onnx.py pipeline on ntu_sample.avi.

Extracts frames, runs det→pose→recognition, prints timing breakdown.
No GUI required.
"""
import time
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo.demo_onnx import YOLOXDetector, RTMPoseEstimator, STGCNRecognizer

VIDEO = "demo/ntu_sample.avi"
MODEL_DIR = "demo/onnx_models"
LABEL_MAP = "tools/data/label_map/nturgbd_120.txt"
SHORT_SIDE = 320
MAX_FRAMES = 30

def main():
    # Extract frames
    cap = cv2.VideoCapture(VIDEO)
    frames = []
    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if min(h, w) > SHORT_SIDE:
            scale = SHORT_SIDE / min(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        frames.append(frame)
    cap.release()
    proc_h, proc_w = frames[0].shape[:2]
    print(f"Extracted {len(frames)} frames ({proc_w}x{proc_h})")

    # Load models
    print("\nLoading ONNX models...")
    t0 = time.perf_counter()
    detector = YOLOXDetector(f"{MODEL_DIR}/yolox_tiny.onnx", device="cpu", score_thr=0.5)
    pose = RTMPoseEstimator(f"{MODEL_DIR}/rtmpose_m.onnx", device="cpu")
    recog = STGCNRecognizer(
        f"{MODEL_DIR}/stgcnpp_ntu120_xsub_hrnet_j.onnx",
        LABEL_MAP, device="cpu", clip_len=100, num_person=2
    )
    load_time = time.perf_counter() - t0
    print(f"Models loaded in {load_time:.1f}s\n")

    # Warmup
    dummy = np.random.randint(0, 255, (proc_h, proc_w, 3), dtype=np.uint8)
    for _ in range(3):
        detector(dummy)
        pose(dummy, [[0, 0, proc_w, proc_h]])

    # Run pipeline
    n = len(frames)
    kpts_buf, scores_buf = [], []

    print(f"Running full pipeline on {n} frames...")
    t_det_total = 0
    t_pose_total = 0

    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        bboxes = detector(f)
        t_det_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        if len(bboxes) > 0:
            kpts, scores = pose(f, bboxes)
        else:
            kpts = np.zeros((0, 17, 2))
            scores = np.zeros((0, 17))
        t_pose_total += time.perf_counter() - t0

        kpts_buf.append(kpts)
        scores_buf.append(scores)

    t0 = time.perf_counter()
    results = recog(kpts_buf, scores_buf, img_shape=(proc_h, proc_w))
    t_recog = time.perf_counter() - t0
    t_total = t_det_total + t_pose_total + t_recog

    # Results
    print(f"\n{'='*55}")
    print(f"  ONNX Pipeline — {n} frames @ {proc_w}x{proc_h} (CPU)")
    print(f"{'='*55}")
    print(f"  {'Stage':<20} {'Total (ms)':<14} {'Per-frame (ms)'}")
    print(f"  {'-'*50}")
    print(f"  {'Detection':<20} {t_det_total*1000:<14.1f} {t_det_total*1000/n:<.1f}")
    print(f"  {'Pose estimation':<20} {t_pose_total*1000:<14.1f} {t_pose_total*1000/n:<.1f}")
    print(f"  {'Recognition':<20} {t_recog*1000:<14.1f} {'—'}")
    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<20} {t_total*1000:<14.1f} {t_total*1000/n:<.1f}")
    print(f"  Effective FPS:     {n/t_total:.1f}")
    print(f"\n  Top-5 predictions:")
    for i, (idx, lbl, prob) in enumerate(results[:5]):
        print(f"    {i+1}. {lbl:<30} {prob:.1%}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
