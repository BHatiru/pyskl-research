#!/usr/bin/env python
"""Real-time skeleton-based action recognition demo using STGCN++.

Two modes:
  --mode stream   (default) Opens webcam, captures continuously, runs inference
                  in a background thread on a sliding window of frames, and
                  overlays the predicted action on the live feed.
  --mode clip     Records a short clip (press 'r'), runs the full pipeline on
                  it, and shows the result before resuming.

All heavy lifting (Faster-RCNN detection, HRNet pose, STGCN++ recognition)
runs on GPU.  The webcam display stays smooth because inference is off-loaded
to a worker thread.

Example
-------
# Streaming (default) — continuous webcam with live action label:
python demo/demo_realtime.py

# Process a pre-recorded video file instead of webcam:
python demo/demo_realtime.py --video demo/ntu_sample.avi

# Clip mode — press 'r' to record a 3-second clip, then auto-infer:
python demo/demo_realtime.py --mode clip

# Use a different STGCN++ checkpoint (e.g., NTU-60):
python demo/demo_realtime.py \
    --config configs/stgcn++/stgcn++_ntu60_xsub_hrnet/j.py \
    --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_hrnet/j.pth \
    --label-map tools/data/label_map/nturgbd_120.txt
"""

import argparse
import copy
import cv2
import mmcv
import numpy as np
import os
import threading
import time
import torch
import warnings
from collections import deque
from scipy.optimize import linear_sum_assignment

from pyskl.apis import inference_recognizer, init_recognizer

# ---------------------------------------------------------------------------
# mmdet / mmpose imports — graceful fallback with clear error
# ---------------------------------------------------------------------------
try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):

    def inference_detector(*a, **kw):
        pass

    def init_detector(*a, **kw):
        pass

    warnings.warn("mmdet is not installed — detection will not work.")

try:
    from mmpose.apis import (
        inference_top_down_pose_model,
        init_pose_model,
        vis_pose_result,
    )
except (ImportError, ModuleNotFoundError):

    def init_pose_model(*a, **kw):
        pass

    def inference_top_down_pose_model(*a, **kw):
        return ([],)

    def vis_pose_result(*a, **kw):
        pass

    warnings.warn("mmpose is not installed — pose estimation will not work.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE_LARGE = 0.85
FONTSCALE_SMALL = 0.55
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_BG = (40, 40, 40)
THICKNESS = 2

# Detector presets: name -> (config_path, checkpoint_url)
_DEFAULT_DET_CONFIG = "demo/faster_rcnn_r50_fpn_1x_coco-person.py"
_DEFAULT_DET_CKPT = (
    "https://download.openmmlab.com/mmdetection/v2.0/"
    "faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/"
    "faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
)
DETECTOR_PRESETS = {
    "faster-rcnn": (
        _DEFAULT_DET_CONFIG,
        _DEFAULT_DET_CKPT,
    ),
    "yolox-tiny": (
        "demo/yolox_tiny_coco.py",
        "https://download.openmmlab.com/mmdetection/v2.0/yolox/"
        "yolox_tiny_8x8_300e_coco/"
        "yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth",
    ),
}

# Pose-estimator presets: name -> (config_path, checkpoint_url)
POSE_PRESETS = {
    "hrnet-w32": (
        "demo/hrnet_w32_coco_256x192.py",
        "https://download.openmmlab.com/mmpose/top_down/hrnet/"
        "hrnet_w32_coco_256x192-c78dce93_20200708.pth",
    ),
    "vipnas-mbv3": (
        "demo/vipnas_mbv3_coco_256x192.py",
        "https://download.openmmlab.com/mmpose/top_down/vipnas/"
        "vipnas_mbv3_coco_256x192-7018731a_20211122.pth",
    ),
}


# ---------------------------------------------------------------------------
# Helpers (ported from demo_skeleton.py)
# ---------------------------------------------------------------------------
def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    """Naive skeleton tracker copied from demo_skeleton.py."""
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t["data"][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]["data"][-1][1], poses[j])
        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]["data"].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[], track_id=num_tracks)
                    new_track["data"] = [(idx, poses[j])]
                    tracks.append(new_track)
    if num_joints is None:
        return None, None
    tracks.sort(key=lambda x: -len(x["data"]))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track["data"]:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time STGCN++ skeleton-based action recognition"
    )
    p.add_argument(
        "--mode",
        choices=["stream", "clip"],
        default="stream",
        help="stream = continuous sliding-window inference; "
        'clip = press "r" to record a clip then infer',
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file. If omitted, webcam is used.",
    )
    p.add_argument(
        "--camera", type=int, default=0, help="Camera device index (default: 0)"
    )
    # --- Action recognition model ---
    p.add_argument(
        "--config",
        default="configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py",
        help="STGCN++ config file",
    )
    p.add_argument(
        "--checkpoint",
        default="http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/"
        "stgcnpp_ntu120_xsub_hrnet/j.pth",
        help="STGCN++ checkpoint (URL or local path)",
    )
    # --- Detection model ---
    p.add_argument(
        "--detector",
        choices=["faster-rcnn", "yolox-tiny", "none"],
        default="faster-rcnn",
        help="Person detector: 'faster-rcnn' (default, accurate), "
        "'yolox-tiny' (~5× faster, single-stage), "
        "'none' (full-frame bbox, 0 ms — single-person only)",
    )
    p.add_argument(
        "--det-config",
        default=None,
        help="Override person detection config (mmdet). "
        "Leave unset to use the --detector preset.",
    )
    p.add_argument(
        "--det-checkpoint",
        default=None,
        help="Override person detection checkpoint. "
        "Leave unset to use the --detector preset.",
    )
    # --- Pose model ---
    p.add_argument(
        "--pose-model",
        choices=["hrnet-w32", "vipnas-mbv3"],
        default="hrnet-w32",
        help="Pose estimator: 'hrnet-w32' (default, 0.746 AP, ~28M params), "
        "'vipnas-mbv3' (0.700 AP, ~3M params, ~3-5\u00d7 faster)",
    )
    p.add_argument(
        "--pose-config",
        default=None,
        help="Override pose estimation config (mmpose). "
        "Leave unset to use the --pose-model preset.",
    )
    p.add_argument(
        "--pose-checkpoint",
        default=None,
        help="Override pose estimation checkpoint. "
        "Leave unset to use the --pose-model preset.",
    )
    # --- Thresholds & knobs ---
    p.add_argument(
        "--det-score-thr",
        type=float,
        default=None,
        help="Detection confidence threshold (default: 0.9 for "
        "faster-rcnn, 0.5 for yolox-tiny)",
    )
    p.add_argument(
        "--label-map",
        default="tools/data/label_map/nturgbd_120.txt",
        help="Label map text file (one label per line)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (cuda:0 or cpu)",
    )
    p.add_argument(
        "--clip-len", type=int, default=30, help="Number of frames per sliding window"
    )
    p.add_argument(
        "--clip-stride",
        type=int,
        default=15,
        help="How many new frames to wait before re-running "
        "inference (stream mode only)",
    )
    p.add_argument(
        "--record-seconds",
        type=float,
        default=3.0,
        help="Seconds to record in clip mode",
    )
    p.add_argument(
        "--short-side",
        type=int,
        default=320,
        help="Resize frames so short side = this (reduces det cost)",
    )
    p.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Process every N-th frame in the buffer to speed up "
        "detection/pose (1 = all frames)",
    )
    p.add_argument(
        "--capture-fps",
        type=int,
        default=15,
        help="Subsample webcam capture to this FPS during recording "
        "(e.g. 15 = keep every 2nd frame from a 30fps cam)",
    )
    args = p.parse_args()

    # Auto-set detection threshold per detector if user didn't specify
    if args.det_score_thr is None:
        args.det_score_thr = {"faster-rcnn": 0.9, "yolox-tiny": 0.5, "none": 0.5}[
            args.detector
        ]

    return args


# ===================================================================
# Core engine
# ===================================================================
class ActionRecognitionEngine:
    """Loads det + pose + recognition models and exposes `infer(frames)`."""

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.no_det = args.detector == "none"

        if self.no_det:
            print("[1/3] Person detector: DISABLED (full-frame bbox)")
            self.det_model = None
        else:
            # Resolve config/checkpoint from preset, allow user override
            preset_cfg, preset_ckpt = DETECTOR_PRESETS[args.detector]
            det_cfg = args.det_config if args.det_config else preset_cfg
            det_ckpt = args.det_checkpoint if args.det_checkpoint else preset_ckpt
            det_name = args.detector.replace("-", " ").upper()
            print(f"[1/3] Loading person detector ({det_name}) ...")
            self.det_model = init_detector(det_cfg, det_ckpt, self.device)

        # --- Pose estimator ---
        pose_preset_cfg, pose_preset_ckpt = POSE_PRESETS[args.pose_model]
        pose_cfg = args.pose_config if args.pose_config else pose_preset_cfg
        pose_ckpt = args.pose_checkpoint if args.pose_checkpoint else pose_preset_ckpt
        pose_name = args.pose_model.replace("-", " ").upper()
        print(f"[2/3] Loading pose estimator ({pose_name}) ...")
        self.pose_model = init_pose_model(pose_cfg, pose_ckpt, self.device)

        # --- Action recogniser config ---
        cfg = mmcv.Config.fromfile(args.config)
        cfg.data.test.pipeline = [
            x for x in cfg.data.test.pipeline if x["type"] != "DecompressPose"
        ]
        self.GCN_flag = "GCN" in cfg.model.type
        self.GCN_nperson = 2
        if self.GCN_flag:
            fmt = [o for o in cfg.data.test.pipeline if o["type"] == "FormatGCNInput"]
            if fmt:
                self.GCN_nperson = fmt[0].get("num_person", 2)

        print("[3/3] Loading action recogniser (STGCN++) ...")
        self.recognizer = init_recognizer(cfg, args.checkpoint, self.device)

        self.label_map = [l.strip() for l in open(args.label_map).readlines()]
        print(f"  → {len(self.label_map)} action classes loaded.")
        print("All models loaded ✓\n")

    # ------------------------------------------------------------------
    def infer(self, frames):
        """Run the full pipeline on a list of BGR numpy frames.

        Returns
        -------
        label : str          top-1 action name
        top5  : list[tuple]  [(name, score), ...] top-5
        dt    : float         wall-clock seconds for this inference
        vis_frames : list[np.ndarray]  frames with skeleton overlay drawn
        """
        t0 = time.time()
        if len(frames) == 0:
            return "No frames", [], 0.0, []

        h, w = frames[0].shape[:2]
        num_frame = len(frames)
        print(f"  ┌ Inference on {num_frame} frames ({w}×{h}) ...")

        # ----- Detection -----
        t_det = time.time()
        det_results = []
        if self.no_det:
            # Single full-frame bounding box per frame (skip detector)
            full_box = np.array([[0, 0, w - 1, h - 1, 1.0]], dtype=np.float32)
            det_results = [full_box] * num_frame
        else:
            for frame in frames:
                res = inference_detector(self.det_model, frame)
                res = res[0][res[0][:, 4] >= self.args.det_score_thr]
                det_results.append(res)
        dt_det = time.time() - t_det
        persons = sum(len(d) for d in det_results)
        det_label = (
            "(skipped)"
            if self.no_det
            else (f"({dt_det/num_frame*1000:.0f} ms/frame, {persons} boxes total)")
        )
        print(f"  │  Detection:   {dt_det:6.2f}s  {det_label}")

        # ----- Pose estimation -----
        t_pose = time.time()
        pose_results = []
        for f, d in zip(frames, det_results):
            d_list = [dict(bbox=x) for x in list(d)]
            pose = inference_top_down_pose_model(
                self.pose_model, f, d_list, format="xyxy"
            )[0]
            pose_results.append(pose)
        dt_pose = time.time() - t_pose
        print(
            f"  │  Pose est.:   {dt_pose:6.2f}s  "
            f"({dt_pose/num_frame*1000:.0f} ms/frame)"
        )

        # ----- Render skeleton overlays -----
        t_vis = time.time()
        vis_frames = []
        for f, poses in zip(frames, pose_results):
            vis = vis_pose_result(
                self.pose_model, f, poses, kpt_score_thr=0.3, radius=4, thickness=2
            )
            vis_frames.append(vis)
        dt_vis = time.time() - t_vis
        print(f"  │  Skeleton viz:{dt_vis:6.2f}s")

        # ----- Build fake annotation -----
        fake_anno = dict(
            frame_dir="",
            label=-1,
            img_shape=(h, w),
            original_shape=(h, w),
            start_index=0,
            modality="Pose",
            total_frames=num_frame,
        )

        if self.GCN_flag:
            tracking_inputs = [
                [p["keypoints"] for p in poses] for poses in pose_results
            ]
            kp, kp_score = pose_tracking(tracking_inputs, max_tracks=self.GCN_nperson)
            fake_anno["keypoint"] = kp
            fake_anno["keypoint_score"] = kp_score
        else:
            num_person = max([len(x) for x in pose_results]) if pose_results else 0
            num_kp = 17
            kp = np.zeros((num_person, num_frame, num_kp, 2), dtype=np.float16)
            kp_score = np.zeros((num_person, num_frame, num_kp), dtype=np.float16)
            for i, poses in enumerate(pose_results):
                for j, pose in enumerate(poses):
                    k = pose["keypoints"]
                    kp[j, i] = k[:, :2]
                    kp_score[j, i] = k[:, 2]
            fake_anno["keypoint"] = kp
            fake_anno["keypoint_score"] = kp_score

        if fake_anno["keypoint"] is None:
            print(f"  └ No person detected")
            return "No person detected", [], time.time() - t0, vis_frames

        # ----- Recognition -----
        t_rec = time.time()
        results = inference_recognizer(self.recognizer, fake_anno)
        dt_rec = time.time() - t_rec
        dt = time.time() - t0

        top1_label = self.label_map[results[0][0]]
        top5 = [(self.label_map[idx], float(score)) for idx, score in results[:5]]
        print(f"  │  Recognition: {dt_rec:6.3f}s")
        print(f"  │  TOTAL:       {dt:6.2f}s  " f"({num_frame/dt:.1f} fps effective)")
        print(f"  └ Result: {top1_label} ({top5[0][1]:.3f})")
        return top1_label, top5, dt, vis_frames


# ===================================================================
# Drawing helpers
# ===================================================================
def draw_overlay(frame, label, top5, inf_hz, mode_tag="stream"):
    """Draw a semi-transparent HUD with action info."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Background bar at top
    bar_h = 40 + 22 * min(len(top5), 5)
    cv2.rectangle(overlay, (0, 0), (w, bar_h), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Top-1 action
    cv2.putText(
        frame,
        f"Action: {label}",
        (12, 30),
        FONTFACE,
        FONTSCALE_LARGE,
        COLOR_GREEN,
        THICKNESS,
    )

    # Top-5 scores
    for i, (name, score) in enumerate(top5[:5]):
        txt = f"{i+1}. {name}: {score:.3f}"
        cv2.putText(
            frame,
            txt,
            (16, 55 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONTSCALE_SMALL,
            COLOR_WHITE,
            1,
        )

    # Bottom status bar
    status = f"[{mode_tag.upper()}] Inference: {inf_hz:.2f} Hz  |  ESC=quit"
    cv2.putText(
        frame, status, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1
    )
    return frame


# ===================================================================
# Stream mode — continuous sliding-window inference
# ===================================================================
def run_stream(engine, args):
    src = args.video if args.video else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: cannot open video source: {src}")
        return

    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video source opened  (fps={fps_cam:.0f})")
    print(
        f"Sliding window: {args.clip_len} frames, "
        f"stride {args.clip_stride}, skip {args.skip_frames}"
    )
    print("Press ESC to quit.\n")

    # Shared state (protected by lock)
    lock = threading.Lock()
    frame_buffer = deque(maxlen=args.clip_len * 3)
    state = dict(
        label="Warming up …",
        top5=[],
        inf_hz=0.0,
        frames_since_last=0,
        running=True,
        skeleton_frame=None,
    )

    def worker():
        """Background inference loop."""
        while state["running"]:
            # Wait until enough new frames have been collected
            with lock:
                n_buf = len(frame_buffer)
                n_since = state["frames_since_last"]
            if n_buf < args.clip_len or n_since < args.clip_stride:
                time.sleep(0.02)
                continue

            # Grab the last clip_len frames
            with lock:
                raw_frames = list(frame_buffer)[-args.clip_len :]
                state["frames_since_last"] = 0

            # Optionally skip frames to reduce workload
            if args.skip_frames > 1:
                raw_frames = raw_frames[:: args.skip_frames]

            label, top5, dt, vis_frames = engine.infer(raw_frames)
            hz = 1.0 / dt if dt > 0 else 0

            # Keep the last skeleton-rendered frame for display
            last_skel = vis_frames[-1] if vis_frames else None

            with lock:
                state["label"] = label
                state["top5"] = top5
                state["inf_hz"] = hz
                state["skeleton_frame"] = last_skel

            torch.cuda.empty_cache()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while cap.isOpened() and state["running"]:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                # Loop video file for demo purposes
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # Optionally resize for faster detection later
        fh, fw = frame.shape[:2]
        if min(fh, fw) > args.short_side:
            scale = args.short_side / min(fh, fw)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        with lock:
            frame_buffer.append(frame.copy())
            state["frames_since_last"] += 1
            label = state["label"]
            top5 = list(state["top5"])
            hz = state["inf_hz"]
            skel_frame = state["skeleton_frame"]

        # If we have a skeleton-rendered frame, blend it onto the live feed
        if skel_frame is not None:
            sh, sw = skel_frame.shape[:2]
            fh2, fw2 = frame.shape[:2]
            if (sh, sw) == (fh2, fw2):
                # Alpha-blend: keep the live frame but overlay skeleton lines
                frame = cv2.addWeighted(frame, 0.3, skel_frame, 0.7, 0)
            else:
                # Resize skeleton frame to match if dims differ
                skel_resized = cv2.resize(skel_frame, (fw2, fh2))
                frame = cv2.addWeighted(frame, 0.3, skel_resized, 0.7, 0)

        display = draw_overlay(frame, label, top5, hz, "stream")
        cv2.imshow("PYSKL Real-time Action Recognition  [ESC = quit]", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    state["running"] = False
    cap.release()
    cv2.destroyAllWindows()
    print("\nStream ended.")


# ===================================================================
# Clip mode — record-then-infer
# ===================================================================
def run_clip(engine, args):
    src = args.video if args.video else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: cannot open video source: {src}")
        return

    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Subsample: only keep every Nth frame to match --capture-fps
    keep_every = max(1, round(fps_cam / args.capture_fps))
    effective_fps = fps_cam / keep_every
    frames_to_record = int(args.record_seconds * effective_fps)

    print(
        f'CLIP MODE — press "r" to record {args.record_seconds}s '
        f"({frames_to_record} frames @ {effective_fps:.0f}fps, "
        f"cam={fps_cam:.0f}fps keep_every={keep_every}), ESC to quit.\n"
    )

    recording = False
    clip_frames = []
    frame_counter = 0
    last_label = 'Press "r" to record a clip'
    last_top5 = []
    last_hz = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        if min(fh, fw) > args.short_side:
            scale = args.short_side / min(fh, fw)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        if recording:
            frame_counter += 1
            # Only keep every Nth frame to achieve target capture fps
            if frame_counter % keep_every == 0:
                clip_frames.append(frame.copy())
            # Draw recording indicator (on every displayed frame)
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"REC {len(clip_frames)}/{frames_to_record}",
                (50, 38),
                FONTFACE,
                0.7,
                (0, 0, 255),
                2,
            )

            if len(clip_frames) >= frames_to_record:
                recording = False
                print(f"  Recorded {len(clip_frames)} frames — running inference …")
                label, top5, dt, vis_frames = engine.infer(clip_frames)
                hz = 1.0 / dt if dt > 0 else 0
                last_label = label
                last_top5 = top5
                last_hz = hz
                print(f"  → {label}  ({dt:.2f}s)\n")

                # Show skeleton playback of the recorded clip
                if vis_frames:
                    for vf in vis_frames:
                        disp = draw_overlay(vf, label, top5, hz, "clip-replay")
                        cv2.imshow(
                            "PYSKL Clip Action Recognition  [r=record, ESC=quit]", disp
                        )
                        if cv2.waitKey(50) & 0xFF == 27:
                            break

                clip_frames = []
                torch.cuda.empty_cache()
        else:
            display = draw_overlay(frame, last_label, last_top5, last_hz, "clip")
            frame = display

        cv2.imshow("PYSKL Clip Action Recognition  [r=record, ESC=quit]", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("r") and not recording:
            recording = True
            clip_frames = []
            frame_counter = 0
            print("Recording …")

    cap.release()
    cv2.destroyAllWindows()
    print("\nClip mode ended.")


# ===================================================================
# Main
# ===================================================================
def main():
    args = parse_args()

    # Validate device
    if "cuda" in args.device and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        args.device = "cpu"

    engine = ActionRecognitionEngine(args)

    if args.mode == "stream":
        run_stream(engine, args)
    else:
        run_clip(engine, args)


if __name__ == "__main__":
    main()
