#!/usr/bin/env python
"""Pure-ONNX real-time skeleton-based action recognition demo.

This script requires ONLY: numpy, opencv-python, onnxruntime(-gpu).
No mmcv, mmpose, mmdet, or pyskl imports at runtime.

Pipeline:
    webcam frame  →  YOLOX-Tiny (person detection)
                  →  RTMPose-m  (17-kp COCO pose estimation)
                  →  STGCN++    (skeleton action recognition, 120 NTU classes)

Models (all in demo/onnx_models/):
    yolox_tiny.onnx                   – 19 MB, 416×416, built-in NMS
    rtmpose_m.onnx                    – 52 MB, 192×256, SimCC output
    stgcnpp_ntu120_xsub_hrnet_j.onnx –  5 MB, (N,2,100,17,3)

Usage:
    python demo/demo_onnx.py --device cuda
    python demo/demo_onnx.py --device cpu --camera 0 --window-frames 30
    python demo/demo_onnx.py --clip path/to/video.mp4

Author: Research demo for advisor meeting
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path


# Auto-add nvidia cuDNN/cublas DLLs to PATH (for onnxruntime CUDA EP)
def _setup_nvidia_path():
    try:
        import nvidia.cudnn

        cudnn_bin = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "bin")
        if os.path.isdir(cudnn_bin) and cudnn_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cudnn_bin + os.pathsep + os.environ.get("PATH", "")
    except (ImportError, Exception):
        pass
    try:
        import importlib

        spec = importlib.util.find_spec("nvidia.cublas")
        if spec and spec.submodule_search_locations:
            cublas_bin = os.path.join(spec.submodule_search_locations[0], "bin")
            if os.path.isdir(cublas_bin) and cublas_bin not in os.environ.get(
                "PATH", ""
            ):
                os.environ["PATH"] = (
                    cublas_bin + os.pathsep + os.environ.get("PATH", "")
                )
    except (ImportError, Exception):
        pass


_setup_nvidia_path()

import cv2
import numpy as np

# ---------------------------------------------------------------------------
#  ONNX Runtime session helper
# ---------------------------------------------------------------------------


def create_session(onnx_path, device="cuda"):
    """Create ONNX Runtime InferenceSession with preferred provider."""
    import onnxruntime as ort

    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, providers=providers)
    actual = sess.get_providers()
    print(f"  {Path(onnx_path).name}: providers={actual}")
    return sess


# ---------------------------------------------------------------------------
#  YOLOX-Tiny  (person detection with built-in NMS)
# ---------------------------------------------------------------------------


class YOLOXDetector:
    """YOLOX-Tiny ONNX detector.

    The HumanArt version includes built-in NMS, so output is:
        dets:   (1, N, 5)  = [x1, y1, x2, y2, score]
        labels: (1, N)     = class IDs (0 = person)
    """

    def __init__(self, onnx_path, device="cuda", input_size=(416, 416), score_thr=0.5):
        self.session = create_session(onnx_path, device)
        self.input_size = input_size  # (H, W)
        self.score_thr = score_thr
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        """Letterbox pad + float32, NO mean/std normalization."""
        h, w = img.shape[:2]
        target_h, target_w = self.input_size
        ratio = min(target_h / h, target_w / w)

        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # HWC → CHW, float32
        blob = padded.transpose(2, 0, 1).astype(np.float32)[None]
        return blob, ratio

    def __call__(self, img):
        """Return list of [x1,y1,x2,y2] in original image coords."""
        blob, ratio = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})
        dets = outputs[0][0]  # (N, 5)
        labels = outputs[1][0]  # (N,)

        # Filter: person class (0) and score
        mask = (labels == 0) & (dets[:, 4] > self.score_thr)
        boxes = dets[mask, :4] / ratio  # rescale to original image
        return boxes.tolist()


# ---------------------------------------------------------------------------
#  RTMPose-m  (top-down 17-keypoint pose estimator, SimCC)
# ---------------------------------------------------------------------------


class RTMPoseEstimator:
    """RTMPose ONNX top-down pose estimator with SimCC decoding.

    Outputs per person: keypoints (17, 2) in image coords, scores (17,).
    """

    MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __init__(self, onnx_path, device="cuda", input_size=(192, 256)):
        self.session = create_session(onnx_path, device)
        self.input_size = input_size  # (W, H)
        self.input_name = self.session.get_inputs()[0].name
        self.simcc_split_ratio = 2.0

    # ---- Preprocessing helpers (ported from rtmlib) ----

    @staticmethod
    def _bbox_xyxy2cs(bbox, padding=1.25):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        w *= padding
        h *= padding
        center = np.array([cx, cy], dtype=np.float32)
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    @staticmethod
    def _get_warp_matrix(center, scale, rot, output_size):
        """Get affine transform matrix (simplified, rot=0)."""
        src_w, src_h = scale
        dst_w, dst_h = output_size

        src_dir = np.array([0, src_w * -0.5], dtype=np.float32)
        dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

        src_p1 = center
        src_p2 = center + src_dir
        src_p3 = np.array(
            [src_p1[0] - src_dir[1], src_p1[1] + src_dir[0]], dtype=np.float32
        )

        dst_p1 = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
        dst_p2 = dst_p1 + dst_dir
        dst_p3 = np.array(
            [dst_p1[0] - dst_dir[1], dst_p1[1] + dst_dir[0]], dtype=np.float32
        )

        src = np.stack([src_p1, src_p2, src_p3]).astype(np.float32)
        dst = np.stack([dst_p1, dst_p2, dst_p3]).astype(np.float32)
        return cv2.getAffineTransform(src, dst)

    def preprocess(self, img, bbox):
        """Affine-crop person ROI, normalize, return blob + metadata."""
        center, scale = self._bbox_xyxy2cs(bbox, padding=1.25)

        w, h = self.input_size
        # Enforce aspect ratio
        aspect_ratio = w / h
        if scale[0] / scale[1] > aspect_ratio:
            scale[1] = scale[0] / aspect_ratio
        else:
            scale[0] = scale[1] * aspect_ratio

        warp_mat = self._get_warp_matrix(center, scale, 0, (w, h))
        warped = cv2.warpAffine(img, warp_mat, (w, h), flags=cv2.INTER_LINEAR)

        # Normalize
        warped = warped.astype(np.float32)
        warped = (warped - self.MEAN) / self.STD
        blob = warped.transpose(2, 0, 1)[None]  # (1,3,H,W)
        return blob, center, scale

    @staticmethod
    def _get_simcc_maximum(simcc_x, simcc_y):
        """Decode SimCC outputs to keypoint locs and scores."""
        N, K, _ = simcc_x.shape
        simcc_x_flat = simcc_x.reshape(N * K, -1)
        simcc_y_flat = simcc_y.reshape(N * K, -1)

        x_locs = np.argmax(simcc_x_flat, axis=1)
        y_locs = np.argmax(simcc_y_flat, axis=1)
        max_val_x = np.max(simcc_x_flat, axis=1)
        max_val_y = np.max(simcc_y_flat, axis=1)

        locs = np.stack([x_locs, y_locs], axis=-1).astype(np.float32)
        vals = 0.5 * (max_val_x + max_val_y)
        locs[vals <= 0.0] = -1

        return locs.reshape(N, K, 2), vals.reshape(N, K)

    def postprocess(self, outputs, center, scale):
        """Decode SimCC → image-space keypoints."""
        simcc_x, simcc_y = outputs
        locs, scores = self._get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / self.simcc_split_ratio

        # Rescale to image coordinates
        input_size = np.array(self.input_size, dtype=np.float32)  # (W, H)
        keypoints = keypoints / input_size * scale
        keypoints = keypoints + center - scale / 2.0

        return keypoints[0], scores[0]  # (17,2), (17,)

    def __call__(self, img, bboxes):
        """Run pose on all person bboxes. Return (N,17,2), (N,17)."""
        if len(bboxes) == 0:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))

        all_kpts, all_scores = [], []
        for bbox in bboxes:
            blob, center, scale = self.preprocess(img, bbox)
            outputs = self.session.run(None, {self.input_name: blob})
            kpts, scores = self.postprocess(outputs, center, scale)
            all_kpts.append(kpts)
            all_scores.append(scores)

        return np.stack(all_kpts), np.stack(all_scores)


# ---------------------------------------------------------------------------
#  STGCN++ action recognizer  (skeleton ONNX)
# ---------------------------------------------------------------------------


class STGCNRecognizer:
    """STGCN++ ONNX recognizer.

    Input:  (1, 2, T, 17, 3) float32   T=100, C=(x_norm, y_norm, score)
    Output: (1, 120) logits
    """

    def __init__(
        self,
        onnx_path,
        label_map_path,
        device="cuda",
        clip_len=100,
        num_person=2,
        img_shape=(1080, 1920),
    ):
        self.session = create_session(onnx_path, device)
        self.input_name = self.session.get_inputs()[0].name
        self.clip_len = clip_len
        self.num_person = num_person
        self.img_h, self.img_w = img_shape

        # Load label map
        with open(label_map_path) as f:
            self.labels = [l.strip() for l in f.readlines()]
        print(f"  {len(self.labels)} action classes loaded.")

    def build_input(self, keypoints_buffer, scores_buffer, img_shape=None):
        """Build STGCN++ input tensor from keypoint buffers.

        Args:
            keypoints_buffer: list of (N_persons, 17, 2) arrays  (pixel coords)
            scores_buffer:    list of (N_persons, 17) arrays
            img_shape: (h, w) of the source frames

        Returns:
            np.ndarray of shape (1, 2, T, 17, 3) ready for inference
        """
        if img_shape is not None:
            h, w = img_shape
        else:
            h, w = self.img_h, self.img_w

        T = self.clip_len
        M = self.num_person
        V = 17
        C = 3

        # Uniform sample to T frames
        n_frames = len(keypoints_buffer)
        if n_frames == 0:
            return np.zeros((1, M, T, V, C), dtype=np.float32)

        indices = np.linspace(0, n_frames - 1, T).astype(int)

        skeleton = np.zeros((M, T, V, C), dtype=np.float32)
        for t_out, t_in in enumerate(indices):
            kpts = keypoints_buffer[t_in]  # (N, 17, 2) pixel coords
            scores = scores_buffer[t_in]  # (N, 17)
            n_persons = min(kpts.shape[0], M)

            for p in range(n_persons):
                # PreNormalize2D: map pixel coords to [-1, 1]
                x_norm = (kpts[p, :, 0] - w / 2.0) / (w / 2.0)
                y_norm = (kpts[p, :, 1] - h / 2.0) / (h / 2.0)
                score = scores[p]

                # Zero out low-confidence keypoints
                low_conf = score < 0.01
                x_norm[low_conf] = 0.0
                y_norm[low_conf] = 0.0
                score[low_conf] = 0.0

                skeleton[p, t_out, :, 0] = x_norm
                skeleton[p, t_out, :, 1] = y_norm
                skeleton[p, t_out, :, 2] = score

        return skeleton[None]  # (1, M, T, V, C)

    def __call__(self, keypoints_buffer, scores_buffer, img_shape=None):
        """Recognize action from skeleton buffer.

        Returns: list of (class_idx, label, probability) sorted by prob.
        """
        inp = self.build_input(keypoints_buffer, scores_buffer, img_shape)
        logits = self.session.run(None, {self.input_name: inp})[0]  # (1, 120)

        # Softmax
        logits = logits[0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        # Top-5
        top_idx = probs.argsort()[::-1][:5]
        results = [(int(i), self.labels[i], float(probs[i])) for i in top_idx]
        return results


# ---------------------------------------------------------------------------
#  Skeleton drawing utilities
# ---------------------------------------------------------------------------

COCO_SKELETON = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),  # legs
    (11, 12),  # hip
    (5, 11),
    (6, 12),  # torso
    (5, 6),  # shoulders
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),  # arms
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # face
]

PALETTE = [
    (255, 128, 0),
    (255, 153, 51),
    (255, 178, 102),
    (230, 230, 0),
    (255, 153, 255),
    (153, 204, 255),
    (255, 102, 255),
    (255, 51, 255),
    (102, 178, 255),
    (51, 153, 255),
    (255, 153, 153),
    (255, 102, 102),
    (255, 51, 51),
    (153, 255, 153),
    (102, 255, 102),
    (51, 255, 51),
    (0, 255, 0),
]


def draw_skeleton(frame, keypoints, scores, kpt_thr=0.3):
    """Draw skeleton overlay on frame.

    Args:
        keypoints: (N, 17, 2) or (17, 2) in pixel coordinates.
        scores: (N, 17) or (17,)
    """
    if keypoints.ndim == 2:
        keypoints = keypoints[None]
        scores = scores[None]

    for person_kpts, person_scores in zip(keypoints, scores):
        # Draw limbs
        for i, j in COCO_SKELETON:
            if person_scores[i] > kpt_thr and person_scores[j] > kpt_thr:
                pt1 = tuple(person_kpts[i].astype(int))
                pt2 = tuple(person_kpts[j].astype(int))
                color = PALETTE[i % len(PALETTE)]
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw keypoints
        for k, (pt, s) in enumerate(zip(person_kpts, person_scores)):
            if s > kpt_thr:
                x, y = int(pt[0]), int(pt[1])
                color = PALETTE[k % len(PALETTE)]
                cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_hud(frame, results, fps, timings, frame_count):
    """Draw heads-up display with action label, FPS, and timing info."""
    h, w = frame.shape[:2]

    # Semi-transparent bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Pipeline timings
    timing_str = " | ".join(f"{k}: {v:.1f}ms" for k, v in timings.items())
    cv2.putText(
        frame, timing_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )

    # Frame count
    cv2.putText(
        frame,
        f"Frames: {frame_count}",
        (w - 160, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    # Top-1 action
    if results:
        idx, label, prob = results[0]
        text = f"{label} ({prob:.1%})"
        cv2.putText(
            frame, text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        # Top-5 sidebar
        for i, (idx, label, prob) in enumerate(results[:5]):
            y = 140 + i * 25
            bar_w = int(prob * 200)
            cv2.rectangle(frame, (10, y - 12), (10 + bar_w, y + 5), (0, 200, 255), -1)
            cv2.putText(
                frame,
                f"{label[:25]} {prob:.1%}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    return frame


# ---------------------------------------------------------------------------
#  Main streaming loop
# ---------------------------------------------------------------------------


def run_stream(args):
    """Real-time webcam streaming with ONNX pipeline."""
    print("\n=== ONNX Real-Time Action Recognition ===")
    print(f"Device: {args.device}")

    # Load models
    model_dir = Path(args.model_dir)
    det_path = str(model_dir / args.det_model)
    pose_path = str(model_dir / args.pose_model)
    recog_path = str(model_dir / args.recog_model)

    recog_device = args.recog_device or args.device
    print("\nLoading models...")
    detector = YOLOXDetector(det_path, device=args.device, score_thr=args.det_score_thr)
    pose_estimator = RTMPoseEstimator(pose_path, device=args.device)
    recognizer = STGCNRecognizer(
        recog_path,
        args.label_map,
        device=recog_device,
        clip_len=args.clip_len,
        num_person=args.num_person,
    )

    # Open camera/video
    source = args.clip if args.clip else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {source}")
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"\nSource: {source} ({src_w}×{src_h} @ {src_fps:.0f}fps)")

    # Processing resolution
    if args.short_side > 0:
        scale = args.short_side / min(src_h, src_w)
        proc_w, proc_h = int(src_w * scale), int(src_h * scale)
    else:
        proc_w, proc_h = src_w, src_h
    print(f"Processing at: {proc_w}×{proc_h}")

    # Buffers
    window = args.window_frames
    kpts_buffer = deque(maxlen=window)
    scores_buffer = deque(maxlen=window)

    # FPS tracking
    fps_window = deque(maxlen=30)
    frame_count = 0
    results = []

    # Detection frequency (skip frames for speed)
    det_every = args.det_every

    cached_bboxes = []

    print(f"\nWindow: {window} frames | Clip len: {args.clip_len}")
    print(f"Det every: {det_every} frames | Det score: {args.det_score_thr}")
    print("Press Q to quit.\n")

    # Warmup
    print("Warming up GPU...")
    dummy_img = np.random.randint(0, 255, (proc_h, proc_w, 3), dtype=np.uint8)
    for _ in range(3):
        detector(dummy_img)
        pose_estimator(dummy_img, [[0, 0, proc_w, proc_h]])
    print("Warmup done.\n")

    while True:
        t_total_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            if args.clip:
                print("Video ended.")
            break

        frame_count += 1

        # Resize for processing
        if proc_w != src_w or proc_h != src_h:
            proc_frame = cv2.resize(frame, (proc_w, proc_h))
        else:
            proc_frame = frame.copy()

        timings = {}

        # --- Detection ---
        t0 = time.perf_counter()
        if frame_count % det_every == 1 or det_every == 1:
            bboxes = detector(proc_frame)
            cached_bboxes = bboxes
        else:
            bboxes = cached_bboxes
        timings["det"] = (time.perf_counter() - t0) * 1000

        # --- Pose estimation ---
        t0 = time.perf_counter()
        if len(bboxes) > 0:
            keypoints, scores = pose_estimator(proc_frame, bboxes)
        else:
            keypoints = np.zeros((0, 17, 2))
            scores = np.zeros((0, 17))
        timings["pose"] = (time.perf_counter() - t0) * 1000

        # --- Buffer management ---
        kpts_buffer.append(keypoints)
        scores_buffer.append(scores)

        # --- Action recognition ---
        t0 = time.perf_counter()
        if len(kpts_buffer) >= args.min_frames:
            results = recognizer(
                list(kpts_buffer), list(scores_buffer), img_shape=(proc_h, proc_w)
            )
        timings["recog"] = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_total_start) * 1000
        timings["total"] = t_total
        fps_window.append(1000.0 / max(t_total, 1e-6))
        current_fps = sum(fps_window) / len(fps_window)

        # --- Visualization ---
        vis_frame = proc_frame.copy()
        if len(keypoints) > 0:
            draw_skeleton(vis_frame, keypoints, scores, kpt_thr=0.3)

        # Draw bboxes
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        draw_hud(vis_frame, results, current_fps, timings, frame_count)

        cv2.imshow("ONNX Action Recognition", vis_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames. Avg FPS: {current_fps:.1f}")


# ---------------------------------------------------------------------------
#  Benchmark mode
# ---------------------------------------------------------------------------


def run_benchmark(args):
    """Benchmark each stage independently."""
    import onnxruntime as ort

    recog_device = args.recog_device or args.device
    print("\n=== ONNX Pipeline Benchmark ===")
    print(f"Device: det/pose={args.device}, recog={recog_device}")

    model_dir = Path(args.model_dir)
    detector = YOLOXDetector(str(model_dir / args.det_model), device=args.device)
    pose_estimator = RTMPoseEstimator(
        str(model_dir / args.pose_model), device=args.device
    )
    recognizer = STGCNRecognizer(
        str(model_dir / args.recog_model), args.label_map, device=recog_device
    )

    N = args.benchmark_iters
    h, w = 480, 640
    dummy_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        detector(dummy_img)
        pose_estimator(dummy_img, [[100, 100, 400, 400]])
        recognizer([np.zeros((1, 17, 2))], [np.zeros((1, 17))], img_shape=(h, w))

    # Detection benchmark
    print(f"\nBenchmarking {N} iterations...")
    t0 = time.perf_counter()
    for _ in range(N):
        detector(dummy_img)
    det_ms = (time.perf_counter() - t0) / N * 1000

    # Pose benchmark (1 person)
    t0 = time.perf_counter()
    for _ in range(N):
        pose_estimator(dummy_img, [[100, 100, 400, 400]])
    pose_ms = (time.perf_counter() - t0) / N * 1000

    # Recognition benchmark
    kpts_buf = [np.random.randn(1, 17, 2) for _ in range(100)]
    scores_buf = [np.random.rand(1, 17) for _ in range(100)]
    t0 = time.perf_counter()
    for _ in range(N):
        recognizer(kpts_buf, scores_buf, img_shape=(h, w))
    recog_ms = (time.perf_counter() - t0) / N * 1000

    total_ms = det_ms + pose_ms + recog_ms
    fps = 1000.0 / total_ms

    print(f'\n{"Stage":<20} {"Time (ms)":<12} {"FPS":<10}')
    print("-" * 42)
    print(f'{"Detection":<20} {det_ms:<12.1f} {1000/det_ms:<10.1f}')
    print(f'{"Pose (1 person)":<20} {pose_ms:<12.1f} {1000/pose_ms:<10.1f}')
    print(f'{"Recognition":<20} {recog_ms:<12.1f} {1000/recog_ms:<10.1f}')
    print("-" * 42)
    print(f'{"TOTAL":<20} {total_ms:<12.1f} {fps:<10.1f}')
    print(f"\n→ Estimated pipeline: {fps:.1f} FPS ({args.device})")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Pure-ONNX Real-Time Skeleton Action Recognition"
    )

    # Modes
    p.add_argument(
        "--clip", type=str, default=None, help="Path to video file (default: webcam)"
    )
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode instead of live demo",
    )

    # Device
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument(
        "--recog-device",
        choices=["cuda", "cpu"],
        default="cpu",
        help="Device for STGCN++ recognizer (default: cpu, avoids cuDNN issues)",
    )

    # Model paths
    p.add_argument(
        "--model-dir",
        default="demo/onnx_models",
        help="Directory containing ONNX models",
    )
    p.add_argument("--det-model", default="yolox_tiny.onnx")
    p.add_argument("--pose-model", default="rtmpose_m.onnx")
    p.add_argument("--recog-model", default="stgcnpp_ntu120_xsub_hrnet_j.onnx")
    p.add_argument("--label-map", default="tools/data/label_map/nturgbd_120.txt")

    # Pipeline
    p.add_argument(
        "--window-frames",
        type=int,
        default=30,
        help="Sliding window size (frames to accumulate)",
    )
    p.add_argument(
        "--clip-len",
        type=int,
        default=100,
        help="Temporal length for STGCN++ (uniform sampled)",
    )
    p.add_argument("--num-person", type=int, default=2, help="Max persons for STGCN++")
    p.add_argument(
        "--min-frames", type=int, default=8, help="Min frames before first recognition"
    )

    # Detection
    p.add_argument("--det-score-thr", type=float, default=0.5)
    p.add_argument(
        "--det-every",
        type=int,
        default=1,
        help="Run detection every N frames (1=every frame)",
    )

    # Resolution
    p.add_argument(
        "--short-side",
        type=int,
        default=0,
        help="Resize short side to this (0=no resize)",
    )

    # Benchmark
    p.add_argument("--benchmark-iters", type=int, default=100)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.benchmark:
        run_benchmark(args)
    else:
        run_stream(args)
