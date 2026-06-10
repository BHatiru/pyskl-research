#!/usr/bin/env python
"""Smart-Care Edge Emergency Node.

Runs the pure-ONNX skeleton HAR pipeline (YOLOX-tiny -> RTMPose -> STGCN++) on a
CPU edge device (Raspberry Pi 5), classifies medical actions, raises debounced
emergency alerts (falling / staggering / nausea), and serves everything to a
laptop browser over the LAN using ONLY the Python standard library:

    GET /              -> the dark live dashboard (dashboard.html)
    GET /stream.mjpg   -> annotated MJPEG video stream
    GET /events        -> Server-Sent-Events feed of structured detections/alerts
    GET /state         -> current state as JSON (poll fallback)

Design goals:
  * Pi-friendly: deps are numpy + opencv + onnxruntime only (no FastAPI/torch).
  * Robust for a live demo: inference runs in its own thread; HTTP handlers only
    read shared state, so a flaky network never stalls the camera.
  * Pull-over-LAN: the laptop just opens http://<pi-ip>:8000/  -> zero setup.

Usage (laptop self-test on the bundled sample video, no display/camera needed):
    python demo_combined/edge_emergency/edge_node.py --selftest 120 \
        --video demo/ntu_sample.avi

Usage (live, on the Pi or laptop with a camera):
    python demo_combined/edge_emergency/edge_node.py --camera 0 \
        --short-side 320 --det-every 2 --recog-every 5 --threads 4

Then open http://<this-host-ip>:8000/ in a browser.
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# --- Reuse the proven pure-ONNX pipeline classes from demo/demo_onnx.py -------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEMO_DIR = _REPO_ROOT / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

import demo_onnx  # noqa: E402  (YOLOXDetector, RTMPoseEstimator, STGCNRecognizer, draw_skeleton)

# Pluggable pose front-ends (rtmpose two-stage / movenet single-stage)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_backends import MoveNetEstimator, run_frontend, has_person  # noqa: E402

# ---------------------------------------------------------------------------
#  Medical action -> emergency severity  (matches demo_combined/inference.py)
# ---------------------------------------------------------------------------

SEVERITY = {
    "falling": "CRITICAL",
    "staggering": "HIGH",
    "nausea/vomiting": "HIGH",
    "touch head": "MEDIUM",
    "touch chest": "MEDIUM",
    "touch back": "MEDIUM",
    "touch neck": "MEDIUM",
    "sneeze/cough": "LOW",
    "standing up": "NORMAL",
    "sitting down": "NORMAL",
    "walking towards": "NORMAL",
    "walking apart": "NORMAL",
    "drinking": "NORMAL",
    "eating": "NORMAL",
    "phone call": "NORMAL",
}

# Severity rank for comparisons (higher = more urgent)
SEV_RANK = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

# Classes that can raise an alert
EMERGENCY_SEVERITIES = {"HIGH", "CRITICAL"}


def severity_of(label):
    return SEVERITY.get(label, "NORMAL")


# ---------------------------------------------------------------------------
#  Shared state (single writer = inference thread; many readers = HTTP handlers)
# ---------------------------------------------------------------------------


class SharedState:
    """Thread-safe snapshot of the latest annotated frame + detection result."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None            # latest annotated frame as JPEG bytes
        self._frame_seq = 0          # increments every encoded frame
        self._event = None           # latest detection/alert dict
        self._event_seq = 0          # increments every recognition result
        self.started = time.time()

    # --- writers ---
    def set_frame(self, jpeg_bytes):
        with self._lock:
            self._jpeg = jpeg_bytes
            self._frame_seq += 1

    def set_event(self, event):
        with self._lock:
            self._event = event
            self._event_seq += 1

    # --- readers ---
    def get_frame(self):
        with self._lock:
            return self._jpeg, self._frame_seq

    def get_event(self):
        with self._lock:
            return self._event, self._event_seq


STATE = SharedState()


# ---------------------------------------------------------------------------
#  Emergency alert engine — temporal debounce + latched alert
# ---------------------------------------------------------------------------


class AlertEngine:
    """Turns a stream of per-recognition top-1 predictions into stable alerts.

    A single noisy frame must NOT trigger an alarm in front of funders, so an
    emergency is confirmed only when an emergency class is top-1 with confidence
    >= conf_thr in at least `min_hits` of the last `window` recognitions. Once
    confirmed, the alert latches for `hold_s` seconds (so it stays visible),
    refreshing while the emergency keeps recurring.
    """

    def __init__(self, conf_thr=0.55, window=5, min_hits=3, hold_s=8.0):
        self.conf_thr = conf_thr
        self.window = window
        self.min_hits = min_hits
        self.hold_s = hold_s
        self.history = deque(maxlen=window)  # (label, conf, severity)
        self.alert = None                    # dict or None
        self.alert_count = 0                 # total alerts raised this session

    def update(self, label, conf, now):
        """Feed one recognition result. Returns (alert_dict_or_None, just_fired)."""
        sev = severity_of(label)
        self.history.append((label, conf, sev))

        # Count confident emergency votes per emergency label in the window
        votes = {}
        for lbl, c, s in self.history:
            if s in EMERGENCY_SEVERITIES and c >= self.conf_thr:
                votes[lbl] = votes.get(lbl, 0) + 1

        # Pick the most-voted emergency label that crosses the threshold,
        # breaking ties by severity rank.
        candidate = None
        if votes:
            candidate = max(
                votes.items(),
                key=lambda kv: (kv[1], SEV_RANK[severity_of(kv[0])]),
            )
            cand_label, cand_hits = candidate
            if cand_hits < self.min_hits:
                candidate = None
            else:
                candidate = cand_label

        just_fired = False
        if candidate is not None:
            cand_sev = severity_of(candidate)
            if self.alert is None or self.alert["label"] != candidate:
                # New alert (or escalated to a different emergency)
                self.alert = {
                    "label": candidate,
                    "severity": cand_sev,
                    "started_at": now,
                    "last_seen": now,
                }
                self.alert_count += 1
                just_fired = True
            else:
                self.alert["last_seen"] = now
        elif self.alert is not None:
            # No fresh emergency vote — clear once the hold window expires
            if now - self.alert["last_seen"] > self.hold_s:
                self.alert = None

        return self.alert, just_fired

    def snapshot(self, now):
        if self.alert is None:
            return {"active": False}
        return {
            "active": True,
            "label": self.alert["label"],
            "severity": self.alert["severity"],
            "age_s": round(now - self.alert["started_at"], 1),
            "count": self.alert_count,
        }


# ---------------------------------------------------------------------------
#  Inference worker — captures frames, runs pipeline, updates STATE
# ---------------------------------------------------------------------------


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def build_pipeline(args):
    model_dir = Path(args.model_dir)
    threads = args.threads
    print("Loading models...")
    if args.pose_backend == "movenet":
        # Single-stage: MoveNet localises + poses one person, NO separate detector.
        mv = args.movenet_model or str(model_dir / "movenet_thunder_int8.tflite")
        detector = None
        pose = MoveNetEstimator(mv, threads=threads)
    else:
        detector = demo_onnx.YOLOXDetector(
            str(model_dir / args.det_model), device="cpu",
            score_thr=args.det_score_thr, threads=threads,
        )
        pose = demo_onnx.RTMPoseEstimator(
            str(model_dir / args.pose_model), device="cpu", threads=threads,
        )
    recog = demo_onnx.STGCNRecognizer(
        str(model_dir / args.recog_model), args.label_map, device="cpu",
        clip_len=args.clip_len, num_person=args.num_person, threads=threads,
    )
    return detector, pose, recog


SEV_BGR = {
    "CRITICAL": (60, 60, 230),
    "HIGH": (40, 130, 245),
    "MEDIUM": (40, 200, 245),
    "LOW": (120, 200, 90),
    "NORMAL": (160, 200, 90),
}


def annotate(frame, keypoints, scores, bboxes, label, conf, severity, alert, fps):
    """Draw skeleton + action label + severity banner onto the frame."""
    if len(keypoints) > 0:
        demo_onnx.draw_skeleton(frame, keypoints, scores, kpt_thr=0.3)
    for bb in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bb]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)

    h, w = frame.shape[:2]
    # Top action bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    col = SEV_BGR.get(severity, (200, 200, 200))
    txt = f"{label} ({conf:.0%})" if label else "...detecting..."
    cv2.putText(frame, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} FPS", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # Emergency banner (bottom) when an alert is latched
    if alert and alert.get("active"):
        bcol = SEV_BGR.get(alert["severity"], (60, 60, 230))
        ov = frame.copy()
        cv2.rectangle(ov, (0, h - 56), (w, h), bcol, -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
        msg = f"! {alert['severity']} EMERGENCY: {alert['label'].upper()}"
        cv2.putText(frame, msg, (14, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.95, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def inference_loop(args, stop_event):
    detector, pose, recog = build_pipeline(args)
    engine = AlertEngine(
        conf_thr=args.alert_conf, window=args.alert_window,
        min_hits=args.alert_hits, hold_s=args.alert_hold,
    )

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source {source!r}")
        stop_event.set()
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    if args.short_side > 0:
        sc = args.short_side / max(1, min(src_w, src_h))
        proc_w, proc_h = int(src_w * sc), int(src_h * sc)
    else:
        proc_w, proc_h = src_w, src_h
    print(f"Source {source!r}: {src_w}x{src_h} -> processing {proc_w}x{proc_h}")

    kpts_buf = deque(maxlen=args.window_frames)
    scores_buf = deque(maxlen=args.window_frames)
    present_buf = deque(maxlen=args.window_frames)  # per-frame person-presence gate
    fps_win = deque(maxlen=30)

    cached_bboxes = []
    results = []
    cur_label, cur_conf, cur_sev = "", 0.0, "NORMAL"
    prob_ema = None  # EMA-smoothed 15-class probability vector
    frame_count = 0
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    # Warmup
    dummy = np.random.randint(0, 255, (proc_h, proc_w, 3), dtype=np.uint8)
    for _ in range(2):
        if detector is not None:
            detector(dummy)
        pose(dummy, [[0, 0, proc_w, proc_h]])

    while not stop_event.is_set():
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            if args.video and args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("Source ended.")
            break
        frame_count += 1
        if (proc_w, proc_h) != (src_w, src_h):
            frame = cv2.resize(frame, (proc_w, proc_h))

        # Front-end: detection (skip cadence) + pose, or single-stage MoveNet
        keypoints, scores, bboxes, cached_bboxes = run_frontend(
            detector, pose, frame, frame_count, args.det_every, cached_bboxes
        )
        kpts_buf.append(keypoints)
        scores_buf.append(scores)
        present_buf.append(has_person(scores))
        cur_present = bool(present_buf[-1])

        # Recognition (skip frames for speed)
        now = time.time()
        new_result = False
        if len(kpts_buf) >= args.min_frames and (
            frame_count % args.recog_every == 0 or args.recog_every == 1
        ):
            if sum(present_buf) >= args.min_frames:
                inp = recog.build_input(list(kpts_buf), list(scores_buf), img_shape=(proc_h, proc_w))
                probs = _softmax(recog.session.run(None, {recog.input_name: inp})[0][0])
                # EMA-smooth the class distribution so a single jittery frame can't
                # flip the decision (kills transient 'staggering' false alarms),
                # while a sustained real fall still builds confidence quickly.
                prob_ema = probs if prob_ema is None else (
                    args.prob_ema * probs + (1.0 - args.prob_ema) * prob_ema)
                order = np.argsort(prob_ema)[::-1]
                idx = int(order[0])
                cur_label, cur_conf = recog.labels[idx], float(prob_ema[idx])
                cur_sev = severity_of(cur_label)
                results = [(int(i), recog.labels[i], float(prob_ema[i])) for i in order[:5]]
                engine.update(cur_label, cur_conf, now)
            else:
                # Too few frames with a real person -> don't hallucinate an emergency.
                results = []
                cur_label, cur_conf, cur_sev = "no person", 0.0, "NORMAL"
                prob_ema = None  # reset smoothing when nobody is present
                engine.update("", 0.0, now)  # benign vote: lets a latched alert decay
            new_result = True

        dt = time.perf_counter() - t0
        fps_win.append(1.0 / max(dt, 1e-6))
        fps = sum(fps_win) / len(fps_win)
        alert_snap = engine.snapshot(now)

        # Encode annotated frame for MJPEG (don't draw a skeleton when no one's there)
        if not args.no_video:
            vis_kp = keypoints if cur_present else np.zeros((0, 17, 2))
            vis_sc = scores if cur_present else np.zeros((0, 17))
            vis = annotate(frame.copy(), vis_kp, vis_sc, bboxes,
                           cur_label, cur_conf, cur_sev, alert_snap, fps)
            ok, buf = cv2.imencode(".jpg", vis, jpeg_params)
            if ok:
                STATE.set_frame(buf.tobytes())

        # Publish a structured event on every fresh recognition
        if new_result:
            top5 = [
                {"label": lbl, "prob": round(p, 4), "severity": severity_of(lbl)}
                for _, lbl, p in results[:5]
            ]
            STATE.set_event({
                "ts": now,
                "action": cur_label,
                "confidence": round(cur_conf, 4),
                "severity": cur_sev,
                "top5": top5,
                "fps": round(fps, 1),
                "latency_ms": round(dt * 1000, 1),
                "persons": int(cur_present),
                "alert": alert_snap,
                "uptime_s": round(now - STATE.started, 1),
            })

        # Pace to target fps when reading a file (so playback isn't too fast)
        if args.video and args.max_fps > 0:
            sleep = (1.0 / args.max_fps) - dt
            if sleep > 0:
                time.sleep(sleep)

    cap.release()
    stop_event.set()


# ---------------------------------------------------------------------------
#  HTTP server (stdlib) — dashboard + MJPEG + SSE + JSON
# ---------------------------------------------------------------------------

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

_DASHBOARD_PATH = Path(__file__).resolve().parent / "dashboard.html"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quieter console
        pass

    def _send_headers(self, code, ctype, extra=None, length=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._serve_dashboard()
        elif path == "/stream.mjpg":
            self._serve_mjpeg()
        elif path == "/events":
            self._serve_sse()
        elif path == "/state":
            self._serve_state()
        elif path == "/healthz":
            self._send_headers(200, "text/plain", length=2)
            self.wfile.write(b"ok")
        else:
            self._send_headers(404, "text/plain", length=9)
            self.wfile.write(b"not found")

    def _serve_dashboard(self):
        try:
            data = _DASHBOARD_PATH.read_bytes()
        except FileNotFoundError:
            data = b"dashboard.html not found next to edge_node.py"
            self._send_headers(500, "text/plain", length=len(data))
            self.wfile.write(data)
            return
        self._send_headers(200, "text/html; charset=utf-8", length=len(data))
        self.wfile.write(data)

    def _serve_state(self):
        event, _ = STATE.get_event()
        data = json.dumps(event or {"action": None}).encode("utf-8")
        self._send_headers(200, "application/json", length=len(data))
        self.wfile.write(data)

    def _serve_mjpeg(self):
        self._send_headers(
            200, "multipart/x-mixed-replace; boundary=frame",
            extra={"Connection": "close"},
        )
        last_seq = -1
        try:
            while True:
                jpeg, seq = STATE.get_frame()
                if jpeg is not None and seq != last_seq:
                    last_seq = seq
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _serve_sse(self):
        self._send_headers(200, "text/event-stream", extra={"Connection": "close"})
        last_seq = -1
        try:
            # Send a hello so the client flips to "connected" immediately
            self.wfile.write(b": connected\n\n")
            while True:
                event, seq = STATE.get_event()
                if event is not None and seq != last_seq:
                    last_seq = seq
                    payload = json.dumps(event)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                else:
                    self.wfile.write(b": ping\n\n")  # keep-alive
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


def get_lan_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
#  Self-test (headless) — verify pipeline/severity/debounce without a server
# ---------------------------------------------------------------------------


def run_selftest(args, n_frames):
    detector, pose, recog = build_pipeline(args)
    engine = AlertEngine(conf_thr=args.alert_conf, window=args.alert_window,
                         min_hits=args.alert_hits, hold_s=args.alert_hold)
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source {source!r}")
        return 1
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    if args.short_side > 0:
        sc = args.short_side / max(1, min(src_w, src_h))
        proc_w, proc_h = int(src_w * sc), int(src_h * sc)
    else:
        proc_w, proc_h = src_w, src_h

    kpts_buf = deque(maxlen=args.window_frames)
    scores_buf = deque(maxlen=args.window_frames)
    present_buf = deque(maxlen=args.window_frames)
    cached, fc, recogs, alerts = [], 0, 0, 0
    t_start = time.perf_counter()
    print(f"Self-test: {n_frames} frames from {source!r} @ {proc_w}x{proc_h}")
    while fc < n_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        fc += 1
        if (proc_w, proc_h) != (src_w, src_h):
            frame = cv2.resize(frame, (proc_w, proc_h))
        kp, sc2, _, cached = run_frontend(
            detector, pose, frame, fc, args.det_every, cached
        )
        kpts_buf.append(kp)
        scores_buf.append(sc2)
        present_buf.append(has_person(sc2))
        if len(kpts_buf) >= args.min_frames and (fc % args.recog_every == 0):
            if sum(present_buf) < args.min_frames:
                engine.update("", 0.0, time.time())  # no person -> benign
                continue
            res = recog(list(kpts_buf), list(scores_buf), img_shape=(proc_h, proc_w))
            recogs += 1
            lbl, conf = res[0][1], res[0][2]
            alert, fired = engine.update(lbl, conf, time.time())
            if fired:
                alerts += 1
                print(f"  [frame {fc:4d}] ALERT {alert['severity']:8} "
                      f"{alert['label']:16} (top1 {lbl} {conf:.0%})")
            elif recogs % 5 == 0:
                print(f"  [frame {fc:4d}] {lbl:16} {conf:5.0%}  sev={severity_of(lbl)}")
    cap.release()
    elapsed = time.perf_counter() - t_start
    print(f"\nProcessed {fc} frames in {elapsed:.1f}s "
          f"({fc/elapsed:.1f} FPS), {recogs} recognitions, {alerts} alerts raised.")
    print("Self-test OK.")
    return 0


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Smart-Care Edge Emergency Node")
    # Source
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument("--video", type=str, default=None, help="Video file instead of webcam")
    p.add_argument("--loop", action="store_true", help="Loop the video file forever")
    p.add_argument("--max-fps", type=float, default=25.0,
                   help="Cap playback FPS when reading a file (0=unlimited)")
    # Server
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-video", action="store_true", help="Disable MJPEG encoding")
    p.add_argument("--jpeg-quality", type=int, default=80)
    # Models
    p.add_argument("--model-dir", default=str(_DEMO_DIR / "onnx_models"))
    p.add_argument("--pose-backend", choices=["rtmpose", "movenet"], default="rtmpose",
                   help="rtmpose = YOLOX+RTMPose two-stage (proven); "
                        "movenet = MoveNet single-stage (~4x faster front-end, single-person)")
    p.add_argument("--movenet-model", default="",
                   help="MoveNet TFLite path (default: <model-dir>/movenet_thunder_int8.tflite). "
                        "Use Thunder, not Lightning — Lightning misses falls.")
    p.add_argument("--det-model", default="yolox_tiny.onnx")
    p.add_argument("--pose-model", default="rtmpose_m.onnx")
    p.add_argument("--recog-model", default="stgcnpp_medical15_cent2d.onnx")
    p.add_argument("--label-map", default=str(_REPO_ROOT / "tools/data/label_map/medical_15.txt"))
    # Pipeline
    p.add_argument("--short-side", type=int, default=320, help="Resize short side (0=none)")
    p.add_argument("--window-frames", type=int, default=30)
    p.add_argument("--clip-len", type=int, default=100)
    p.add_argument("--num-person", type=int, default=2)
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--det-every", type=int, default=2)
    p.add_argument("--recog-every", type=int, default=5)
    p.add_argument("--det-score-thr", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=4)
    # Alert engine
    p.add_argument("--prob-ema", type=float, default=0.5,
                   help="EMA weight on each new recognition's probabilities "
                        "(1.0=no smoothing, lower=smoother/steadier; 0.5 default)")
    p.add_argument("--alert-conf", type=float, default=0.55)
    p.add_argument("--alert-window", type=int, default=5)
    p.add_argument("--alert-hits", type=int, default=3)
    p.add_argument("--alert-hold", type=float, default=8.0)
    # Modes
    p.add_argument("--selftest", type=int, default=0,
                   help="Run N frames headless (no server) and exit")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest > 0:
        sys.exit(run_selftest(args, args.selftest))

    stop = threading.Event()
    worker = threading.Thread(target=inference_loop, args=(args, stop), daemon=True)
    worker.start()

    # Give the worker a moment to load models / open the source before serving
    time.sleep(0.5)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ip = get_lan_ip()
    print("\n" + "=" * 60)
    print("  Smart-Care Edge Emergency Node is LIVE")
    print(f"  Open the dashboard on the laptop:  http://{ip}:{args.port}/")
    print(f"  (local: http://127.0.0.1:{args.port}/)")
    print("=" * 60 + "\n")
    try:
        while not stop.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop.set()
        server.shutdown()


if __name__ == "__main__":
    main()
