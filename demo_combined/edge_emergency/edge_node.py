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
import ssl
import subprocess
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

# Severities that can raise a latched alert / phone notification.
# Demo decision (2026-06-11): only CRITICAL (falling) alerts -- staggering/nausea
# (HIGH) still show as the live status, but must NOT trigger the alarm/notification.
EMERGENCY_SEVERITIES = {"CRITICAL"}


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
        self._vitals = None          # latest Apple Watch vitals (HR/SpO2/temp/anomaly)
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

    def set_vitals(self, vitals):
        vitals["_rx"] = time.time()   # Pi-clock receipt time (for freshness)
        with self._lock:
            self._vitals = vitals

    def get_vitals(self):
        with self._lock:
            return self._vitals


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

    def __init__(self, conf_thr=0.55, window=5, min_hits=3, hold_s=8.0, refire_s=5.0):
        self.conf_thr = conf_thr
        self.window = window
        self.min_hits = min_hits
        self.hold_s = hold_s
        self.refire_s = refire_s             # re-fire interval while still active
        self.history = deque(maxlen=window)  # (label, conf, severity)
        self.alert = None                    # dict or None
        self.alert_count = 0                 # total distinct alert episodes
        self.fire_seq = 0                    # increments on every (re)fire

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
                self.fire_seq += 1
                self.alert = {
                    "label": candidate,
                    "severity": cand_sev,
                    "started_at": now,
                    "last_seen": now,
                    "last_fired": now,
                }
                self.alert_count += 1
                just_fired = True
            else:
                # Same emergency persists -> re-fire every refire_s so a person
                # who STAYS fallen keeps raising alerts/notifications.
                self.alert["last_seen"] = now
                if now - self.alert["last_fired"] >= self.refire_s:
                    self.alert["last_fired"] = now
                    self.fire_seq += 1
                    just_fired = True
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
            "fires": self.fire_seq,
        }


# ---------------------------------------------------------------------------
#  Inference worker — captures frames, runs pipeline, updates STATE
# ---------------------------------------------------------------------------


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def resolve_source(args):
    """Pick the OpenCV capture source: --source URL > --video file > --camera index.

    A string (URL/path) is passed straight to cv2.VideoCapture, so a phone running
    an IP-webcam app (http://<phone-ip>:8080/video) works as a wireless camera.
    """
    if args.source:
        return args.source
    if args.video:
        return args.video
    return args.camera


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

_VIGNETTE_CACHE = {}

def _red_vignette(frame, strength=0.65):
    """Blend a soft red vignette into the frame edges in-place (emergency cue)."""
    h, w = frame.shape[:2]
    m = _VIGNETTE_CACHE.get((h, w))
    if m is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2.0) / (w / 2.0)) ** 2 + ((yy - h / 2.0) / (h / 2.0)) ** 2)
        m = (np.clip((r - 0.55) / 0.6, 0.0, 1.0) ** 1.5)[..., None]  # 0 center -> 1 edges
        _VIGNETTE_CACHE[(h, w)] = m
    a = m * strength
    red = np.array([45, 45, 235], dtype=np.float32)  # BGR
    np.copyto(frame, (frame.astype(np.float32) * (1.0 - a) + red * a).astype(np.uint8))


def annotate(frame, keypoints, scores, bboxes, label, conf, severity, alert, fps):
    """Draw a CLEAN skeleton overlay (+ a bottom emergency banner only).

    The action label, confidence, FPS etc. are shown by the dashboard panels, so
    we deliberately do NOT burn them onto the video — that keeps the feed clean
    and avoids the duplicated/overlapping text the old HUD produced.
    """
    if len(keypoints) > 0:
        demo_onnx.draw_skeleton(frame, keypoints, scores, kpt_thr=0.3)

    # On an active emergency, tint the frame edges with a soft red vignette —
    # cleaner than burned-in low-res text; the dashboard shows the details.
    if alert and alert.get("active"):
        _red_vignette(frame)
    return frame


def inference_loop(args, stop_event):
    detector, pose, recog = build_pipeline(args)
    engine = AlertEngine(
        conf_thr=args.alert_conf, window=args.alert_window,
        min_hits=args.alert_hits, hold_s=args.alert_hold, refire_s=args.alert_refire,
    )

    source = resolve_source(args)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source {source!r}")
        stop_event.set()
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low on live cameras
    except Exception:
        pass

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
    loop_prev = None  # previous iteration start (for true throughput incl. fps cap)

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
        if loop_prev is not None:                       # true end-to-end rate
            fps_win.append(1.0 / max(t0 - loop_prev, 1e-6))
        loop_prev = t0
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

        dt = time.perf_counter() - t0  # processing latency (excludes fps-cap sleep)
        fps = (sum(fps_win) / len(fps_win)) if fps_win else (1.0 / max(dt, 1e-6))
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
                "vitals": STATE.get_vitals(),   # Apple Watch HR/SpO2/temp/anomaly
                "uptime_s": round(now - STATE.started, 1),
            })

        # Cap the processing rate: paces file playback AND throttles a live camera
        # to keep CPU load / temperature down (applies to every source now).
        if args.max_fps > 0:
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

# --- PWA assets (served in-memory so a phone can "install" the dashboard) ------
_THEME = "#0a0e13"

_MANIFEST = json.dumps({
    "name": "Smart-Care Monitor",
    "short_name": "Smart-Care",
    "description": "Live edge emergency detection",
    "start_url": "/",
    "scope": "/",
    "display": "standalone",
    "orientation": "any",
    "background_color": _THEME,
    "theme_color": _THEME,
    "icons": [
        {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"},
        {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
    ],
}).encode("utf-8")

# Minimal service worker: enables install + lets the page raise OS notifications.
_SW_JS = (
    b"self.addEventListener('install',e=>self.skipWaiting());\n"
    b"self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));\n"
    b"self.addEventListener('notificationclick',e=>{e.notification.close();"
    b"e.waitUntil(clients.matchAll({type:'window'}).then(c=>c.length?c[0].focus():clients.openWindow('/')));});\n"
)

_ICON_PTS = "64,272 176,272 204,264 230,144 270,388 300,238 322,280 452,280"  # ECG heartbeat
_ICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
    '<rect width="512" height="512" rx="116" fill="#0d9488"/>'
    f'<polyline points="{_ICON_PTS}" fill="none" stroke="#ffffff" stroke-width="26" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
).encode("utf-8")

# Compact, glanceable status page for a phone home-screen "web widget" app.
WIDGET_HTML = b"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Smart-Care</title>
<style>html,body{margin:0;height:100%;background:#0a0e14;color:#eaf0f6;
font-family:Inter,system-ui,Arial}.w{height:100vh;display:flex;flex-direction:column;
justify-content:center;padding:5%;box-sizing:border-box}.st{font-size:8vw;font-weight:800;
line-height:1.05}.dot{display:inline-block;width:.6em;height:.6em;border-radius:50%;margin-right:.35em}
.sub{font-size:3.6vw;color:#7d8c9e;margin-top:6px}.row{display:flex;gap:6%;margin-top:8px;
font-family:ui-monospace,monospace}.m b{font-size:6vw}.m span{font-size:3.2vw;color:#7d8c9e}</style>
</head><body><div class="w"><div class="st" id="st"><span class="dot" id="dot"></span><span id="t">—</span></div>
<div class="sub" id="sub">connecting…</div><div class="row"><div class="m"><b id="hr">—</b><span> bpm</span></div>
<div class="m"><b id="o2">—</b><span> %SpO2</span></div></div></div><script>
var SEV={CRITICAL:"#f43f5e",HIGH:"#fb923c",MEDIUM:"#fbbf24",LOW:"#a3e635",NORMAL:"#34d399"};
function g(i){return document.getElementById(i)}
function tick(){fetch("/state",{cache:"no-store"}).then(function(r){return r.json()}).then(function(d){
var a=d.alert||{active:false},txt,col;
if(a.active&&a.severity==="CRITICAL"){txt="FALL DETECTED";col=SEV.CRITICAL}
else{txt=d.action||"—";col=SEV[d.severity]||"#34d399"}
g("t").textContent=txt;g("dot").style.background=col;g("st").style.color=col;
var v=d.vitals||{};g("hr").textContent=v.heartRate?Math.round(v.heartRate):"—";
g("o2").textContent=v.spo2?Math.round(v.spo2):"—";
g("sub").textContent="updated "+new Date().toLocaleTimeString([],{hour12:false});
}).catch(function(){g("sub").textContent="offline"})}
tick();setInterval(tick,2000);</script></body></html>"""

_ICON_PNG_CACHE = {}

def _icon_png(size):
    """Render the heartbeat app-icon to PNG via cv2 (cached)."""
    if size in _ICON_PNG_CACHE:
        return _ICON_PNG_CACHE[size]
    s = size / 512.0
    img = np.empty((size, size, 3), np.uint8)
    img[:] = (136, 148, 13)  # BGR of #0d9488 teal
    pts = np.array([[int(x) for x in p.split(",")] for p in _ICON_PTS.split()], np.float32)
    pts = (pts * s).astype(np.int32)
    cv2.polylines(img, [pts], False, (255, 255, 255), max(2, int(26 * s)), cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    data = buf.tobytes() if ok else b""
    _ICON_PNG_CACHE[size] = data
    return data


_CERT_DIR = Path(__file__).resolve().parent / ".certs"

def ensure_self_signed_cert(ip=None):
    """Return (cert_path, key_path), generating a self-signed cert via openssl if absent."""
    _CERT_DIR.mkdir(exist_ok=True)
    cert = _CERT_DIR / "cert.pem"
    key = _CERT_DIR / "key.pem"
    if cert.exists() and key.exists():
        return str(cert), str(key)
    san = "subjectAltName=DNS:localhost,DNS:smartcare.local,IP:127.0.0.1,IP:10.42.0.1"
    if ip and ip != "127.0.0.1":
        san += f",IP:{ip}"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(key), "-out", str(cert), "-days", "825",
         "-subj", "/CN=Smart-Care Edge", "-addext", san],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return str(cert), str(key)


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
        elif path == "/manifest.webmanifest":
            self._send_bytes(200, "application/manifest+json", _MANIFEST)
        elif path == "/sw.js":
            self._send_bytes(200, "application/javascript", _SW_JS)
        elif path == "/widget":
            self._send_bytes(200, "text/html; charset=utf-8", WIDGET_HTML)
        elif path == "/icon.svg":
            self._send_bytes(200, "image/svg+xml", _ICON_SVG)
        elif path in ("/icon-192.png", "/icon-512.png", "/icon-180.png"):
            self._send_bytes(200, "image/png", _icon_png(int(path.split("-")[1].split(".")[0])))
        else:
            self._send_headers(404, "text/plain", length=9)
            self.wfile.write(b"not found")

    def _send_bytes(self, code, ctype, data):
        self._send_headers(code, ctype, length=len(data))
        self.wfile.write(data)

    def do_POST(self):
        # Ingest Apple Watch vitals from the iPhone companion app:
        #   POST /vitals  {"heartRate":72,"spo2":98,"temperature":36.6,
        #                  "anomaly":"Normal","isAnomaly":false,"emergency":false}
        path = self.path.split("?", 1)[0]
        if path == "/vitals":
            try:
                n = int(self.headers.get("Content-Length", 0) or 0)
                data = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
                STATE.set_vitals(data)
                self._send_bytes(200, "application/json", b'{"ok":true}')
            except Exception as e:
                self._send_bytes(400, "application/json",
                                 json.dumps({"ok": False, "error": str(e)}).encode())
        else:
            self._send_headers(404, "text/plain", length=9)
            self.wfile.write(b"not found")

    def do_OPTIONS(self):  # CORS preflight (harmless; iOS doesn't need it)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

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
                         min_hits=args.alert_hits, hold_s=args.alert_hold,
                         refire_s=args.alert_refire)
    source = resolve_source(args)
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
    p.add_argument("--source", type=str, default=None,
                   help="Network video stream URL — use a phone as the camera: run an "
                        "IP-webcam app and pass e.g. http://<phone-ip>:8080/video (MJPEG) "
                        "or rtsp://<phone-ip>:.../  . Overrides --camera/--video.")
    p.add_argument("--loop", action="store_true", help="Loop the video file forever")
    p.add_argument("--max-fps", type=float, default=25.0,
                   help="Cap playback FPS when reading a file (0=unlimited)")
    # Server
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--https", action="store_true",
                   help="Serve over HTTPS with an auto-generated self-signed cert. "
                        "Required for phone OS notifications + 'add to home screen' (PWA). "
                        "Browsers will warn about the cert once — accept it to proceed.")
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
    p.add_argument("--alert-refire", type=float, default=5.0,
                   help="Re-fire an active alert every N seconds while it persists "
                        "(so a person who stays fallen keeps raising notifications).")
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
    ip = get_lan_ip()
    scheme = "http"
    if args.https:
        try:
            cert, key = ensure_self_signed_cert(ip)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            server.socket = ctx.wrap_socket(server.socket, server_side=True)
            scheme = "https"
        except Exception as e:
            print(f"  ! HTTPS setup failed ({e}); serving plain HTTP instead.")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("\n" + "=" * 60)
    print("  Smart-Care Edge Emergency Node is LIVE")
    print(f"  Open the dashboard:  {scheme}://{ip}:{args.port}/")
    print(f"  (local: {scheme}://127.0.0.1:{args.port}/)")
    if scheme == "https":
        print("  NOTE: self-signed cert -> the browser warns once; tap Advanced -> Proceed.")
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
