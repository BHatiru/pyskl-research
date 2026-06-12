#!/usr/bin/env python
"""Smart-Care Control Panel — a laptop GUI to drive the Raspberry Pi edge node
over SSH without typing commands.

Connect to the Pi, start/stop the demo, change boot parameters, configure the
Pi's Wi-Fi, and open the dashboard — all with buttons. Wraps the same SSH
commands we use by hand.

Run:   python control_panel.py
Deps:  paramiko  (pip install paramiko).  GUI = Tkinter (stdlib).
Package to a standalone Windows .exe with:
       pyinstaller --onefile --noconsole --name SmartCareControl control_panel.py
"""
import os
import sys

# When frozen by PyInstaller (conda-based Tk), point Tcl/Tk at the bundled data
# BEFORE importing tkinter, or _tkinter fails to load its DLLs.
if getattr(sys, "frozen", False):
    _base = getattr(sys, "_MEIPASS", "")
    for _var, _sub in (("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")):
        _p = os.path.join(_base, _sub)
        if os.path.isdir(_p):
            os.environ[_var] = _p

import queue
import shlex
import threading
import time
import webbrowser
import tkinter as tk
from tkinter import ttk

try:
    import paramiko
except ImportError:
    raise SystemExit("Missing dependency: pip install paramiko")

# ── Pi-side layout (matches the repo on the Pi) ──────────────────────────────
REPO = "pyskl-research"
NODE = "demo_combined/edge_emergency/edge_node.py"
VENV_PY = ".venv-edge/bin/python"
PATTERN = "[e]dge_node.py"          # bracket = pkill/pgrep won't match itself
DEFAULTS = dict(host="smartcare.local", user="batyr",
                key="~/.ssh/pi_smartcare", port=8443)


# ════════════════════════════════════════════════════════════════════════════
#  SSH backend (no GUI — independently testable)
# ════════════════════════════════════════════════════════════════════════════
class PiController:
    def __init__(self):
        self.client = None
        self.host = self.user = self.password = None

    @property
    def connected(self):
        t = self.client.get_transport() if self.client else None
        return bool(t and t.is_active())

    def connect(self, host, user, key_path=None, password=None):
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kw = dict(hostname=host, username=user, timeout=8, auth_timeout=8, banner_timeout=8)
        kp = os.path.expanduser(key_path) if key_path else None
        if kp and os.path.exists(kp):
            kw["key_filename"] = kp
        if password:
            kw["password"] = password
            kw["look_for_keys"] = not (kp and os.path.exists(kp))
        c.connect(**kw)
        self.client, self.host, self.user, self.password = c, host, user, password

    def close(self):
        if self.client:
            self.client.close()
            self.client = None

    def run(self, cmd, timeout=25):
        if not self.connected:
            raise RuntimeError("not connected")
        _i, o, e = self.client.exec_command(cmd, timeout=timeout)
        return o.read().decode("utf-8", "replace").strip(), \
            e.read().decode("utf-8", "replace").strip()

    def fire(self, cmd):
        """Launch a detached/background command without waiting for EOF."""
        if not self.connected:
            raise RuntimeError("not connected")
        ch = self.client.get_transport().open_session()
        ch.settimeout(8)
        ch.exec_command(cmd)
        time.sleep(0.4)
        try:
            ch.close()
        except Exception:
            pass

    def sudo(self, cmd):
        pw = self.password or ""
        return self.run(f"echo {shlex.quote(pw)} | sudo -S bash -c {shlex.quote(cmd)} 2>&1")

    def is_running(self):
        out, _ = self.run(f"pgrep -f '{PATTERN}' >/dev/null && echo YES || echo NO")
        return out.strip().endswith("YES")

    def temperature(self):
        out, _ = self.run("vcgencmd measure_temp 2>/dev/null || echo n/a")
        return out.replace("temp=", "").strip()

    def status(self):
        return {"running": self.is_running(), "temp": self.temperature(),
                "ip": self.run("hostname -I | awk '{print $1}'")[0]}

    def start(self, p):
        if self.is_running():
            return "already running"
        src = (f"--camera {int(p['camera'])}" if p["source"] == "camera"
               else f"--source {shlex.quote(p['url'])}")
        args = (f"{src} --pose-backend {p['pose']} --num-person {int(p['persons'])} "
                f"--short-side {int(p['short'])} --recog-every {int(p['recog'])} "
                f"--max-fps {p['maxfps']} --port {DEFAULTS['port']} --https --http-port 8000")
        cmd = (f"cd ~/{REPO} && PYTHONIOENCODING=utf-8 nohup setsid {VENV_PY} -u {NODE} "
               f"{args} > /tmp/edge.log 2>&1 < /dev/null & disown")
        self.fire(cmd)
        time.sleep(2.5)
        return "started" if self.is_running() else "start issued (check status)"

    def stop(self):
        self.run(f"pkill -f '{PATTERN}'")
        time.sleep(1)
        return "stopped" if not self.is_running() else "still running?"

    def add_wifi(self, ssid, psk, priority=15):
        name = "panel-" + ("".join(ch for ch in ssid if ch.isalnum())[:24] or "wifi")
        cmd = (f"nmcli connection add type wifi ifname wlan0 con-name {shlex.quote(name)} "
               f"autoconnect yes connection.autoconnect-priority {priority} "
               f"ssid {shlex.quote(ssid)} wifi-sec.key-mgmt wpa-psk "
               f"wifi-sec.psk {shlex.quote(psk)}")
        out, _ = self.sudo(cmd)
        return out or f"added '{ssid}' (joins when in range)"


# ════════════════════════════════════════════════════════════════════════════
#  GUI
# ════════════════════════════════════════════════════════════════════════════
BG, CARD, INSET, LINE, TXT, MUT, ACC = (
    "#0d1320", "#141d2e", "#0b111d", "#243248", "#e7edf5", "#8aa0bd", "#2dd4bf")


class App:
    def __init__(self, root):
        self.root = root
        self.pi = PiController()
        self.q = queue.Queue()
        root.title("Smart-Care · Control Panel")
        root.configure(bg=BG)
        root.geometry("460x600")
        root.minsize(380, 420)
        self._style()
        self._build()
        self.root.after(150, self._drain)
        self._set_conn(False)

    # ── styling ──────────────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", background=BG, foreground=TXT, fieldbackground=INSET, bordercolor=LINE)
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=CARD, foreground=TXT, font=("Segoe UI", 9))
        s.configure("BG.TLabel", background=BG, foreground=TXT, font=("Segoe UI", 9))
        s.configure("Hdr.TLabel", background=BG, foreground=TXT, font=("Segoe UI", 13, "bold"))
        s.configure("Mut.TLabel", background=CARD, foreground=MUT, font=("Segoe UI", 8))
        s.configure("CardTitle.TLabel", background=CARD, foreground=MUT, font=("Segoe UI", 8, "bold"))
        s.configure("TButton", background=INSET, foreground=TXT, font=("Segoe UI", 9), padding=5, borderwidth=1)
        s.map("TButton", background=[("active", LINE)])
        s.configure("Accent.TButton", background=ACC, foreground="#062a26", font=("Segoe UI", 9, "bold"))
        s.map("Accent.TButton", background=[("active", "#5fe6d6")])
        s.configure("Link.TButton", background=CARD, foreground=MUT, borderwidth=0, font=("Segoe UI", 8))
        s.map("Link.TButton", background=[("active", CARD)], foreground=[("active", ACC)])
        s.configure("TEntry", fieldbackground=INSET, foreground=TXT, insertcolor=TXT)
        s.configure("TRadiobutton", background=CARD, foreground=TXT)
        s.configure("TCombobox", fieldbackground=INSET, foreground=TXT)
        s.configure("Vertical.TScrollbar", background=CARD, troughcolor=BG, bordercolor=BG, arrowcolor=MUT)

    # ── small layout helpers ──────────────────────────────────────────────────
    def _card(self, title):
        outer = tk.Frame(self.body, bg=LINE)            # 1px border
        outer.pack(fill="x", padx=10, pady=(0, 9))
        f = tk.Frame(outer, bg=CARD)
        f.pack(fill="x", padx=1, pady=1)
        inner = ttk.Frame(f, style="Card.TFrame", padding=11)
        inner.pack(fill="x")
        ttk.Label(inner, text=title.upper(), style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))
        return inner

    def _field(self, parent, label, init="", show=None, width=None):
        row = ttk.Frame(parent, style="Card.TFrame"); row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=16).pack(side="left")
        e = ttk.Entry(row, show=show, width=width)
        e.pack(side="left", fill="x", expand=True)
        if init:
            e.insert(0, init)
        return e

    def _collapsible(self, parent, title):
        st = {"open": False}
        btn = ttk.Button(parent, text="▸ " + title, style="Link.TButton")
        btn.pack(anchor="w", pady=(4, 0))
        body = ttk.Frame(parent, style="Card.TFrame")

        def toggle():
            st["open"] = not st["open"]
            if st["open"]:
                body.pack(fill="x"); btn.config(text="▾ " + title)
            else:
                body.forget(); btn.config(text="▸ " + title)
        btn.config(command=toggle)
        return body

    # ── build ──────────────────────────────────────────────────────────────────
    def _build(self):
        # fixed header
        top = ttk.Frame(self.root, style="TFrame"); top.pack(fill="x")
        ttk.Label(top, text="Smart-Care · Control Panel", style="Hdr.TLabel").pack(anchor="w", padx=12, pady=(11, 1))
        self.connlbl = ttk.Label(top, text="● disconnected", style="Mut.TLabel", background=BG)
        self.connlbl.pack(anchor="w", padx=12, pady=(0, 8))

        # scrollable body
        cont = ttk.Frame(self.root, style="TFrame"); cont.pack(fill="both", expand=True)
        canvas = tk.Canvas(cont, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(cont, orient="vertical", command=canvas.yview)
        self.body = ttk.Frame(canvas, style="TFrame")
        win = canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        # — Connection — (host + buttons up front; the rest tucked away)
        c = self._card("Connection")
        self.e_host = self._field(c, "Host / IP", DEFAULTS["host"])
        bf = ttk.Frame(c, style="Card.TFrame"); bf.pack(fill="x", pady=(6, 0))
        ttk.Button(bf, text="Connect", style="Accent.TButton", command=self.on_connect).pack(side="left")
        ttk.Button(bf, text="Find Pi", command=self.on_find).pack(side="left", padx=5)
        ttk.Button(bf, text="Disconnect", command=self.on_disconnect).pack(side="left")
        det = self._collapsible(c, "login details")
        self.e_user = self._field(det, "User", DEFAULTS["user"])
        self.e_key = self._field(det, "Key file", DEFAULTS["key"])
        self.e_pw = self._field(det, "Password", show="•")

        # — Demo server —
        sv = self._card("Demo server")
        self.srvlbl = ttk.Label(sv, text="status: —      temp: —", style="TLabel")
        self.srvlbl.pack(anchor="w", pady=(0, 7))
        g = ttk.Frame(sv, style="Card.TFrame"); g.pack(fill="x")
        ttk.Button(g, text="▶ Start", style="Accent.TButton", command=self.on_start).pack(side="left")
        ttk.Button(g, text="■ Stop", command=self.on_stop).pack(side="left", padx=5)
        ttk.Button(g, text="↻ Restart", command=self.on_restart).pack(side="left")
        ttk.Button(g, text="⟳", width=3, command=self.on_status).pack(side="left", padx=5)
        ttk.Button(sv, text="🖥  Open dashboard", command=self.on_dash).pack(fill="x", pady=(7, 0))

        # — Boot parameters — (just the common ones; rest under Advanced)
        pf = self._card("Boot parameters")
        self.v_source = tk.StringVar(value="camera")
        sr = ttk.Frame(pf, style="Card.TFrame"); sr.pack(fill="x", pady=2)
        ttk.Label(sr, text="Source", width=16).pack(side="left")
        ttk.Radiobutton(sr, text="Camera", variable=self.v_source, value="camera").pack(side="left")
        ttk.Radiobutton(sr, text="Phone URL", variable=self.v_source, value="url").pack(side="left", padx=(8, 0))
        self.e_maxfps = self._field(pf, "Max FPS (0=uncap)", "0")
        adv = self._collapsible(pf, "advanced")
        self.e_cam = self._field(adv, "Camera index", "4")
        self.e_url = self._field(adv, "Stream URL", "http://192.168.1.72:8080/video")
        self.v_pose = tk.StringVar(value="movenet")
        pr = ttk.Frame(adv, style="Card.TFrame"); pr.pack(fill="x", pady=2)
        ttk.Label(pr, text="Pose backend", width=16).pack(side="left")
        ttk.Combobox(pr, textvariable=self.v_pose, values=["movenet", "rtmpose"],
                     state="readonly", width=12).pack(side="left")
        self.e_short = self._field(adv, "Short side", "320")
        self.e_recog = self._field(adv, "Recog every N", "4")
        self.e_persons = self._field(adv, "Person slots", "2")

        # — Wi-Fi —
        wf = self._card("Configure Pi Wi-Fi")
        self.e_ssid = self._field(wf, "Network (SSID)")
        self.e_wpw = self._field(wf, "Wi-Fi password", show="•")
        ttk.Button(wf, text="Add Wi-Fi to Pi", command=self.on_wifi).pack(fill="x", pady=(7, 0))

        # — Log —
        lf = self._card("Log")
        self.log = tk.Text(lf, height=6, bg=INSET, fg=MUT, insertbackground=TXT,
                           relief="flat", font=("Consolas", 8), wrap="word")
        self.log.pack(fill="both", expand=True)

    # ── async plumbing ────────────────────────────────────────────────────────
    def _logmsg(self, msg):
        self.q.put(("log", msg))

    def _async(self, label, fn):
        self._logmsg(f"… {label}")

        def worker():
            try:
                res = fn()
                self.q.put(("log", f"✓ {label}: {res}" if res else f"✓ {label}"))
            except Exception as e:
                self.q.put(("log", f"✗ {label}: {e}"))
            self.q.put(("status", None))
        threading.Thread(target=worker, daemon=True).start()

    def _drain(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", time.strftime("%H:%M:%S ") + str(payload) + "\n")
                    self.log.see("end")
                elif kind == "status":
                    self._refresh_status_async()
                elif kind == "conn":
                    self._set_conn(payload)
                elif kind == "srv":
                    self.srvlbl.config(text=payload)
        except queue.Empty:
            pass
        self.root.after(150, self._drain)

    def _set_conn(self, ok):
        self.connlbl.config(text=("● connected to " + (self.pi.host or "")) if ok else "● disconnected",
                            foreground=(ACC if ok else MUT))

    def _refresh_status_async(self):
        if not self.pi.connected:
            return

        def worker():
            try:
                st = self.pi.status()
                self.q.put(("srv", f"status: {'RUNNING' if st['running'] else 'stopped'}"
                                   f"      temp: {st['temp']}      {st['ip']}"))
            except Exception as e:
                self.q.put(("log", f"status error: {e}"))
        threading.Thread(target=worker, daemon=True).start()

    # ── button handlers ───────────────────────────────────────────────────────
    def _params(self):
        return dict(source=self.v_source.get(), camera=self.e_cam.get() or "4",
                    url=self.e_url.get(), pose=self.v_pose.get(),
                    persons=self.e_persons.get() or "2", short=self.e_short.get() or "320",
                    recog=self.e_recog.get() or "4", maxfps=self.e_maxfps.get() or "0")

    def on_connect(self):
        host, user = self.e_host.get().strip(), self.e_user.get().strip()
        key, pw = self.e_key.get().strip(), self.e_pw.get()

        def fn():
            self.pi.connect(host, user, key or None, pw or None)
            self.q.put(("conn", True))
            return "ok"
        self._async(f"connect {user}@{host}", fn)

    def on_find(self):
        def fn():
            for h in (self.e_host.get().strip(), "smartcare.local", "192.168.1.81"):
                if not h:
                    continue
                try:
                    self.pi.connect(h, self.e_user.get().strip(),
                                    self.e_key.get().strip() or None, self.e_pw.get() or None)
                    self.e_host.delete(0, "end"); self.e_host.insert(0, h)
                    self.q.put(("conn", True))
                    return f"found at {h}"
                except Exception:
                    continue
            raise RuntimeError("Pi not reachable (powered on? same network?)")
        self._async("find Pi", fn)

    def on_disconnect(self):
        self.pi.close(); self._set_conn(False); self._logmsg("disconnected")

    def on_start(self):
        p = self._params(); self._async("start server", lambda: self.pi.start(p))

    def on_stop(self):
        self._async("stop server", self.pi.stop)

    def on_restart(self):
        p = self._params()

        def fn():
            self.pi.stop(); time.sleep(2); return self.pi.start(p)
        self._async("restart server", fn)

    def on_status(self):
        self._refresh_status_async(); self._logmsg("refreshed status")

    def on_dash(self):
        host = self.pi.host or self.e_host.get().strip()
        webbrowser.open(f"https://{host}:{DEFAULTS['port']}/")

    def on_wifi(self):
        ssid, pw = self.e_ssid.get().strip(), self.e_wpw.get()
        if not ssid:
            self._logmsg("enter an SSID first"); return
        self._async(f"add Wi-Fi '{ssid}'", lambda: self.pi.add_wifi(ssid, pw))


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
