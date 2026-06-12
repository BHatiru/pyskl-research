#!/usr/bin/env python
"""Smart-Care Control Panel — a laptop GUI to drive the Raspberry Pi edge node
over SSH without typing commands.

Connect to the Pi, start/stop the demo, change boot parameters, configure the
Pi's Wi-Fi, and open the dashboard — all with buttons. Wraps the same SSH
commands we use by hand.

Run:   python control_panel.py
Deps:  paramiko  (pip install paramiko).  GUI = Tkinter (stdlib).
Package to a standalone Windows .exe later with:
       pyinstaller --onefile --noconsole --name SmartCareControl control_panel.py
"""
import os
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
        self.host = None
        self.user = None
        self.password = None

    @property
    def connected(self):
        return self.client is not None and self.client.get_transport() is not None \
            and self.client.get_transport().is_active()

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
        """Launch a detached/background command without waiting for EOF
        (a nohup/setsid/disown launch never closes the channel otherwise)."""
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

    # ── status / control ────────────────────────────────────────────────────
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
        if p["source"] == "camera":
            src = f"--camera {int(p['camera'])}"
        else:
            src = f"--source {shlex.quote(p['url'])}"
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
        name = "panel-" + "".join(ch for ch in ssid if ch.isalnum())[:24] or "panel-wifi"
        cmd = (f"nmcli connection add type wifi ifname wlan0 con-name {shlex.quote(name)} "
               f"autoconnect yes connection.autoconnect-priority {priority} "
               f"ssid {shlex.quote(ssid)} wifi-sec.key-mgmt wpa-psk "
               f"wifi-sec.psk {shlex.quote(psk)}")
        out, _ = self.sudo(cmd)
        return out or f"added '{ssid}' (will join when in range)"

    def tail_log(self, n=12):
        out, _ = self.run(f"tail -n {n} /tmp/edge.log 2>/dev/null "
                          f"| grep -vE 'GetGpuDevices|device_discovery'")
        return out


# ════════════════════════════════════════════════════════════════════════════
#  GUI
# ════════════════════════════════════════════════════════════════════════════
BG, CARD, LINE, TXT, MUT, ACC = "#0d1320", "#141d2e", "#243248", "#e7edf5", "#8aa0bd", "#2dd4bf"


class App:
    def __init__(self, root):
        self.root = root
        self.pi = PiController()
        self.q = queue.Queue()
        root.title("Smart-Care · Control Panel")
        root.configure(bg=BG)
        root.geometry("560x760")
        self._style()
        self._build()
        self.root.after(120, self._drain)
        self._set_conn(False)

    # ── styling ──────────────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", background=BG, foreground=TXT, fieldbackground=CARD, bordercolor=LINE)
        s.configure("TFrame", background=BG)
        s.configure("Card.TLabelframe", background=CARD, bordercolor=LINE, relief="solid", borderwidth=1)
        s.configure("Card.TLabelframe.Label", background=CARD, foreground=MUT, font=("Segoe UI", 9, "bold"))
        s.configure("TLabel", background=CARD, foreground=TXT, font=("Segoe UI", 9))
        s.configure("Hdr.TLabel", background=BG, foreground=TXT, font=("Segoe UI", 14, "bold"))
        s.configure("Mut.TLabel", background=CARD, foreground=MUT, font=("Segoe UI", 8))
        s.configure("TButton", background=CARD, foreground=TXT, font=("Segoe UI", 9), padding=6, borderwidth=1)
        s.map("TButton", background=[("active", LINE)])
        s.configure("Accent.TButton", background=ACC, foreground="#062a26", font=("Segoe UI", 9, "bold"))
        s.map("Accent.TButton", background=[("active", "#5fe6d6")])
        s.configure("TEntry", fieldbackground="#0b111d", foreground=TXT, insertcolor=TXT)
        s.configure("TRadiobutton", background=CARD, foreground=TXT)
        s.configure("TCombobox", fieldbackground="#0b111d", foreground=TXT)

    def _card(self, parent, title):
        f = ttk.Labelframe(parent, text=" " + title + " ", style="Card.TLabelframe", padding=10)
        f.pack(fill="x", padx=12, pady=(0, 10))
        return f

    def _row(self, parent, label, widget, r):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=3)
        widget.grid(row=r, column=1, sticky="ew", pady=3)
        parent.columnconfigure(1, weight=1)

    def _build(self):
        ttk.Label(self.root, text="Smart-Care · Control Panel", style="Hdr.TLabel").pack(
            anchor="w", padx=12, pady=(12, 2))
        self.connlbl = ttk.Label(self.root, text="● disconnected", style="Mut.TLabel", background=BG)
        self.connlbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Connection
        c = self._card(self.root, "Connection")
        self.e_host = ttk.Entry(c); self.e_host.insert(0, DEFAULTS["host"])
        self.e_user = ttk.Entry(c); self.e_user.insert(0, DEFAULTS["user"])
        self.e_key = ttk.Entry(c); self.e_key.insert(0, DEFAULTS["key"])
        self.e_pw = ttk.Entry(c, show="•")
        self._row(c, "Host / IP", self.e_host, 0)
        self._row(c, "User", self.e_user, 1)
        self._row(c, "Key file", self.e_key, 2)
        self._row(c, "Password (for Wi-Fi/sudo)", self.e_pw, 3)
        bf = ttk.Frame(c, style="TFrame"); bf.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        bf.configure(style="TFrame")
        ttk.Button(bf, text="Connect", style="Accent.TButton", command=self.on_connect).pack(side="left")
        ttk.Button(bf, text="Find Pi", command=self.on_find).pack(side="left", padx=6)
        ttk.Button(bf, text="Disconnect", command=self.on_disconnect).pack(side="left")

        # Server control
        sv = self._card(self.root, "Demo server")
        self.srvlbl = ttk.Label(sv, text="status: —    temp: —", style="TLabel")
        self.srvlbl.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        ttk.Button(sv, text="▶ Start", style="Accent.TButton", command=self.on_start).grid(row=1, column=0, padx=(0, 6))
        ttk.Button(sv, text="■ Stop", command=self.on_stop).grid(row=1, column=1, padx=6)
        ttk.Button(sv, text="↻ Restart", command=self.on_restart).grid(row=1, column=2, padx=6)
        ttk.Button(sv, text="Refresh", command=self.on_status).grid(row=1, column=3, padx=6)
        ttk.Button(sv, text="🖥 Open dashboard", command=self.on_dash).grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        # Parameters
        pf = self._card(self.root, "Boot parameters")
        self.v_source = tk.StringVar(value="camera")
        ttk.Radiobutton(pf, text="RealSense / camera", variable=self.v_source, value="camera").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(pf, text="Phone / stream URL", variable=self.v_source, value="url").grid(row=0, column=1, sticky="w")
        self.e_cam = ttk.Entry(pf, width=8); self.e_cam.insert(0, "4")
        self.e_url = ttk.Entry(pf); self.e_url.insert(0, "http://192.168.1.72:8080/video")
        self._row(pf, "Camera index", self.e_cam, 1)
        self._row(pf, "Stream URL", self.e_url, 2)
        self.v_pose = tk.StringVar(value="movenet")
        cb = ttk.Combobox(pf, textvariable=self.v_pose, values=["movenet", "rtmpose"], state="readonly", width=10)
        self._row(pf, "Pose backend", cb, 3)
        self.e_short = ttk.Entry(pf, width=8); self.e_short.insert(0, "320")
        self.e_recog = ttk.Entry(pf, width=8); self.e_recog.insert(0, "4")
        self.e_maxfps = ttk.Entry(pf, width=8); self.e_maxfps.insert(0, "0")
        self.e_persons = ttk.Entry(pf, width=8); self.e_persons.insert(0, "2")
        self._row(pf, "Short side", self.e_short, 4)
        self._row(pf, "Recognise every N frames", self.e_recog, 5)
        self._row(pf, "Max FPS (0 = uncapped)", self.e_maxfps, 6)
        self._row(pf, "Person slots", self.e_persons, 7)
        ttk.Label(pf, text="Changes apply on next Start / Restart.", style="Mut.TLabel").grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # Wi-Fi
        wf = self._card(self.root, "Configure Pi Wi-Fi")
        self.e_ssid = ttk.Entry(wf)
        self.e_wpw = ttk.Entry(wf, show="•")
        self._row(wf, "Network name (SSID)", self.e_ssid, 0)
        self._row(wf, "Wi-Fi password", self.e_wpw, 1)
        ttk.Button(wf, text="Add Wi-Fi to Pi", command=self.on_wifi).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(wf, text="Pi joins it automatically when that network is in range.", style="Mut.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # Log
        lf = self._card(self.root, "Log")
        self.log = tk.Text(lf, height=8, bg="#0b111d", fg=MUT, insertbackground=TXT,
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
                txt = f"status: {'RUNNING' if st['running'] else 'stopped'}    temp: {st['temp']}    ip: {st['ip']}"
                self.q.put(("srv", txt))
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
