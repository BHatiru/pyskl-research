#!/usr/bin/env python
"""Mock Apple Watch vitals sender.

POSTs simulated HR / SpO2 / temperature + anomaly to the Pi's /vitals endpoint so
the combined dashboard can be rehearsed BEFORE the iOS app's HTTP bridge lands.
Mirrors the watch app's DataSimulator modes (eisenchamp/smart_house_app).

Usage:
    # cycle normal -> anomalies -> normal (good for a live demo):
    python mock_vitals.py --url https://192.168.1.81:8443/vitals

    # hold one condition:
    python mock_vitals.py --url https://192.168.1.81:8443/vitals --mode tachycardia

    # plain-HTTP node:
    python mock_vitals.py --url http://192.168.1.81:8000/vitals
"""
import argparse
import json
import random
import ssl
import time
import urllib.request

# mode -> (base HR, base SpO2, base temp, anomaly label, isAnomaly)
MODES = {
    "normal":      (72, 98, 36.6, "Normal", False),
    "tachycardia": (130, 97, 37.0, "Tachycardia", True),
    "bradycardia": (42, 96, 36.4, "Bradycardia", True),
    "lowspo2":     (88, 88, 36.5, "Low SpO₂", True),
    "fever":       (100, 96, 39.2, "Temperature Anomaly", True),
    "multi":       (140, 87, 39.5, "Multi-parameter Anomaly", True),
}
# A demo-friendly storyline: mostly normal, occasional anomalies.
CYCLE = ["normal"] * 6 + ["tachycardia"] * 3 + ["normal"] * 6 + ["lowspo2"] * 3 \
        + ["normal"] * 6 + ["multi"] * 3


def _jit(base, spread):
    return round(base + random.uniform(-spread, spread), 1)


def reading(mode):
    hr, sp, tp, label, anom = MODES[mode]
    return {
        "heartRate": _jit(hr, 4),
        "spo2": _jit(sp, 1),
        "temperature": _jit(tp, 0.2),
        "anomaly": label,
        "isAnomaly": anom,
        "emergency": mode == "multi",
        "ts": time.time(),
    }


def main():
    p = argparse.ArgumentParser(description="Mock Apple Watch vitals -> Pi /vitals")
    p.add_argument("--url", default="https://192.168.1.81:8443/vitals")
    p.add_argument("--mode", choices=list(MODES) + ["cycle"], default="cycle")
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # accept the Pi's self-signed cert
    use_ctx = ctx if args.url.lower().startswith("https") else None

    print(f"POSTing mock vitals -> {args.url}  (mode={args.mode}, every {args.interval}s)")
    print("Ctrl-C to stop.\n")
    i = 0
    while True:
        mode = CYCLE[i % len(CYCLE)] if args.mode == "cycle" else args.mode
        v = reading(mode)
        body = json.dumps(v).encode()
        req = urllib.request.Request(
            args.url, data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3, context=use_ctx) as r:
                ok = (r.status == 200)
        except Exception as e:
            ok = False
            print(f"  POST failed: {e}")
        flag = "!" if v["isAnomaly"] else " "
        print(f" {flag}[{mode:12}] HR {v['heartRate']:>5}  SpO2 {v['spo2']:>4}  "
              f"T {v['temperature']:>4}  {v['anomaly']:<24} {'ok' if ok else 'ERR'}")
        i += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
