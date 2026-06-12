# Apple Watch → Smart-Care Pi — integration handoff

The Pi accepts vitals from the iPhone companion app and shows them live on the
dashboard (the **Vitals · Apple Watch** panel). The watch app
(`eisenchamp/smart_house_app`) is on-device only, so the **only iOS change needed
is a ~15-line HTTP POST** — no Pi changes.

> **If you were having trouble connecting — read the "Why it fails" section.**
> Use the **plain-HTTP** endpoint (port 8000), grant the **Local Network**
> permission, and make sure the phone is on the **same Wi-Fi** as the Pi.

---

## 1. The endpoint — use plain HTTP, port 8000

```
POST http://<PI-IP>:8000/vitals
Content-Type: application/json
```

**Verify it works first** (from any laptop on the same Wi-Fi — no app needed):

```bash
curl -X POST http://<PI-IP>:8000/vitals \
  -H "Content-Type: application/json" \
  -d '{"heartRate":72,"spo2":98,"temperature":36.6,"anomaly":"Normal","isAnomaly":false}'
# -> {"ok":true}      and the dashboard's Vitals panel flips to LIVE
```

If that curl works but the app doesn't, the problem is **on the iOS side** (§3),
not the Pi.

> **Why port 8000 and not 8443?** 8443 is HTTPS with a **self-signed** cert, which
> iOS `URLSession` rejects unless you add a trust delegate. Port **8000 is plain
> HTTP** — far simpler for a LAN device. (Both ports accept `/vitals`.)

## 2. Finding `<PI-IP>` (it changes per network!)

The Pi's IP depends on the Wi-Fi it's on. To find the current one:
- It's printed when the node starts (`Dashboard: https://<ip>:8443/`), **or**
- shown in the **Control Panel** status line, **or**
- try the mDNS name **`smartcare.local`** (works from iOS too): `http://smartcare.local:8000/vitals`.

👉 **Best practice: make the Pi host a settings field in the app**, default
`smartcare.local`, so you don't recompile when the network changes. *(Right now
it's `192.168.0.212`, but assume it will change.)*

## 3. Why it fails on iOS (the 3 usual culprits)

1. **Local Network permission (iOS 14+).** An app **cannot** talk to a LAN device
   until the user grants "Local Network" access — and it **fails silently** otherwise.
   - Add to **Info.plist**: `NSLocalNetworkUsageDescription` = *"Connect to the Smart-Care monitor on your network."*
   - The permission prompt appears on the **first** local connection attempt — the user must tap **Allow**. (Settings → the app → Local Network to re-enable.)
2. **App Transport Security blocks plain HTTP.** Add to **Info.plist**:
   ```xml
   <key>NSAppTransportSecurity</key>
   <dict><key>NSAllowsLocalNetworking</key><true/></dict>
   ```
3. **Phone not on the same Wi-Fi as the Pi.** Cellular/other SSID → no route.

## 4. The JSON contract

```json
{
  "heartRate": 72,
  "spo2": 98,
  "temperature": 36.6,
  "anomaly": "Normal",          // AnomalyResult.label
  "isAnomaly": false,
  "emergency": false,
  "ts": 1733940000.0            // optional unix seconds
}
```
Post on each new reading (~1 Hz) and again on emergency. Dashboard shows `LIVE`
while readings are fresh (<20 s), else `PAIRING`; abnormal renders amber.

## 5. Swift — add to `Smart_home/WatchSessionManager.swift`

```swift
import Foundation

enum Hub {
    // 🔧 Set to your Pi. Plain HTTP + port 8000.  smartcare.local works on iOS.
    static let url = URL(string: "http://smartcare.local:8000/vitals")!

    static func post(hr: Double, spo2: Double, temp: Double,
                     anomaly: String, isAnomaly: Bool, emergency: Bool) {
        let payload: [String: Any] = [
            "heartRate": hr, "spo2": spo2, "temperature": temp,
            "anomaly": anomaly, "isAnomaly": isAnomaly, "emergency": emergency,
            "ts": Date().timeIntervalSince1970,
        ]
        guard let body = try? JSONSerialization.data(withJSONObject: payload) else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        req.timeoutInterval = 3
        URLSession.shared.dataTask(with: req) { _, resp, err in
            if let err = err { print("vitals POST failed:", err) }            // watch the console
            else if let r = resp as? HTTPURLResponse { print("vitals POST:", r.statusCode) }
        }.resume()
    }
}
```

Call it where readings/emergencies arrive — your `WatchSessionManager` already
has `healthData` + `predictionLabel`:

```swift
// in session(_:didReceiveMessage:) after updating healthData:
Hub.post(hr: healthData.heartRate, spo2: healthData.spo2, temp: healthData.temperature,
         anomaly: predictionLabel, isAnomaly: false, emergency: false)

// in handleEmergency(...):
Hub.post(hr: healthData.heartRate, spo2: healthData.spo2, temp: healthData.temperature,
         anomaly: predictionLabel, isAnomaly: true, emergency: true)
```

## 6. Troubleshooting checklist

| symptom | cause / fix |
|---|---|
| `curl` to `:8000/vitals` fails too | Pi node not running, or wrong IP. Start it; re-check the IP. |
| `curl` works, app doesn't | iOS side: Local Network permission **not granted**, or ATS, or wrong host. |
| First POST hangs ~30 s then fails | Local Network prompt was dismissed/denied → enable in iOS Settings. |
| Works once, then stops | phone left the Wi-Fi, or Pi IP changed → use `smartcare.local`. |
| Console prints a status code (200) but no LIVE | check the JSON keys match §4. |

## 7. Demo without a real Apple Watch

- Their app has `DataSimulator.swift` (normal / tachy / brady / lowSpO₂ / fever /
  multi) — drive `Hub.post(...)` from a simulated reading on a timer.
- Or run our Python mock (no app at all):
  `python demo_combined/edge_emergency/mock_vitals.py --url http://<PI-IP>:8000/vitals`

## 8. (Optional) HTTPS variant

If you prefer 8443/HTTPS, it works too — but add a `URLSessionDelegate` that
trusts the self-signed cert (`URLCredential(trust:)`), and post to
`https://<PI-IP>:8443/vitals`. Plain HTTP (§1) is recommended for the demo.

## Next: fusion (optional)

Today the Pi **displays** vitals. Next step: fuse them — a fall (skeleton)
**co-occurring** with abnormal vitals (low SpO₂ / tachycardia) → higher-confidence
emergency than either alone.
