# Apple Watch ↔ Smart-Care Pi — integration spec

The Pi edge node now accepts vitals from the iPhone companion app and shows them
live on the dashboard (the **Vitals · Apple Watch** panel). The watch app
(`eisenchamp/smart_house_app`) is on-device only, so the **only change needed on
the iOS side is a ~15-line HTTP POST** — no Pi changes required.

```
Apple Watch (HealthKit / DataSimulator)
        │  WCSession.sendMessage   (already exists)
        ▼
iPhone  WatchSessionManager + AnomalyDetector (CoreML)
        │  NEW: URLSession POST /vitals   ◄── add this
        ▼
Raspberry Pi  edge_node.py  ──►  dashboard "Vitals · Apple Watch" panel
                                  (HR / SpO₂ / Temp / anomaly status)
```

## The contract — `POST https://<pi-ip>:8443/vitals`

JSON body (all fields optional; send what you have):

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

Response: `{"ok":true}`. Post on every new reading (e.g. ~1 Hz) and again on an
emergency. The dashboard shows `LIVE` while readings are fresh (<20 s), else
`PAIRING`. An abnormal status (`isAnomaly`/`emergency`) renders amber.

> The Pi serves HTTPS with a **self-signed** cert (so the phone dashboard can do
> notifications). iOS `URLSession` rejects self-signed certs unless you trust them
> — the delegate below handles that. If you'd rather use plain HTTP, run the Pi
> node without `--https` (port 8000) and add an ATS exception instead.

## Add to `Smart_home/WatchSessionManager.swift`

```swift
import Foundation

// Posts vitals to the Pi hub; trusts the Pi's self-signed HTTPS cert.
final class HubPoster: NSObject, URLSessionDelegate {
    static let shared = HubPoster()

    // 🔧 Set your Pi's LAN IP (printed when edge_node starts). 8443 = HTTPS.
    static let hubURL = URL(string: "https://192.168.1.81:8443/vitals")!

    private lazy var session = URLSession(configuration: .default,
                                          delegate: self, delegateQueue: nil)

    func post(_ payload: [String: Any]) {
        guard let body = try? JSONSerialization.data(withJSONObject: payload) else { return }
        var req = URLRequest(url: Self.hubURL)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        req.timeoutInterval = 3
        session.dataTask(with: req).resume()
    }

    // Trust the Pi's self-signed certificate (LAN-local only).
    func urlSession(_ s: URLSession, didReceive ch: URLAuthenticationChallenge,
                    completionHandler done: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        if let trust = ch.protectionSpace.serverTrust {
            done(.useCredential, URLCredential(trust: trust))
        } else { done(.performDefaultHandling, nil) }
    }
}
```

Then call it from `WatchSessionManager` (it already has `healthData` +
`predictionLabel`). Add a helper and call it where readings/emergencies arrive:

```swift
func pushVitals(isAnomaly: Bool = false, emergency: Bool = false) {
    HubPoster.shared.post([
        "heartRate":   healthData.heartRate,
        "spo2":        healthData.spo2,
        "temperature": healthData.temperature,
        "anomaly":     predictionLabel,
        "isAnomaly":   isAnomaly,
        "emergency":   emergency,
        "ts":          Date().timeIntervalSince1970,
    ])
}
```

- In `session(_:didReceiveMessage:)`, after updating `healthData`, call `self.pushVitals()`.
- In `handleEmergency(...)`, call `self.pushVitals(isAnomaly: true, emergency: true)`.

## Demo without a real Apple Watch

The app already has `DataSimulator.swift` (modes: normal / tachycardia /
bradycardia / lowSpO₂ / fever / multi-anomaly). Drive `pushVitals()` from a
simulated reading on a timer — the dashboard shows the chosen condition live,
alongside the Pi's skeleton fall detection. One screen, two subsystems.

## Next (optional) — fusion

Today the Pi just **displays** vitals. A later step: fuse them — e.g. a fall
(skeleton) **co-occurring** with abnormal vitals (low SpO₂ / tachycardia) raises
a higher-confidence emergency than either alone.
