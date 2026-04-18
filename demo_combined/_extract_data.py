"""Extract patient scenarios for the HTML dashboard."""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, json, glob

files = sorted(glob.glob("ntu_action_dataset/integrated_v2/*.parquet"))
iv2 = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

scenarios = []
combos = [
    ("falling_down", "Bradycardia", "CRITICAL"),
    ("staggering", "Tachycardia", "CRITICAL"),
    ("chest_pain", "High pressure", "HIGH"),
    ("headache", "Low pressure", "HIGH"),
    ("nausea_vomiting", "Nitroglycerine", "MEDIUM"),
    ("back_pain", "Normal", "MEDIUM"),
    ("walking_towards", "Normal", "LOW"),
    ("sitting_down", "Normal", "LOW"),
]

for action, vitals_class, level in combos:
    sub = iv2[(iv2.ntu_action_name == action) & (iv2.class_name == vitals_class)]
    if len(sub) == 0:
        continue
    seq_id = sub.groupby("sequence_id").HR.apply(lambda x: x.notna().sum()).idxmax()
    seq = sub[sub.sequence_id == seq_id].sort_values("timestamp_s")
    step = max(1, len(seq) // 30)
    ts = seq.iloc[::step]
    scenarios.append({
        "action": action.replace("_", " ").title(),
        "action_raw": action,
        "vitals_class": vitals_class,
        "level": level,
        "subject_id": int(seq.subject_id.iloc[0]),
        "hr": ts.HR.ffill().fillna(75).round(1).tolist()[:30],
        "spo2": ts.SpO2.ffill().fillna(97).round(1).tolist()[:30],
        "pulse": ts.Pulse.ffill().fillna(75).round(1).tolist()[:30],
        "time": ts.timestamp_s.round(1).tolist()[:30],
    })

with open("demo_combined/_dashboard_data.json", "w") as f:
    json.dump(scenarios, f)

for s in scenarios:
    act = s["action"]
    vit = s["vitals_class"]
    lev = s["level"]
    pts = len(s["hr"])
    print(f"  {act:25s} + {vit:15s} -> {lev:8s} ({pts} pts)")
print(f"\n{len(scenarios)} scenarios saved to demo_combined/_dashboard_data.json")
