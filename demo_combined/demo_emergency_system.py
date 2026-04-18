"""
Combined Emergency Detection System — Demo
===========================================

Two federated subsystems under one umbrella:
  A) Skeleton HAR   — action recognition from video/skeleton keypoints
  B) Vitals Monitor — physiological anomaly detection from sensor data

Fusion: Decision-level emergency classification on the integrated dataset.

Usage:
    python demo_combined/demo_emergency_system.py

Outputs:
    demo_combined/system_overview.png    — architecture + data overview
    demo_combined/patient_dashboard.png  — 4-patient monitoring dashboard
    demo_combined/model_performance.png  — both models + fusion accuracy
"""

import os
import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "ntu_action_dataset")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── NTU-25 skeleton graph (for visualization) ───────────────────────────────
NTU25_EDGES = [
    (0, 1), (1, 20), (20, 2), (2, 3),           # spine
    (20, 4), (4, 5), (5, 6), (6, 7),             # left arm
    (7, 21), (7, 22),                             # left hand
    (20, 8), (8, 9), (9, 10), (10, 11),           # right arm
    (11, 23), (11, 24),                           # right hand
    (0, 12), (12, 13), (13, 14), (14, 15),        # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),        # right leg
]

EMERGENCY_LEVELS = {
    "CRITICAL": {"color": "#d32f2f", "icon": "⚠"},
    "HIGH":     {"color": "#f57c00", "icon": "▲"},
    "MEDIUM":   {"color": "#fbc02d", "icon": "●"},
    "LOW":      {"color": "#388e3c", "icon": "✓"},
}

# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load integrated_v2 dataset and skeleton sequences."""
    print("Loading integrated dataset...")

    # Integrated vitals + skeleton features
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "integrated_v2", "*.parquet")))
    iv2 = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  integrated_v2: {iv2.shape[0]:,} rows, {iv2.sequence_id.nunique()} sequences, "
          f"{iv2.subject_id.nunique()} subjects")

    # Raw 3D skeleton keypoints
    seqs = np.load(os.path.join(DATASET_DIR, "sequences.npy"))
    with open(os.path.join(DATASET_DIR, "sequence_ids.json")) as f:
        seq_ids = json.load(f)
    ntu_to_idx = {nid: i for i, nid in enumerate(seq_ids)}
    print(f"  sequences.npy: {seqs.shape} (samples, frames, joints, xyz)")

    return iv2, seqs, ntu_to_idx


def prepare_features(iv2):
    """Aggregate per-sequence features for both classifiers."""
    vitals_cols = ["HR", "Pulse", "SpO2", "etCO2", "awRR"]
    ntu_feat_cols = [c for c in iv2.columns if c.startswith("ntu_") and c not in
                     ("ntu_action_class", "ntu_action_name", "ntu_is_medical", "ntu_id")]

    # Aggregate per sequence
    agg_dict = {col: "mean" for col in vitals_cols}
    agg_dict.update({col: "first" for col in ntu_feat_cols})
    agg_dict["class_name"] = "first"
    agg_dict["ntu_action_name"] = "first"
    agg_dict["ntu_is_medical"] = "first"
    agg_dict["ntu_id"] = "first"
    agg_dict["subject_id"] = "first"

    seq_df = iv2.groupby("sequence_id").agg(agg_dict).reset_index()

    # Fill NaN vitals with column median (simulating sensor dropout handling)
    for col in vitals_cols:
        med = seq_df[col].median()
        seq_df[col] = seq_df[col].fillna(med)

    # Fill NaN skeleton features
    for col in ntu_feat_cols:
        seq_df[col] = seq_df[col].fillna(0.0)

    return seq_df, vitals_cols, ntu_feat_cols


# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL TRAINING (lightweight — seconds on tabular data)
# ═════════════════════════════════════════════════════════════════════════════

def train_models(seq_df, vitals_cols, ntu_feat_cols):
    """Train two classifiers: vitals condition + skeleton action."""
    print("\nTraining classifiers...")

    # Encode labels
    le_vitals = LabelEncoder().fit(seq_df["class_name"])
    le_action = LabelEncoder().fit(seq_df["ntu_action_name"])

    y_vitals = le_vitals.transform(seq_df["class_name"])
    y_action = le_action.transform(seq_df["ntu_action_name"])

    # Vitals features
    X_vitals = RobustScaler().fit_transform(seq_df[vitals_cols].values)

    # Skeleton features
    X_skel = RobustScaler().fit_transform(seq_df[ntu_feat_cols].values)

    # Train/test split (subject-aware: subjects 0-239 train, 240-299 test)
    train_mask = seq_df["subject_id"] < 240
    test_mask = ~train_mask

    # ── Vitals classifier ──
    clf_vitals = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
    clf_vitals.fit(X_vitals[train_mask], y_vitals[train_mask])
    vitals_acc = accuracy_score(y_vitals[test_mask], clf_vitals.predict(X_vitals[test_mask]))
    print(f"  Vitals classifier:  {vitals_acc:.1%} accuracy ({len(le_vitals.classes_)} classes)")

    # ── Skeleton action classifier ──
    clf_action = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    clf_action.fit(X_skel[train_mask], y_action[train_mask])
    action_acc = accuracy_score(y_action[test_mask], clf_action.predict(X_skel[test_mask]))
    print(f"  Action classifier:  {action_acc:.1%} accuracy ({len(le_action.classes_)} classes)")

    # Predictions + probabilities on test set
    pred_vitals = clf_vitals.predict(X_vitals[test_mask])
    pred_action = clf_action.predict(X_skel[test_mask])
    prob_vitals = clf_vitals.predict_proba(X_vitals[test_mask])
    prob_action = clf_action.predict_proba(X_skel[test_mask])

    results = {
        "le_vitals": le_vitals, "le_action": le_action,
        "clf_vitals": clf_vitals, "clf_action": clf_action,
        "pred_vitals": pred_vitals, "pred_action": pred_action,
        "prob_vitals": prob_vitals, "prob_action": prob_action,
        "y_vitals_test": y_vitals[test_mask], "y_action_test": y_action[test_mask],
        "test_mask": test_mask, "train_mask": train_mask,
        "vitals_acc": vitals_acc, "action_acc": action_acc,
        "X_vitals": X_vitals, "X_skel": X_skel,
    }
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 3. EMERGENCY FUSION
# ═════════════════════════════════════════════════════════════════════════════

MEDICAL_ACTIONS = {"falling_down", "staggering", "chest_pain", "headache",
                   "back_pain", "neck_pain", "nausea_vomiting"}
ABNORMAL_VITALS = {"Bradycardia", "Tachycardia", "Low pressure", "High pressure",
                   "Nitroglycerine"}


def compute_emergency_score(action_name, action_conf, vitals_name, vitals_conf):
    """Decision-level fusion: combine both predictions into emergency score."""
    score = 0.0

    # Action component (0–0.5)
    if action_name in MEDICAL_ACTIONS:
        action_severity = {"falling_down": 1.0, "staggering": 0.85,
                           "chest_pain": 0.9, "nausea_vomiting": 0.7,
                           "headache": 0.5, "back_pain": 0.4,
                           "neck_pain": 0.4}
        score += 0.5 * action_severity.get(action_name, 0.5) * action_conf
    else:
        score += 0.0  # daily actions contribute nothing

    # Vitals component (0–0.5)
    if vitals_name in ABNORMAL_VITALS:
        vitals_severity = {"Bradycardia": 0.9, "Tachycardia": 0.85,
                           "Low pressure": 0.8, "High pressure": 0.75,
                           "Nitroglycerine": 0.6}
        score += 0.5 * vitals_severity.get(vitals_name, 0.5) * vitals_conf
    else:
        score += 0.0  # normal vitals contribute nothing

    return min(score, 1.0)


def score_to_level(score):
    if score >= 0.6:
        return "CRITICAL"
    elif score >= 0.4:
        return "HIGH"
    elif score >= 0.2:
        return "MEDIUM"
    else:
        return "LOW"


# ═════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION — System Overview
# ═════════════════════════════════════════════════════════════════════════════

def draw_system_overview(seq_df):
    """Figure 1: System architecture + data overview."""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.97, top=0.92, bottom=0.08)

    fig.suptitle("Combined Emergency Detection System", fontsize=18, fontweight="bold", y=0.97)

    # ── Panel A: Architecture diagram ──
    ax = fig.add_subplot(gs[0, :2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("System Architecture", fontsize=13, fontweight="bold", loc="left")

    boxes = [
        (0.3, 3.5, 2.8, 1.2, "#e3f2fd", "Subsystem A\nSkeleton HAR (FL)\nSTGCN++ → 10 actions"),
        (0.3, 1.2, 2.8, 1.2, "#fce4ec", "Subsystem B\nVitals Monitor (FL)\nMLP → 6 conditions"),
        (4.2, 2.2, 2.5, 1.5, "#fff3e0", "Decision Fusion\n\nEmergency\nScoring"),
        (7.8, 2.2, 1.8, 1.5, "#e8f5e9", "Alert\nLevel\n\nCRITICAL\nHIGH / LOW"),
    ]
    for x, y, w, h, color, text in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="#555", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.5,
                fontfamily="monospace")

    # Arrows
    for xs, ys, xe, ye in [(3.1, 4.1, 4.2, 3.3), (3.1, 1.8, 4.2, 2.7),
                            (6.7, 2.95, 7.8, 2.95)]:
        ax.annotate("", xy=(xe, ye), xytext=(xs, ys),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))

    ax.text(1.7, 0.5, "Video / Skeleton\nkeypoints", ha="center", fontsize=8, color="#1565c0")
    ax.annotate("", xy=(1.7, 1.2), xytext=(1.7, 0.85),
                arrowprops=dict(arrowstyle="-|>", color="#1565c0", lw=1))
    ax.text(1.7, 0.0, "HR, SpO₂, etCO₂\nPulse, awRR", ha="center", fontsize=8, color="#c62828",
            va="top")

    # ── Panel B: Class distribution ──
    ax2 = fig.add_subplot(gs[0, 2])
    action_counts = seq_df["ntu_action_name"].value_counts()
    colors = ["#ef5350" if a in MEDICAL_ACTIONS else "#42a5f5" for a in action_counts.index]
    ax2.barh(range(len(action_counts)), action_counts.values, color=colors)
    ax2.set_yticks(range(len(action_counts)))
    ax2.set_yticklabels([a.replace("_", " ") for a in action_counts.index], fontsize=8)
    ax2.set_xlabel("Sequences")
    ax2.set_title("Action Distribution", fontsize=11, fontweight="bold")
    ax2.invert_yaxis()
    # Legend
    from matplotlib.patches import Patch
    ax2.legend([Patch(color="#ef5350"), Patch(color="#42a5f5")],
               ["Medical", "Daily"], loc="lower right", fontsize=8)

    # ── Panel C: Vitals class distribution ──
    ax3 = fig.add_subplot(gs[1, 0])
    vit_counts = seq_df["class_name"].value_counts()
    colors_v = ["#66bb6a" if v == "Normal" else "#ef5350" for v in vit_counts.index]
    ax3.bar(range(len(vit_counts)), vit_counts.values, color=colors_v)
    ax3.set_xticks(range(len(vit_counts)))
    ax3.set_xticklabels(vit_counts.index, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Sequences")
    ax3.set_title("Vitals Conditions", fontsize=11, fontweight="bold")

    # ── Panel D: Cross-tabulation heatmap ──
    ax4 = fig.add_subplot(gs[1, 1:])
    ct = pd.crosstab(seq_df["class_name"], seq_df["ntu_action_name"])
    ct = ct.reindex(columns=sorted(ct.columns), index=sorted(ct.index))
    im = ax4.imshow(ct.values, cmap="YlOrRd", aspect="auto")
    ax4.set_xticks(range(ct.shape[1]))
    ax4.set_xticklabels([c.replace("_", " ") for c in ct.columns], rotation=45, ha="right", fontsize=7)
    ax4.set_yticks(range(ct.shape[0]))
    ax4.set_yticklabels(ct.index, fontsize=8)
    ax4.set_title("Vitals × Action Cross-Tab (sequence counts)", fontsize=11, fontweight="bold")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            val = ct.values[i, j]
            ax4.text(j, i, str(val), ha="center", va="center", fontsize=7,
                     color="white" if val > ct.values.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax4, shrink=0.7)

    path = os.path.join(OUTPUT_DIR, "system_overview.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION — Patient Dashboard
# ═════════════════════════════════════════════════════════════════════════════

def draw_skeleton(ax, joints, title=""):
    """Draw a 3D NTU-25 skeleton in 2D (x-z front view)."""
    x, z = joints[:, 0], joints[:, 1]  # x = horizontal, y = height (NTU y-axis)
    for i, j in NTU25_EDGES:
        ax.plot([x[i], x[j]], [z[i], z[j]], color="#1565c0", linewidth=1.5, zorder=1)
    ax.scatter(x, z, c="#1565c0", s=25, zorder=2, edgecolors="white", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(x.min() - 0.3, x.max() + 0.3)
    ax.set_ylim(z.min() - 0.3, z.max() + 0.3)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")


def draw_patient_dashboard(iv2, seqs, ntu_to_idx, seq_df, results):
    """Figure 2: 4-patient monitoring dashboard with vitals + skeleton + alert."""

    # Select 4 interesting test sequences
    test_df = seq_df[results["test_mask"]].copy()
    test_df["pred_vitals"] = results["le_vitals"].inverse_transform(results["pred_vitals"])
    test_df["pred_action"] = results["le_action"].inverse_transform(results["pred_action"])
    test_df["conf_vitals"] = results["prob_vitals"].max(axis=1)
    test_df["conf_action"] = results["prob_action"].max(axis=1)

    # Pick representative samples
    candidates = []
    # CRITICAL: falling + abnormal vitals
    c = test_df[(test_df["ntu_action_name"] == "falling_down") &
                (test_df["class_name"].isin(ABNORMAL_VITALS))]
    if len(c) > 0:
        candidates.append(c.iloc[0])
    # HIGH: staggering or chest pain + abnormal
    c = test_df[(test_df["ntu_action_name"].isin(["staggering", "chest_pain"])) &
                (test_df["class_name"].isin(ABNORMAL_VITALS))]
    if len(c) > 0:
        candidates.append(c.iloc[0])
    # MEDIUM: medical action + normal vitals
    c = test_df[(test_df["ntu_action_name"].isin(MEDICAL_ACTIONS)) &
                (test_df["class_name"] == "Normal")]
    if len(c) > 0:
        candidates.append(c.iloc[0])
    # LOW: daily action + normal vitals
    c = test_df[(test_df["ntu_action_name"].isin(["walking_towards", "sitting_down", "standing_up"])) &
                (test_df["class_name"] == "Normal")]
    if len(c) > 0:
        candidates.append(c.iloc[0])

    # Fallback — fill to 4
    while len(candidates) < 4:
        candidates.append(test_df.iloc[len(candidates)])

    n_panels = len(candidates)
    fig = plt.figure(figsize=(18, 4.2 * n_panels))
    fig.patch.set_facecolor("white")
    fig.suptitle("Patient Monitoring Dashboard — Combined Emergency Detection",
                 fontsize=16, fontweight="bold", y=0.995)

    for row, sample in enumerate(candidates):
        sid = sample["sequence_id"]
        ntu_id = sample["ntu_id"]

        # Get vitals time series
        seq_rows = iv2[iv2["sequence_id"] == sid].sort_values("timestamp_s")

        # Get skeleton
        skel_idx = ntu_to_idx.get(ntu_id)
        skel = seqs[skel_idx] if skel_idx is not None else None  # (64, 25, 3)
        mid_frame = skel[skel.shape[0] // 2] if skel is not None else None

        # Emergency score
        action_name = sample["pred_action"]
        vitals_name = sample["pred_vitals"]
        action_conf = sample["conf_action"]
        vitals_conf = sample["conf_vitals"]
        e_score = compute_emergency_score(action_name, action_conf, vitals_name, vitals_conf)
        level = score_to_level(e_score)
        level_info = EMERGENCY_LEVELS[level]

        # ── Row layout: [skeleton | vitals | predictions | alert] ──
        gs_row = gridspec.GridSpec(n_panels, 4, hspace=0.4, wspace=0.35,
                                   left=0.04, right=0.97, top=0.96, bottom=0.04)

        # Skeleton
        ax_skel = fig.add_subplot(gs_row[row, 0])
        if mid_frame is not None:
            draw_skeleton(ax_skel, mid_frame,
                          f"Skeleton — Subject {sample['subject_id']}")
        else:
            ax_skel.text(0.5, 0.5, "No skeleton\ndata", ha="center", va="center")
            ax_skel.axis("off")

        # Vitals time series
        ax_vit = fig.add_subplot(gs_row[row, 1])
        ts = seq_rows["timestamp_s"].values
        for col, color, label in [("HR", "#e53935", "HR"),
                                   ("SpO2", "#1e88e5", "SpO₂"),
                                   ("Pulse", "#fb8c00", "Pulse")]:
            vals = seq_rows[col].values
            mask = ~np.isnan(vals)
            if mask.sum() > 2:
                ax_vit.plot(ts[mask], vals[mask], color=color, linewidth=1.2,
                            label=label, alpha=0.85)
        ax_vit.set_xlabel("Time (s)", fontsize=8)
        ax_vit.set_ylabel("Value", fontsize=8)
        ax_vit.legend(fontsize=7, loc="upper right")
        ax_vit.set_title(f"Vitals — GT: {sample['class_name']}", fontsize=9, fontweight="bold")
        ax_vit.tick_params(labelsize=7)

        # Prediction bars
        ax_pred = fig.add_subplot(gs_row[row, 2])
        labels_bar = [f"Action:\n{action_name.replace('_',' ')}", f"Vitals:\n{vitals_name}"]
        confs = [action_conf, vitals_conf]
        bar_colors = ["#1565c0", "#c62828"]
        bars = ax_pred.barh([0, 1], confs, color=bar_colors, height=0.5)
        ax_pred.set_xlim(0, 1)
        ax_pred.set_yticks([0, 1])
        ax_pred.set_yticklabels(labels_bar, fontsize=8)
        ax_pred.set_xlabel("Confidence", fontsize=8)
        ax_pred.set_title(f"Predictions — GT: {sample['ntu_action_name'].replace('_',' ')}",
                          fontsize=9, fontweight="bold")
        for bar, conf in zip(bars, confs):
            ax_pred.text(conf + 0.02, bar.get_y() + bar.get_height() / 2,
                         f"{conf:.0%}", va="center", fontsize=9, fontweight="bold")
        ax_pred.tick_params(labelsize=7)

        # Alert panel
        ax_alert = fig.add_subplot(gs_row[row, 3])
        ax_alert.set_xlim(0, 1)
        ax_alert.set_ylim(0, 1)
        ax_alert.axis("off")

        alert_bg = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.05",
                                   facecolor=level_info["color"], alpha=0.15,
                                   edgecolor=level_info["color"], linewidth=3)
        ax_alert.add_patch(alert_bg)
        ax_alert.text(0.5, 0.7, level_info["icon"], ha="center", va="center",
                      fontsize=28, color=level_info["color"])
        ax_alert.text(0.5, 0.42, level, ha="center", va="center",
                      fontsize=18, fontweight="bold", color=level_info["color"])
        ax_alert.text(0.5, 0.2, f"Score: {e_score:.2f}", ha="center", va="center",
                      fontsize=11, color="#333")

    path = os.path.join(OUTPUT_DIR, "patient_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION — Model Performance + Fusion
# ═════════════════════════════════════════════════════════════════════════════

def draw_performance(seq_df, results):
    """Figure 3: Both models' accuracy + fusion emergency detection."""
    le_vitals = results["le_vitals"]
    le_action = results["le_action"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Model Performance & Emergency Fusion", fontsize=15, fontweight="bold")

    # ── Panel A: Vitals confusion matrix ──
    cm_v = confusion_matrix(results["y_vitals_test"], results["pred_vitals"])
    im = axes[0].imshow(cm_v, cmap="Blues", interpolation="nearest")
    axes[0].set_xticks(range(len(le_vitals.classes_)))
    axes[0].set_xticklabels(le_vitals.classes_, rotation=45, ha="right", fontsize=7)
    axes[0].set_yticks(range(len(le_vitals.classes_)))
    axes[0].set_yticklabels(le_vitals.classes_, fontsize=7)
    axes[0].set_title(f"Vitals Classifier — {results['vitals_acc']:.1%}", fontsize=11,
                      fontweight="bold")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")
    for i in range(cm_v.shape[0]):
        for j in range(cm_v.shape[1]):
            axes[0].text(j, i, str(cm_v[i, j]), ha="center", va="center", fontsize=7,
                         color="white" if cm_v[i, j] > cm_v.max() * 0.5 else "black")

    # ── Panel B: Action confusion matrix ──
    cm_a = confusion_matrix(results["y_action_test"], results["pred_action"])
    short_names = [a[:8] for a in le_action.classes_]
    im2 = axes[1].imshow(cm_a, cmap="Oranges", interpolation="nearest")
    axes[1].set_xticks(range(len(le_action.classes_)))
    axes[1].set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    axes[1].set_yticks(range(len(le_action.classes_)))
    axes[1].set_yticklabels(short_names, fontsize=7)
    axes[1].set_title(f"Action Classifier — {results['action_acc']:.1%}", fontsize=11,
                      fontweight="bold")
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")
    for i in range(cm_a.shape[0]):
        for j in range(cm_a.shape[1]):
            axes[1].text(j, i, str(cm_a[i, j]), ha="center", va="center", fontsize=7,
                         color="white" if cm_a[i, j] > cm_a.max() * 0.5 else "black")

    # ── Panel C: Emergency fusion - score distribution by true severity ──
    test_df = seq_df[results["test_mask"]].copy()
    test_df["pred_action_name"] = results["le_action"].inverse_transform(results["pred_action"])
    test_df["pred_vitals_name"] = results["le_vitals"].inverse_transform(results["pred_vitals"])
    test_df["conf_action"] = results["prob_action"].max(axis=1)
    test_df["conf_vitals"] = results["prob_vitals"].max(axis=1)

    scores = []
    true_levels = []
    for _, row in test_df.iterrows():
        s = compute_emergency_score(row["pred_action_name"], row["conf_action"],
                                    row["pred_vitals_name"], row["conf_vitals"])
        scores.append(s)

        # Ground-truth severity
        is_med_action = row["ntu_action_name"] in MEDICAL_ACTIONS
        is_abn_vitals = row["class_name"] in ABNORMAL_VITALS
        if is_med_action and is_abn_vitals:
            true_levels.append("Both abnormal")
        elif is_med_action:
            true_levels.append("Action only")
        elif is_abn_vitals:
            true_levels.append("Vitals only")
        else:
            true_levels.append("Both normal")

    colors_map = {"Both abnormal": "#d32f2f", "Action only": "#f57c00",
                  "Vitals only": "#fbc02d", "Both normal": "#388e3c"}
    for cat in ["Both abnormal", "Action only", "Vitals only", "Both normal"]:
        cat_scores = [s for s, t in zip(scores, true_levels) if t == cat]
        if cat_scores:
            axes[2].hist(cat_scores, bins=20, alpha=0.6, label=f"{cat} (n={len(cat_scores)})",
                         color=colors_map[cat], edgecolor="white")
    axes[2].axvline(0.6, color="#d32f2f", linestyle="--", linewidth=1, label="CRITICAL threshold")
    axes[2].axvline(0.4, color="#f57c00", linestyle="--", linewidth=1, label="HIGH threshold")
    axes[2].axvline(0.2, color="#fbc02d", linestyle="--", linewidth=1, label="MEDIUM threshold")
    axes[2].set_xlabel("Emergency Score")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Fusion Score Distribution by Ground Truth", fontsize=11, fontweight="bold")
    axes[2].legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_performance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print fusion summary
    pred_levels = [score_to_level(s) for s in scores]
    print(f"\n  Emergency fusion summary (n={len(scores)} test sequences):")
    for lev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        n = pred_levels.count(lev)
        print(f"    {lev:10s}: {n:4d} ({n/len(scores):5.1%})")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print(" COMBINED EMERGENCY DETECTION SYSTEM — DEMO")
    print(" Skeleton HAR + Vitals Anomaly → Emergency Notification")
    print("=" * 65)

    iv2, seqs, ntu_to_idx = load_data()
    seq_df, vitals_cols, ntu_feat_cols = prepare_features(iv2)

    print(f"\n  Prepared {len(seq_df)} sequences")
    print(f"  Vitals features: {len(vitals_cols)} ({', '.join(vitals_cols)})")
    print(f"  Skeleton features: {len(ntu_feat_cols)}")

    results = train_models(seq_df, vitals_cols, ntu_feat_cols)

    print("\nGenerating visualizations...")
    draw_system_overview(seq_df)
    draw_patient_dashboard(iv2, seqs, ntu_to_idx, seq_df, results)
    draw_performance(seq_df, results)

    print("\n" + "=" * 65)
    print(" DEMO COMPLETE")
    print(f" Vitals accuracy:  {results['vitals_acc']:.1%}")
    print(f" Action accuracy:  {results['action_acc']:.1%}")
    print(f" Output dir: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
