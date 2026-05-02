"""
Inference module for 2D (COCO-17) and 3D (NTU-25) STGCN++ models.

Loads trained checkpoints and provides predict() for single or batch samples.
Used by the Streamlit dashboard and standalone scripts.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Ensure demo1_fed_skeleton is importable
_ROOT = Path(__file__).resolve().parent.parent
_DEMO_DIR = _ROOT / "demo1_fed_skeleton"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from models.stgcn import build_model  # noqa: E402

# ── Labels ──────────────────────────────────────────────────────────────────
MEDICAL_LABELS = [
    "falling", "staggering", "nausea/vomiting",
    "touch head", "touch chest", "touch back", "touch neck",
    "sneeze/cough", "standing up", "sitting down",
    "walking towards", "walking apart",
    "drinking", "eating", "phone call",
]

LABEL_SEVERITY = {
    "falling": "CRITICAL", "staggering": "HIGH",
    "nausea/vomiting": "HIGH", "touch head": "MEDIUM",
    "touch chest": "MEDIUM", "touch back": "MEDIUM",
    "touch neck": "MEDIUM", "sneeze/cough": "LOW",
    "standing up": "NORMAL", "sitting down": "NORMAL",
    "walking towards": "NORMAL", "walking apart": "NORMAL",
    "drinking": "NORMAL", "eating": "NORMAL", "phone call": "NORMAL",
}

NUM_CLASSES = len(MEDICAL_LABELS)

# ── Skeleton graphs ─────────────────────────────────────────────────────────
COCO17_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12),  # legs
    (11, 12),                                  # hip
    (5, 11), (6, 12),                          # torso
    (5, 6),                                    # shoulders
    (5, 7), (6, 8), (7, 9), (8, 10),          # arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),  # face
]

COCO17_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

NTU25_EDGES = [
    (0, 1), (1, 20), (20, 2), (2, 3),           # spine
    (20, 4), (4, 5), (5, 6), (6, 7),             # left arm
    (7, 21), (7, 22),                             # left hand
    (20, 8), (8, 9), (9, 10), (10, 11),           # right arm
    (11, 23), (11, 24),                           # right hand
    (0, 12), (12, 13), (13, 14), (14, 15),        # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),        # right leg
]

NTU25_JOINT_NAMES = [
    "base of spine", "middle of spine", "neck", "head",
    "left shoulder", "left elbow", "left wrist", "left hand",
    "right shoulder", "right elbow", "right wrist", "right hand",
    "left hip", "left knee", "left ankle", "left foot",
    "right hip", "right knee", "right ankle", "right foot",
    "spine", "tip of left hand", "left thumb",
    "tip of right hand", "right thumb",
]

# ── Body part coloring (for both graphs) ────────────────────────────────────
NTU25_BODY_PARTS = {
    "spine":     {"joints": [0, 1, 2, 3, 20], "color": "#4FC3F7"},
    "left_arm":  {"joints": [4, 5, 6, 7, 21, 22], "color": "#81C784"},
    "right_arm": {"joints": [8, 9, 10, 11, 23, 24], "color": "#FF8A65"},
    "left_leg":  {"joints": [12, 13, 14, 15], "color": "#CE93D8"},
    "right_leg": {"joints": [16, 17, 18, 19], "color": "#FFD54F"},
}

COCO17_BODY_PARTS = {
    "head":      {"joints": [0, 1, 2, 3, 4], "color": "#4FC3F7"},
    "torso":     {"joints": [5, 6, 11, 12], "color": "#90A4AE"},
    "left_arm":  {"joints": [5, 7, 9], "color": "#81C784"},
    "right_arm": {"joints": [6, 8, 10], "color": "#FF8A65"},
    "left_leg":  {"joints": [11, 13, 15], "color": "#CE93D8"},
    "right_leg": {"joints": [12, 14, 16], "color": "#FFD54F"},
}


def get_joint_color(joint_idx: int, graph: str) -> str:
    """Return hex color for a joint based on body part."""
    parts = NTU25_BODY_PARTS if graph == "ntu" else COCO17_BODY_PARTS
    for part_info in parts.values():
        if joint_idx in part_info["joints"]:
            return part_info["color"]
    return "#BDBDBD"


def get_edge_color(i: int, j: int, graph: str) -> str:
    """Return hex color for an edge based on the body part of its joints."""
    parts = NTU25_BODY_PARTS if graph == "ntu" else COCO17_BODY_PARTS
    for part_info in parts.values():
        if i in part_info["joints"] and j in part_info["joints"]:
            return part_info["color"]
    return "#BDBDBD"


# ── Model loading ───────────────────────────────────────────────────────────
_DEFAULT_CKPT_DIR = _DEMO_DIR / "outputs"

_MODEL_CONFIGS = {
    "2d": dict(graph="coco", num_classes=NUM_CLASSES, in_channels=3,
               base_channels=64, num_stages=6, num_person=2, dropout=0.0),
    "3d": dict(graph="ntu", num_classes=NUM_CLASSES, in_channels=3,
               base_channels=64, num_stages=6, num_person=2, dropout=0.0),
}


class ActionRecognizer:
    """Wrapper around STGCN++ for inference."""

    def __init__(self, mode: str = "2d",
                 checkpoint: Optional[str] = None,
                 device: str = "cpu"):
        assert mode in ("2d", "3d"), f"mode must be '2d' or '3d', got {mode!r}"
        self.mode = mode
        self.graph = "coco" if mode == "2d" else "ntu"
        self.num_joints = 17 if mode == "2d" else 25
        self.device = torch.device(device)

        self.model = build_model(**_MODEL_CONFIGS[mode]).to(self.device)
        ckpt = checkpoint or str(
            _DEFAULT_CKPT_DIR / f"cent_{mode}_15cls_best.pt")
        state = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> dict:
        """
        Predict action from skeleton tensor.

        Args:
            x: (M, T, V, C) or (N, M, T, V, C) numpy array.
               M=2 persons, T=100 frames, V=17|25 joints, C=3 channels.

        Returns:
            dict with keys: label, label_idx, confidence, severity,
                            top5 (list of (label, prob) tuples),
                            probs (full probability vector).
        """
        if x.ndim == 4:
            x = x[np.newaxis]  # add batch dim
        t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        logits = self.model(t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        top5_idx = probs.argsort()[::-1][:5]
        return {
            "label": MEDICAL_LABELS[idx],
            "label_idx": idx,
            "confidence": float(probs[idx]),
            "severity": LABEL_SEVERITY[MEDICAL_LABELS[idx]],
            "top5": [(MEDICAL_LABELS[i], float(probs[i])) for i in top5_idx],
            "probs": probs.tolist(),
        }

    def predict_batch(self, x: np.ndarray) -> list[dict]:
        """Predict for a batch of samples. x: (N, M, T, V, C)."""
        t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        logits = self.model(t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        results = []
        for p in probs:
            idx = int(p.argmax())
            top5_idx = p.argsort()[::-1][:5]
            results.append({
                "label": MEDICAL_LABELS[idx],
                "label_idx": idx,
                "confidence": float(p[idx]),
                "severity": LABEL_SEVERITY[MEDICAL_LABELS[idx]],
                "top5": [(MEDICAL_LABELS[i], float(p[i])) for i in top5_idx],
                "probs": p.tolist(),
            })
        return results


# ── Data loading helpers ────────────────────────────────────────────────────
def load_test_data(mode: str = "2d") -> tuple[np.ndarray, np.ndarray]:
    """Load centralized test set. Returns (x, y)."""
    sub = "fed_medical_2d" if mode == "2d" else "fed_medical_3d"
    path = _DEMO_DIR / "data" / sub / "centralized" / "test.npz"
    data = np.load(path)
    return data["x"], data["y"]


def get_random_samples(mode: str = "2d", n: int = 5,
                       class_idx: Optional[int] = None,
                       seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Get n random samples from test set, optionally filtered by class."""
    x, y = load_test_data(mode)
    rng = np.random.RandomState(seed)
    if class_idx is not None:
        mask = y == class_idx
        x, y = x[mask], y[mask]
    if len(x) == 0:
        return np.array([]), np.array([])
    idxs = rng.choice(len(x), size=min(n, len(x)), replace=False)
    return x[idxs], y[idxs]


# ── CLI demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on test samples")
    parser.add_argument("--mode", choices=["2d", "3d"], default="2d")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading {args.mode.upper()} model on {args.device}...")
    recognizer = ActionRecognizer(mode=args.mode, device=args.device)

    x, y = get_random_samples(args.mode, n=args.n, seed=123)
    print(f"\nRunning inference on {len(x)} test samples...\n")

    correct = 0
    for i, (sample, label) in enumerate(zip(x, y)):
        result = recognizer.predict(sample)
        ok = "OK" if result["label_idx"] == label else "WRONG"
        correct += result["label_idx"] == label
        print(f"  [{i+1:2d}] True: {MEDICAL_LABELS[label]:20s}  "
              f"Pred: {result['label']:20s}  "
              f"Conf: {result['confidence']:.3f}  "
              f"Severity: {result['severity']:8s}  {ok}")

    print(f"\nAccuracy: {correct}/{len(x)} = {correct/len(x):.1%}")
