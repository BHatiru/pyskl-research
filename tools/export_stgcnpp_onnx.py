#!/usr/bin/env python
"""Export STGCN++ model from pyskl to ONNX format.

This script loads a trained STGCN++ checkpoint and exports it to ONNX,
enabling inference without mmcv, mmpose, mmdet, or pyskl at runtime.

Usage:
    conda activate pyskl
    python tools/export_stgcnpp_onnx.py \
        --config configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py \
        --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth \
        --output demo/onnx_models/stgcnpp_ntu120_xsub_hrnet_j.onnx

The exported model takes input of shape (N, M, T, V, C):
    N = batch size
    M = number of persons (2, padded with zeros)
    T = number of frames (100, uniformly sampled)
    V = number of joints (17, COCO format)
    C = channels (3: x_normalized, y_normalized, confidence_score)

Output shape: (N, num_classes) logits.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class STGCNppONNXWrapper(nn.Module):
    """Wraps STGCN backbone + GCNHead into a single ONNX-exportable module.

    Combines the backbone forward, adaptive average pooling, person averaging,
    and the final classification linear layer.
    """

    def __init__(self, backbone, cls_head):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_cls = cls_head.fc_cls

    def forward(self, x):
        """
        Args:
            x: (N, M, T, V, C) skeleton input
        Returns:
            logits: (N, num_classes)
        """
        # Backbone: (N, M, T, V, C) -> (N, M, C_out, T', V)
        feat = self.backbone(x)

        N, M, C, T, V = feat.shape
        # Pool over time and joints
        feat = feat.reshape(N * M, C, T, V)
        feat = self.pool(feat)  # (N*M, C, 1, 1)
        feat = feat.reshape(N, M, C)
        feat = feat.mean(dim=1)  # (N, C) - average over persons

        # Classify
        logits = self.fc_cls(feat)  # (N, num_classes)
        return logits


def main():
    parser = argparse.ArgumentParser(description="Export STGCN++ to ONNX")
    parser.add_argument("--config", required=True, help="pyskl config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path or URL")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Temporal dimension (must match training)",
    )
    parser.add_argument("--num-person", type=int, default=2)
    parser.add_argument("--num-joints", type=int, default=17)
    parser.add_argument(
        "--in-channels", type=int, default=3, help="Input channels (x, y, score)"
    )
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument(
        "--simplify", action="store_true", help="Simplify ONNX with onnx-simplifier"
    )
    args = parser.parse_args()

    # -- Build model from pyskl config --
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from pyskl.models import build_model
    from pyskl.utils import cache_checkpoint

    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model)

    # Handle URL or local checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path.startswith("http"):
        ckpt_path = cache_checkpoint(ckpt_path)
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model.eval()

    print(f"Loaded model from {args.checkpoint}")
    print(f"  Backbone: {type(model.backbone).__name__}")
    print(f"  Head: {type(model.cls_head).__name__}")
    print(f"  Num classes: {model.cls_head.fc_cls.out_features}")

    # -- Create ONNX wrapper --
    wrapper = STGCNppONNXWrapper(model.backbone, model.cls_head)
    wrapper.eval()

    # -- Create sample input --
    T = args.num_frames
    M = args.num_person
    V = args.num_joints
    C = args.in_channels
    dummy = torch.randn(1, M, T, V, C)

    # -- Verify PyTorch output --
    with torch.no_grad():
        pt_out = wrapper(dummy)
    print(f"  PyTorch output shape: {pt_out.shape}")  # (1, 120)

    # -- Export to ONNX --
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["skeleton"],
        output_names=["logits"],
        dynamic_axes={
            "skeleton": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"  Exported ONNX to {args.output}")

    # -- Optional: simplify --
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify

            model_onnx = onnx.load(args.output)
            model_onnx, ok = simplify(model_onnx)
            assert ok, "ONNX simplification failed"
            onnx.save(model_onnx, args.output)
            print("  Simplified ONNX model")
        except ImportError:
            print("  Warning: onnx-simplifier not installed, skipping")

    # -- Verify with ONNX Runtime --
    import onnxruntime as ort

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"skeleton": dummy.numpy()})[0]
    max_diff = np.abs(pt_out.numpy() - ort_out).max()
    print(f"  ONNX Runtime verification: max_diff = {max_diff:.6f}")
    if max_diff < 1e-4:
        print("  ✓ Export successful!")
    else:
        print(f"  ⚠ Difference is larger than expected ({max_diff:.4f})")

    # -- Print model info --
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nModel summary:")
    print(f"  File: {args.output}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  Input: skeleton (N, {M}, {T}, {V}, {C})")
    print(f"  Output: logits (N, {model.cls_head.fc_cls.out_features})")
    print(f"  Opset: {args.opset}")


if __name__ == "__main__":
    main()
