"""
Streamlit Dashboard — Medical Action Recognition Demo
=====================================================

Interactive web dashboard showing:
- 3D skeleton visualization with Plotly (rotatable, animated)
- Real-time inference from 2D & 3D STGCN++ models
- Per-class metrics and confusion matrices
- Side-by-side model comparison

Run:
    cd demo_combined
    streamlit run dashboard_app.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Ensure imports work
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "demo1_fed_skeleton"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import (
    ActionRecognizer,
    COCO17_BODY_PARTS,
    COCO17_EDGES,
    COCO17_JOINT_NAMES,
    LABEL_SEVERITY,
    MEDICAL_LABELS,
    NTU25_BODY_PARTS,
    NTU25_EDGES,
    NTU25_JOINT_NAMES,
    NUM_CLASSES,
    get_edge_color,
    get_joint_color,
    get_random_samples,
    load_test_data,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Action Recognition",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEVERITY_COLORS = {
    "CRITICAL": "#d32f2f",
    "HIGH": "#f57c00",
    "MEDIUM": "#fbc02d",
    "LOW": "#66bb6a",
    "NORMAL": "#42a5f5",
}

# ── Cached model loading ───────────────────────────────────────────────────
@st.cache_resource
def load_model(mode: str, device: str):
    return ActionRecognizer(mode=mode, device=device)


@st.cache_data
def cached_test_data(mode: str):
    return load_test_data(mode)


# ── Skeleton visualization ──────────────────────────────────────────────────
def plot_skeleton_3d(skeleton: np.ndarray, graph: str, frame_idx: int = 50,
                     person_idx: int = 0, title: str = "") -> go.Figure:
    """
    Plot a single frame of a skeleton in 3D using Plotly.

    skeleton: (M, T, V, C) — M persons, T frames, V joints, C=3 channels
    """
    edges = NTU25_EDGES if graph == "ntu" else COCO17_EDGES
    joint_names = NTU25_JOINT_NAMES if graph == "ntu" else COCO17_JOINT_NAMES
    num_joints = 25 if graph == "ntu" else 17

    kp = skeleton[person_idx, frame_idx]  # (V, 3)

    # For 2D skeletons, C=(x, y, score) — use score as z with small spread
    is_2d = graph == "coco"
    if is_2d:
        x, y, score = kp[:, 0], kp[:, 1], kp[:, 2]
        z = np.zeros_like(x)  # flat plane for 2D
    else:
        x, y, z = kp[:, 0], kp[:, 1], kp[:, 2]
        score = np.ones(num_joints)  # 3D always has full confidence

    # Filter active joints
    active = np.abs(x) + np.abs(y) > 0.01

    fig = go.Figure()

    # Draw edges
    for i, j in edges:
        if not (active[i] and active[j]):
            continue
        color = get_edge_color(i, j, graph)
        fig.add_trace(go.Scatter3d(
            x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
            mode="lines",
            line=dict(color=color, width=6),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Draw joints
    colors = [get_joint_color(j, graph) for j in range(num_joints)]
    sizes = [10 if active[j] else 0 for j in range(num_joints)]
    hover = [f"<b>{joint_names[j]}</b><br>x={x[j]:.3f}<br>y={y[j]:.3f}<br>z={z[j]:.3f}"
             if active[j] else "" for j in range(num_joints)]

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        marker=dict(size=sizes, color=colors, opacity=0.9,
                    line=dict(width=1, color="white")),
        text=[joint_names[j] if active[j] else "" for j in range(num_joints)],
        textposition="top center",
        textfont=dict(size=8, color="white"),
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    ))

    # Layout — Y is vertical (up) in skeleton data
    axis_range = [-1.2, 1.2]
    if is_2d:
        # 2D: look straight at the XY plane from +Z, hide Z-axis
        camera = dict(
            eye=dict(x=0, y=0, z=2.5),
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
        )
    else:
        # 3D: front view, slightly elevated and to the right for natural perspective
        camera = dict(
            eye=dict(x=0.4, y=0.3, z=2.2),
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
        )
    z_axis = (
        dict(range=[-0.1, 0.1], visible=False, showbackground=False)
        if is_2d else
        dict(range=axis_range, showgrid=True, gridcolor="#333",
             showbackground=True, backgroundcolor="#0f3460",
             title=dict(text="Z", font=dict(color="#aaa")))
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis=dict(range=axis_range, showgrid=True, gridcolor="#333",
                       showbackground=True, backgroundcolor="#1a1a2e",
                       title=dict(text="X", font=dict(color="#aaa"))),
            yaxis=dict(range=axis_range, showgrid=True, gridcolor="#333",
                       showbackground=True, backgroundcolor="#16213e",
                       title=dict(text="Y (up)" if not is_2d else "Y",
                                  font=dict(color="#aaa"))),
            zaxis=z_axis,
            camera=camera,
            aspectmode="cube" if not is_2d else "manual",
            aspectratio=dict(x=1, y=1, z=0.01) if is_2d else None,
        ),
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0a0a1a",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )
    return fig


def plot_skeleton_sequence(skeleton: np.ndarray, graph: str,
                           person_idx: int = 0, n_frames: int = 8,
                           title: str = "") -> go.Figure:
    """Plot multiple frames side-by-side as a sequence strip."""
    edges = NTU25_EDGES if graph == "ntu" else COCO17_EDGES
    T = skeleton.shape[1]
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)
    is_2d = graph == "coco"

    fig = go.Figure()
    spacing = 3.0  # x-offset between frames

    for fi, frame_idx in enumerate(frame_indices):
        kp = skeleton[person_idx, frame_idx]  # (V, 3)
        x_off = fi * spacing

        if is_2d:
            x, y = kp[:, 0] + x_off, kp[:, 1], 
            z = np.zeros(len(kp))
            score = kp[:, 2]
        else:
            x, y, z = kp[:, 0] + x_off, kp[:, 1], kp[:, 2]
            score = np.ones(len(kp))

        active = np.abs(kp[:, 0]) + np.abs(kp[:, 1]) > 0.01

        # Edges
        for i, j in edges:
            if not (active[i] and active[j]):
                continue
            color = get_edge_color(i, j, graph)
            # Fade earlier frames
            opacity = 0.3 + 0.7 * (fi / max(n_frames - 1, 1))
            fig.add_trace(go.Scatter3d(
                x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
                mode="lines",
                line=dict(color=color, width=4),
                opacity=opacity,
                showlegend=False,
                hoverinfo="skip",
            ))

        # Joints
        colors = [get_joint_color(j, graph) for j in range(len(kp))]
        sizes = [7 if active[j] else 0 for j in range(len(kp))]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=sizes, color=colors,
                        opacity=0.3 + 0.7 * (fi / max(n_frames - 1, 1))),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Frame label
        cx = x_off
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[0], z=[-1.3 if not is_2d else -0.3],
            mode="text",
            text=[f"t={frame_idx}"],
            textfont=dict(size=9, color="#888"),
            showlegend=False,
            hoverinfo="skip",
        ))

    x_range = [-1.5, n_frames * spacing - spacing + 1.5]
    center_x = (n_frames - 1) * spacing / 2
    if is_2d:
        camera = dict(
            eye=dict(x=0, y=0, z=3.5),
            up=dict(x=0, y=1, z=0),
            center=dict(x=center_x, y=0, z=0),
        )
    else:
        camera = dict(
            eye=dict(x=0.3, y=0.3, z=3.5),
            up=dict(x=0, y=1, z=0),
            center=dict(x=center_x, y=0, z=0),
        )
    z_axis = (
        dict(range=[-0.1, 0.1], visible=False, showbackground=False, showgrid=False)
        if is_2d else
        dict(range=[-1.5, 1.5], showgrid=False, showbackground=True,
             backgroundcolor="#0a0a1a", title="")
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        scene=dict(
            xaxis=dict(range=x_range, showgrid=False, showticklabels=False,
                       showbackground=True, backgroundcolor="#0a0a1a", title=""),
            yaxis=dict(range=[-1.5, 1.5], showgrid=False,
                       showbackground=True, backgroundcolor="#0a0a1a", title=""),
            zaxis=z_axis,
            camera=camera,
            aspectmode="manual",
            aspectratio=dict(x=3, y=1, z=0.01 if is_2d else 1),
        ),
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0a0a1a",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=400,
    )
    return fig


def plot_animated_skeleton(skeleton: np.ndarray, graph: str,
                           person_idx: int = 0, fps: int = 10,
                           title: str = "") -> go.Figure:
    """Create an animated 3D skeleton that plays through frames."""
    edges = NTU25_EDGES if graph == "ntu" else COCO17_EDGES
    joint_names = NTU25_JOINT_NAMES if graph == "ntu" else COCO17_JOINT_NAMES
    is_2d = graph == "coco"
    T = skeleton.shape[1]
    # Sample ~30 frames for smooth animation
    step = max(1, T // 30)
    frame_indices = list(range(0, T, step))

    # Build frames
    frames = []
    for fi in frame_indices:
        kp = skeleton[person_idx, fi]
        if is_2d:
            x, y, z = kp[:, 0], kp[:, 1], np.zeros(len(kp))
        else:
            x, y, z = kp[:, 0], kp[:, 1], kp[:, 2]
        active = np.abs(kp[:, 0]) + np.abs(kp[:, 1]) > 0.01

        traces = []
        # Edges
        for i, j in edges:
            if not (active[i] and active[j]):
                continue
            color = get_edge_color(i, j, graph)
            traces.append(go.Scatter3d(
                x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
                mode="lines", line=dict(color=color, width=6),
                showlegend=False, hoverinfo="skip",
            ))
        # Joints
        colors = [get_joint_color(j, graph) for j in range(len(kp))]
        sizes = [10 if active[j] else 0 for j in range(len(kp))]
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.9,
                        line=dict(width=1, color="white")),
            showlegend=False,
            hovertext=[joint_names[j] if active[j] else "" for j in range(len(kp))],
            hoverinfo="text",
        ))
        frames.append(go.Frame(data=traces, name=str(fi)))

    # Initial frame
    fig = go.Figure(data=frames[0].data if frames else [], frames=frames)

    axis_range = [-1.2, 1.2]
    if is_2d:
        camera = dict(
            eye=dict(x=0, y=0, z=2.5),
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
        )
    else:
        camera = dict(
            eye=dict(x=0.4, y=0.3, z=2.2),
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
        )
    z_axis = (
        dict(range=[-0.1, 0.1], visible=False, showbackground=False)
        if is_2d else
        dict(range=axis_range, showbackground=True, backgroundcolor="#0f3460", title="Z")
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis=dict(range=axis_range, showbackground=True, backgroundcolor="#1a1a2e", title="X"),
            yaxis=dict(range=axis_range, showbackground=True, backgroundcolor="#16213e",
                       title="Y (up)" if not is_2d else "Y"),
            zaxis=z_axis,
            camera=camera,
            aspectmode="cube" if not is_2d else "manual",
            aspectratio=dict(x=1, y=1, z=0.01) if is_2d else None,
        ),
        # uirevision prevents camera reset when animation frames change
        uirevision="skeleton-animation",
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0a0a1a",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=20),
        height=550,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=0, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=1000 // fps, redraw=True),
                                      fromcurrent=True, mode="immediate",
                                      transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, steps=[
                dict(args=[[str(fi)], dict(frame=dict(duration=0, redraw=True),
                                           mode="immediate",
                                           transition=dict(duration=0))],
                     label=str(fi), method="animate")
                for fi in frame_indices
            ],
            x=0.05, len=0.9, y=-0.05,
            currentvalue=dict(prefix="Frame: ", font=dict(size=12, color="white")),
            font=dict(color="#888"),
        )],
    )
    return fig


# ── Sidebar ─────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.title("🏥 Medical HAR")
    st.sidebar.markdown("**STGCN++ Skeleton Action Recognition**")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["🦴 Skeleton Viewer", "🔍 Single Inference", "📊 Model Comparison",
         "📋 Results Overview"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Model Settings**")
    device = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0)
    st.sidebar.caption(f"15 medical action classes")
    st.sidebar.caption(f"STGCN++ (6-stage, ~450K params)")

    return page, device


# ── Pages ───────────────────────────────────────────────────────────────────
def page_skeleton_viewer(device: str):
    st.header("🦴 3D Skeleton Viewer")
    st.caption("Explore skeleton sequences from the test set. Rotate, zoom, and animate.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mode = st.selectbox("Skeleton Type", ["3d", "2d"], index=0,
                            format_func=lambda m: "3D NTU-25 joints" if m == "3d" else "2D COCO-17 joints")
    with col2:
        class_idx = st.selectbox("Action Class", range(NUM_CLASSES),
                                 format_func=lambda i: f"[{i}] {MEDICAL_LABELS[i]}")
    with col3:
        sample_idx = st.number_input("Sample #", min_value=0, max_value=50, value=0)
    with col4:
        viz_mode = st.selectbox("View", ["Animated", "Single Frame", "Sequence Strip"])

    graph = "ntu" if mode == "3d" else "coco"

    # Load samples for this class
    x, y = get_random_samples(mode, n=50, class_idx=class_idx, seed=42)
    if len(x) == 0:
        st.warning("No samples found for this class.")
        return

    idx = min(sample_idx, len(x) - 1)
    sample = x[idx]
    true_label = MEDICAL_LABELS[int(y[idx])]
    severity = LABEL_SEVERITY[true_label]

    # Run inference
    model = load_model(mode, device)
    result = model.predict(sample)

    # Header with prediction
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.metric("True Action", true_label)
    with c2:
        st.metric("Predicted", result["label"],
                  delta=f"{result['confidence']:.1%} confidence")
    with c3:
        sev_color = SEVERITY_COLORS.get(result["severity"], "#888")
        st.markdown(f"### <span style='color:{sev_color}'>{result['severity']}</span>",
                    unsafe_allow_html=True)

    # Visualization
    if viz_mode == "Animated":
        fig = plot_animated_skeleton(sample, graph, title=f"{true_label} — Animated Skeleton ({mode.upper()})")
        st.plotly_chart(fig, use_container_width=True)
    elif viz_mode == "Single Frame":
        frame = st.slider("Frame", 0, 99, 50)
        fig = plot_skeleton_3d(sample, graph, frame_idx=frame,
                               title=f"{true_label} — Frame {frame} ({mode.upper()})")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = plot_skeleton_sequence(sample, graph,
                                     title=f"{true_label} — Sequence ({mode.upper()})")
        st.plotly_chart(fig, use_container_width=True)

    # Top-5 predictions
    st.subheader("Top-5 Predictions")
    for label, prob in result["top5"]:
        sev = LABEL_SEVERITY[label]
        color = SEVERITY_COLORS.get(sev, "#888")
        st.progress(prob, text=f"**{label}** ({prob:.1%}) — {sev}")


def page_single_inference(device: str):
    st.header("🔍 Single Sample Inference")
    st.caption("Compare 2D and 3D model predictions on the same underlying action.")

    col1, col2 = st.columns(2)
    with col1:
        class_idx = st.selectbox("Action Class", range(NUM_CLASSES),
                                 format_func=lambda i: f"[{i}] {MEDICAL_LABELS[i]}",
                                 key="inf_class")
    with col2:
        sample_idx = st.number_input("Sample #", 0, 50, 0, key="inf_sample")

    # Load both
    x_2d, y_2d = get_random_samples("2d", n=50, class_idx=class_idx, seed=42)
    x_3d, y_3d = get_random_samples("3d", n=50, class_idx=class_idx, seed=42)

    model_2d = load_model("2d", device)
    model_3d = load_model("3d", device)

    left, right = st.columns(2)

    for col, mode, x, y, mdl in [
        (left, "2d", x_2d, y_2d, model_2d),
        (right, "3d", x_3d, y_3d, model_3d),
    ]:
        with col:
            graph = "coco" if mode == "2d" else "ntu"
            label = "2D COCO-17" if mode == "2d" else "3D NTU-25"
            st.subheader(f"📌 {label}")

            if len(x) == 0:
                st.warning("No samples")
                continue

            idx = min(sample_idx, len(x) - 1)
            sample = x[idx]
            result = mdl.predict(sample)

            sev_color = SEVERITY_COLORS.get(result["severity"], "#888")
            st.markdown(
                f"**Prediction:** {result['label']} "
                f"({result['confidence']:.1%}) — "
                f"<span style='color:{sev_color}'>{result['severity']}</span>",
                unsafe_allow_html=True,
            )

            fig = plot_skeleton_3d(sample, graph, frame_idx=50,
                                   title=f"{result['label']} — {label}")
            st.plotly_chart(fig, use_container_width=True, key=f"skel_{mode}")

            # Probability bars
            st.markdown("**Class Probabilities:**")
            probs = np.array(result["probs"])
            top5_idx = probs.argsort()[::-1][:5]
            for i in top5_idx:
                st.progress(float(probs[i]),
                            text=f"{MEDICAL_LABELS[i]} ({probs[i]:.1%})")


def page_model_comparison(device: str):
    st.header("📊 Model Comparison")

    import json

    output_dir = _ROOT / "demo1_fed_skeleton" / "outputs"

    # Load results
    results = {}
    for tag in ["cent_2d_15cls", "cent_3d_15cls"]:
        path = output_dir / f"{tag}_results.json"
        if path.exists():
            with open(path) as f:
                results[tag] = json.load(f)

    if not results:
        st.error("No results found. Run training first.")
        return

    # Summary metrics
    cols = st.columns(len(results))
    for col, (tag, r) in zip(cols, results.items()):
        with col:
            label = "2D COCO-17" if "2d" in tag else "3D NTU-25"
            st.subheader(label)
            st.metric("Best Accuracy", f"{r['best_acc']:.2%}")
            st.metric("Best Epoch", r["best_epoch"])
            st.metric("Parameters", f"{r['params']:,}")
            st.metric("Training Time", f"{r['total_time_s'] / 60:.1f} min")

    # Per-class comparison
    st.subheader("Per-Class Accuracy")

    if len(results) >= 2:
        r2 = results.get("cent_2d_15cls", {})
        r3 = results.get("cent_3d_15cls", {})
        pca2 = r2.get("per_class_acc", {})
        pca3 = r3.get("per_class_acc", {})

        classes = list(range(NUM_CLASSES))
        acc_2d = [pca2.get(str(c), {}).get("accuracy", 0) for c in classes]
        acc_3d = [pca3.get(str(c), {}).get("accuracy", 0) for c in classes]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="2D COCO-17", x=MEDICAL_LABELS, y=acc_2d,
                             marker_color="#42a5f5"))
        fig.add_trace(go.Bar(name="3D NTU-25", x=MEDICAL_LABELS, y=acc_3d,
                             marker_color="#ff8a65"))
        fig.update_layout(
            barmode="group", yaxis_title="Accuracy",
            yaxis_range=[0, 1.05],
            paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
            font=dict(color="white"),
            xaxis_tickangle=-45,
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices side by side
    st.subheader("Confusion Matrices")
    cm_cols = st.columns(len(results))
    for col, (tag, r) in zip(cm_cols, results.items()):
        with col:
            label = "2D COCO-17" if "2d" in tag else "3D NTU-25"
            cm = np.array(r.get("confusion_matrix", []))
            if cm.size == 0:
                st.info("No confusion matrix")
                continue
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
            fig = go.Figure(data=go.Heatmap(
                z=cm_norm, x=MEDICAL_LABELS, y=MEDICAL_LABELS,
                colorscale="Blues", zmin=0, zmax=1,
                text=cm, texttemplate="%{text}",
                textfont=dict(size=8),
            ))
            fig.update_layout(
                title=label, xaxis_title="Predicted", yaxis_title="True",
                paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
                font=dict(color="white", size=9),
                xaxis_tickangle=-45, height=550,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Learning curves
    st.subheader("Learning Curves")
    if any("history" in r for r in results.values()):
        fig = go.Figure()
        for tag, r in results.items():
            hist = r.get("history", [])
            if not hist:
                continue
            label = "2D" if "2d" in tag else "3D"
            epochs = [h["epoch"] for h in hist]
            fig.add_trace(go.Scatter(
                x=epochs, y=[h["test_acc"] for h in hist],
                name=f"{label} Test Acc", mode="lines",
            ))
            fig.add_trace(go.Scatter(
                x=epochs, y=[h["train_loss"] for h in hist],
                name=f"{label} Train Loss", mode="lines",
                yaxis="y2", line=dict(dash="dot"),
            ))
        fig.update_layout(
            xaxis_title="Epoch",
            yaxis=dict(title="Accuracy", side="left"),
            yaxis2=dict(title="Loss", side="right", overlaying="y"),
            paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
            font=dict(color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


def page_results_overview(device: str):
    st.header("📋 Results Overview")

    import json

    output_dir = _ROOT / "demo1_fed_skeleton" / "outputs"

    # Find all results
    result_files = sorted(output_dir.glob("*_results.json"))
    if not result_files:
        st.warning("No result files found.")
        return

    all_results = {}
    for f in result_files:
        tag = f.stem.replace("_results", "")
        with open(f) as fh:
            all_results[tag] = json.load(fh)

    # Summary table
    st.subheader("All Experiments")
    rows = []
    for tag, r in all_results.items():
        if "best_acc" in r:
            acc = r["best_acc"]
            exp_type = "Centralized"
        else:
            acc = r.get("final_test_acc", 0)
            exp_type = f"FL ({r.get('mode', '?')})"
        rows.append({
            "Tag": tag,
            "Type": exp_type,
            "Accuracy": f"{acc:.2%}" if acc else "N/A",
            "Params": f"{r.get('params', 0):,}",
            "Time": f"{r.get('total_time_s', 0) / 60:.1f}m",
        })
    st.table(rows)

    # Expandable details
    for tag, r in all_results.items():
        with st.expander(f"📄 {tag}"):
            st.json(r)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    page, device = sidebar()

    if "Skeleton Viewer" in page:
        page_skeleton_viewer(device)
    elif "Single Inference" in page:
        page_single_inference(device)
    elif "Model Comparison" in page:
        page_model_comparison(device)
    elif "Results Overview" in page:
        page_results_overview(device)


if __name__ == "__main__":
    main()
