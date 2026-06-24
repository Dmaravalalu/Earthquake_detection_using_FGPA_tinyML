"""
Phase 2.5 — Interactive CPU Demo
================================
A Streamlit web app that proves the trained int8 TFLite model works on real
STEAD traces before we burn it into the FPGA.

Two tabs:
  1. Architecture — visual flow diagram + per-layer table + size/latency stats.
  2. Live Demo    — pick a trace, see waveform + sliding-window inference,
                    predicted vs actual P-arrival, detection latency.

Inference uses the *same* int8 TFLite model that goes to the FPGA, so this
demo is the definitive "does the model work" check.

Run
---
  source .venv-tf/bin/activate
  TF_USE_LEGACY_KERAS=1 streamlit run app.py
"""

import json
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from pathlib import Path

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from train_cnn import build_base_model, apply_qat
from epicenter_utils import (
    decode_azimuth,
    decode_distance,
    destination_point,
    haversine_km,
    preprocess_window,
)

# ── Constants ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
TFLITE_PATH = ROOT / "out" / "eew_cnn_int8.tflite"
EPI_MODEL_PATH = ROOT / "out" / "epicenter_cnn.keras"
EPI_NORM_PATH  = ROOT / "out" / "epicenter_norm.json"
EPI_METRICS_PATH = ROOT / "out" / "epicenter_metrics.json"
CSV_PATH    = Path.home() / "Downloads" / "archive" / "merge.csv"
HDF5_PATH   = Path.home() / "Downloads" / "archive" / "merge.hdf5"

WINDOW = 200
PRE_ARRIVAL = 50
SAMPLE_RATE = 100   # Hz
TRACE_LEN_S = 60    # 6000 samples / 100 Hz

# Epicenter side-project — must match prepare_epicenter_dataset.py
EPI_WINDOW = 500
EPI_PRE_ARRIVAL = 100

CHANNEL_NAMES = ("Vertical (Z)", "North-South (N)", "East-West (E)")
CHANNEL_COLORS = ("#c0392b", "#27ae60", "#2980b9")

st.set_page_config(page_title="EEW CNN Demo",
                   page_icon="〰",
                   layout="wide")


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource
def load_interpreter():
    interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    return interp


@st.cache_resource(show_spinner="Loading Keras QAT model for activations …")
def load_keras_model():
    """Rebuild architecture + load weights (the standard tfmot workaround
    for the .keras save/load round-trip bug)."""
    base = build_base_model()
    qat = apply_qat(base)
    qat.compile(optimizer="adam", loss="binary_crossentropy")
    qat.load_weights(str(ROOT / "out" / "eew_cnn_qat.keras"))
    return qat


@st.cache_resource(show_spinner="Building activation extractor …")
def build_activation_extractor():
    """Returns a tf.keras.Model that yields the post-ReLU output of each
    Conv block plus the GAP output, given a (1, 200, 3) input."""
    qat = load_keras_model()
    # tfmot wraps each layer; original layer names survive as substrings.
    wanted = ("conv_1", "conv_2", "conv_3", "gap")
    outs = []
    for w in wanted:
        # find the post-ReLU output for conv blocks; the GAP layer itself for gap
        layer_name = w if w == "gap" else f"relu_{w[-1]}"
        for layer in qat.layers:
            if layer_name in layer.name:
                outs.append(layer.output)
                break
    return tf.keras.Model(inputs=qat.inputs, outputs=outs)


@st.cache_data(show_spinner="Loading STEAD catalogue …")
def load_catalogue():
    df = pd.read_csv(CSV_PATH, low_memory=False,
                     usecols=["trace_name", "trace_category",
                              "p_arrival_sample", "s_arrival_sample",
                              "source_magnitude", "source_magnitude_type",
                              "source_depth_km", "source_distance_km",
                              "back_azimuth_deg",
                              "receiver_latitude", "receiver_longitude",
                              "source_latitude", "source_longitude",
                              "snr_db"])
    df["snr_max"] = df["snr_db"].map(_parse_snr_max)
    return df


@st.cache_resource(show_spinner="Loading epicenter model …")
def load_epicenter_model():
    """Loads the side-project epicenter regression model + its distance
    encoder params. Returns (model, mu, sigma) or (None, None, None) if
    the model hasn't been trained yet."""
    if not EPI_MODEL_PATH.exists() or not EPI_NORM_PATH.exists():
        return None, None, None
    model = tf.keras.models.load_model(str(EPI_MODEL_PATH), compile=False)
    norm = json.loads(EPI_NORM_PATH.read_text())
    return model, float(norm["log_dist_mu"]), float(norm["log_dist_sigma"])


def _parse_snr_max(s):
    if not isinstance(s, str):
        return np.nan
    try:
        vals = [float(x) for x in s.strip("[]").replace(",", " ").split()]
        return max(vals) if vals else np.nan
    except Exception:
        return np.nan


def load_waveform(trace_name):
    with h5py.File(HDF5_PATH, "r") as hf:
        return np.array(hf[f"data/{trace_name}"], dtype=np.float32)


# ── Inference ────────────────────────────────────────────────────────────────
def predict_along_trace(waveform, interpreter, stride=5):
    """Slide a 200-sample window across the full trace, return (window_starts,
    probabilities). Uses the same per-trace peak normalization as training."""
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    n = waveform.shape[0]
    starts = np.arange(0, n - WINDOW + 1, stride)
    probs = np.zeros(len(starts), dtype=np.float32)

    for i, s in enumerate(starts):
        w = waveform[s:s + WINDOW].astype(np.float32)
        peak = np.max(np.abs(w))
        if peak > 0:
            w = w / peak
        q = np.clip(w / in_scale + in_zp, -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], q[None])
        interpreter.invoke()
        probs[i] = (interpreter.get_tensor(out["index"])[0, 0] - out_zp) * out_scale

    return starts, probs


def find_first_detection(starts, probs, threshold=0.5,
                         sustain_samples=3):
    """Earliest window where probability stays ≥ threshold for `sustain_samples`
    consecutive windows. Returns (start_sample, predicted_p_sample) or
    (None, None) if no detection."""
    above = probs >= threshold
    for i in range(len(above) - sustain_samples + 1):
        if above[i:i + sustain_samples].all():
            return int(starts[i]), int(starts[i] + PRE_ARRIVAL)
    return None, None


# ── UI: Sidebar (trace picker) ────────────────────────────────────────────────
def sidebar_picker(df):
    st.sidebar.header("Pick a trace")
    mode = st.sidebar.radio("How to pick?",
                            ["Random earthquake", "Random noise",
                             "By trace name"])

    if mode == "Random earthquake":
        min_snr = st.sidebar.slider("Min SNR (dB)", 0, 80, 30)
        col_a, col_b = st.sidebar.columns(2)
        min_mag = col_a.number_input("Min mag", 0.0, 9.0, 3.0, 0.1)
        max_mag = col_b.number_input("Max mag", 0.0, 9.0, 5.0, 0.1)
        seed = st.sidebar.number_input("Random seed", 0, 999_999, 42)

        mask = ((df["trace_category"] == "earthquake_local")
                & df["snr_max"].notna() & (df["snr_max"] >= min_snr)
                & df["source_magnitude"].notna()
                & (df["source_magnitude"].astype(float) >= min_mag)
                & (df["source_magnitude"].astype(float) <= max_mag)
                & df["p_arrival_sample"].notna())
        candidates = df[mask]
        st.sidebar.caption(f"{len(candidates):,} traces match")
        if len(candidates) == 0:
            return None
        return candidates.sample(n=1, random_state=int(seed)).iloc[0]

    if mode == "Random noise":
        seed = st.sidebar.number_input("Random seed", 0, 999_999, 42)
        candidates = df[df["trace_category"] == "noise"]
        return candidates.sample(n=1, random_state=int(seed)).iloc[0]

    name = st.sidebar.text_input("trace_name",
                                 value="PFVI.PM_20130131164841_EV")
    rows = df[df["trace_name"] == name]
    if len(rows) == 0:
        st.sidebar.error("No such trace_name")
        return None
    return rows.iloc[0]


# ── UI: Architecture tab ──────────────────────────────────────────────────────
def render_architecture():
    st.subheader("Network architecture")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trainable params", "8,225")
    c2.metric("Model size (int8)",
              f"{TFLITE_PATH.stat().st_size / 1024:.1f} KB"
              if TFLITE_PATH.exists() else "—")
    c3.metric("Input shape", "(200, 3)")
    c4.metric("Output", "P(earthquake)")

    st.markdown("**Layer flow** — input flows left to right.")
    st.pyplot(_arch_diagram())

    st.markdown("**Per-layer detail**")
    layers = [
        ("waveform", "Input",                "(200, 3)",         0),
        ("to_2d",    "Reshape",              "(200, 1, 3)",      0),
        ("conv_1",   "Conv2D 16, kernel 3×1", "(200, 1, 16)",   160),
        ("bn_1",     "BatchNorm",            "(200, 1, 16)",     64),
        ("relu_1",   "ReLU",                 "(200, 1, 16)",      0),
        ("pool_1",   "MaxPool 2×1",          "(100, 1, 16)",      0),
        ("conv_2",   "Conv2D 32, kernel 3×1","(100, 1, 32)",  1_568),
        ("bn_2",     "BatchNorm",            "(100, 1, 32)",    128),
        ("relu_2",   "ReLU",                 "(100, 1, 32)",      0),
        ("pool_2",   "MaxPool 2×1",          "(50, 1, 32)",       0),
        ("conv_3",   "Conv2D 64, kernel 3×1","(50, 1, 64)",   6_208),
        ("bn_3",     "BatchNorm",            "(50, 1, 64)",     256),
        ("relu_3",   "ReLU",                 "(50, 1, 64)",       0),
        ("pool_3",   "MaxPool 2×1",          "(25, 1, 64)",       0),
        ("gap",      "GlobalAvgPool2D",      "(64,)",             0),
        ("prob_eq",  "Dense 1 + sigmoid",    "(1,)",             65),
    ]
    df = pd.DataFrame(layers,
                      columns=["Name", "Type", "Output shape", "Params"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _arch_diagram():
    """Custom flow diagram drawn with matplotlib — 4 stages of conv, then GAP+Dense."""
    blocks = [
        ("Input\n(200, 3)",        "#ecf0f1"),
        ("Reshape\n(200,1,3)",     "#ecf0f1"),
        ("Conv2D-16\nBN + ReLU\nMaxPool ↓2", "#3498db"),
        ("Conv2D-32\nBN + ReLU\nMaxPool ↓2", "#2ecc71"),
        ("Conv2D-64\nBN + ReLU\nMaxPool ↓2", "#9b59b6"),
        ("GAP\n(64,)",             "#f39c12"),
        ("Dense 1\nsigmoid",       "#e74c3c"),
        ("P(quake)",               "#ecf0f1"),
    ]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.set_xlim(0, len(blocks) * 1.3)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, (label, color) in enumerate(blocks):
        x = i * 1.3 + 0.1
        rect = mpatches.FancyBboxPatch(
            (x, 0.4), 1.0, 1.2,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#2c3e50", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.5, 1.0, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white"
                if color not in ("#ecf0f1", "#f39c12") else "#2c3e50")
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x + 1.25, 1.0), xytext=(x + 1.05, 1.0),
                        arrowprops=dict(arrowstyle="->", lw=1.2,
                                        color="#2c3e50"))
    return fig


# ── UI: Live demo tab ─────────────────────────────────────────────────────────
def render_live_demo(row, interpreter):
    if row is None:
        st.info("Pick a trace in the sidebar to start.")
        return

    cat = row["trace_category"]
    p_actual = row.get("p_arrival_sample")
    s_actual = row.get("s_arrival_sample")
    p_actual = int(p_actual) if pd.notna(p_actual) else None
    s_actual = int(s_actual) if pd.notna(s_actual) else None

    # Trace metadata header
    st.subheader("Trace metadata")
    cols = st.columns(6)
    cols[0].metric("trace_name", row["trace_name"])
    cols[1].metric("category", cat)
    if cat == "earthquake_local":
        cols[2].metric("magnitude",
                       f"{row['source_magnitude']:.1f} {row.get('source_magnitude_type','') or ''}".strip())
        cols[3].metric("depth (km)", f"{row['source_depth_km']:.1f}")
        cols[4].metric("distance (km)", f"{row['source_distance_km']:.1f}")
        cols[5].metric("SNR_max (dB)", f"{row['snr_max']:.1f}")

    # Load + run sliding-window inference
    waveform = load_waveform(row["trace_name"])
    stride = st.slider("Sliding-window stride (samples)", 1, 50, 5,
                       help="Smaller = more precise predicted P-arrival, "
                            "but more inference calls. 5 = 50 ms resolution.")
    threshold = st.slider("Detection threshold", 0.05, 0.95, 0.50, 0.05)

    with st.spinner("Running sliding-window inference …"):
        starts, probs = predict_along_trace(waveform, interpreter, stride=stride)
    p_det_start, p_det_arrival = find_first_detection(
        starts, probs, threshold=threshold)

    # Plots
    fig = _waveform_and_probability_plot(
        waveform, probs, starts, p_actual, s_actual,
        p_det_arrival, threshold)
    st.pyplot(fig)

    # Prediction summary
    st.subheader("Prediction summary")
    c1, c2, c3, c4 = st.columns(4)
    if p_actual is not None:
        c1.metric("Actual P-arrival",
                  f"{p_actual / SAMPLE_RATE:.2f} s",
                  help=f"Sample {p_actual}")
    else:
        c1.metric("Actual P-arrival", "—")

    if p_det_arrival is not None:
        c2.metric("Predicted P-arrival",
                  f"{p_det_arrival / SAMPLE_RATE:.2f} s",
                  help=f"Sample {p_det_arrival}")
    else:
        c2.metric("Predicted P-arrival", "no detection")

    if p_actual is not None and p_det_arrival is not None:
        latency_ms = (p_det_arrival - p_actual) * 1000 / SAMPLE_RATE
        c3.metric("Detection latency",
                  f"{latency_ms:+.0f} ms",
                  help="Negative = early (false alarm before P), "
                       "positive = late (model fired after P arrived)")
    else:
        c3.metric("Detection latency", "—")
    c4.metric("Peak probability", f"{probs.max():.3f}")

    if cat == "earthquake_local":
        if p_det_arrival is not None:
            st.success(f"✅ Model detected the earthquake "
                       f"({len(probs)} inferences run, peak prob {probs.max():.3f}).")
        else:
            st.error("❌ Model failed to detect the earthquake — "
                     "try lowering the threshold.")
    else:
        if p_det_arrival is None:
            st.success(f"✅ Model correctly stayed silent on noise "
                       f"(peak prob {probs.max():.3f}).")
        else:
            st.warning(f"⚠ Model fired a **false alarm** on noise at "
                       f"t = {p_det_arrival / SAMPLE_RATE:.2f} s "
                       f"(peak prob {probs.max():.3f}).")

    # ── Per-layer activation visualization ───────────────────────────────────
    st.subheader("Per-layer activations")
    st.caption("What each Conv block outputs for the window the model used to "
               "make its top prediction. Each row of a heatmap is one filter; "
               "x-axis is time within the 200-sample window. Brighter = more "
               "activation. GAP is shown as a 1-D bar of 64 channel means.")

    # Use the window with the max probability (the model's most confident slice)
    if len(probs) > 0:
        best = int(np.argmax(probs))
        win_start = int(starts[best])
    else:
        win_start = 0
    window = waveform[win_start:win_start + WINDOW].astype(np.float32)
    peak = np.max(np.abs(window))
    if peak > 0:
        window = window / peak

    extractor = build_activation_extractor()
    acts = extractor(window[None])
    acts = [a.numpy()[0] for a in acts]   # strip batch dim

    fig_act = _activation_grid(acts, win_start, peak_prob=probs.max())
    st.pyplot(fig_act)


def _activation_grid(acts, win_start, peak_prob):
    """Plot conv1, conv2, conv3 as heatmaps and GAP as a bar."""
    titles = [f"conv_1 → ReLU  (200 × 16)",
              f"conv_2 → ReLU  (100 × 32)",
              f"conv_3 → ReLU  (50 × 64)",
              f"GAP  (64,)"]
    fig, axes = plt.subplots(4, 1, figsize=(13, 8.5),
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.5]})
    for i in range(3):
        a = acts[i]
        if a.ndim == 3:        # (T, 1, F) → squeeze the singleton
            a = a[:, 0, :]
        a = a.T                # (F, T) so filters are rows
        im = axes[i].imshow(a, aspect="auto", cmap="magma",
                            interpolation="nearest")
        axes[i].set_title(titles[i], fontsize=9)
        axes[i].set_ylabel("filter")
        fig.colorbar(im, ax=axes[i], fraction=0.022, pad=0.01)
    axes[2].set_xlabel("time step inside window")

    # GAP — 1D bar
    gap = acts[3]
    axes[3].bar(np.arange(len(gap)), gap, color="#f39c12", edgecolor="#7f8c8d",
                linewidth=0.3)
    axes[3].set_title(titles[3], fontsize=9)
    axes[3].set_xlabel("channel index")
    axes[3].set_ylabel("mean act.")
    axes[3].set_xlim(-0.5, len(gap) - 0.5)

    fig.suptitle(f"Activations for window starting at sample {win_start} "
                 f"(t = {win_start/SAMPLE_RATE:.2f} s, peak prob {peak_prob:.3f})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


# ── UI: Epicenter tab (side-project, software-only — not on FPGA) ───────────
def render_epicenter(row):
    st.subheader("Epicenter regression — software-only side project")
    st.caption(
        "An independent second model that estimates **source distance** and "
        "**back-azimuth** from a 5-second 3-channel crop around the P-arrival, "
        "then uses spherical geometry (destination-point formula) to place the "
        "epicenter on the map. Localization error is reported as a Haversine "
        "distance to the catalogued source. **This model is not deployed on the "
        "FPGA** — it's a separate Keras model, intended as a demo only."
    )
    st.warning(
        "**Known limitation — distance is moderately predictive, back-azimuth "
        "is essentially random.** Test-set MAE ≈ 34 km on distance but ≈ 85° "
        "on back-azimuth (uniform random baseline ≈ 90°). Single-station "
        "back-azimuth from raw waveforms has a fundamental ±180° ambiguity "
        "without first-motion polarity processing — see "
        "`epicenter_future_work.md` for the proposed redesign (band-pass "
        "filtering, longer window, polarization features). The map dot will "
        "land at roughly the right distance from the receiver but in a "
        "near-random direction.",
        icon="⚠️",
    )

    model, mu, sigma = load_epicenter_model()
    if model is None:
        st.error(
            "Epicenter model not found. Train it first:\n\n"
            "```bash\n"
            "python prepare_epicenter_dataset.py "
            "--csv  ~/Downloads/archive/merge.csv "
            "--hdf5 ~/Downloads/archive/merge.hdf5 --out ./out\n"
            "TF_USE_LEGACY_KERAS=1 python train_epicenter.py "
            "--data ./out/epicenter_data.npz --out-dir ./out\n"
            "```"
        )
        return

    if row is None or row["trace_category"] != "earthquake_local":
        st.info("Pick a **random earthquake** in the sidebar to run the "
                "epicenter model — it only operates on earthquake traces.")
        return

    needed = ("source_distance_km", "back_azimuth_deg",
              "receiver_latitude", "receiver_longitude",
              "source_latitude", "source_longitude",
              "p_arrival_sample")
    missing = [k for k in needed if pd.isna(row.get(k))]
    if missing:
        st.warning(f"This trace is missing metadata for: {', '.join(missing)}. "
                   "Pick another earthquake.")
        return

    # ── Run inference on a 500-sample crop around P ─────────────────────────
    waveform = load_waveform(row["trace_name"])
    p_actual = int(row["p_arrival_sample"])
    start = p_actual - EPI_PRE_ARRIVAL
    end = start + EPI_WINDOW
    if start < 0 or end > waveform.shape[0]:
        st.warning("P-arrival too close to the trace edge — "
                   "can't take the 5-second epicenter window.")
        return

    raw_window = waveform[start:end, :].astype(np.float32)
    # Match training: global peak norm first (as on-disk dataset is stored),
    # then per-channel renorm with peaks kept as the aux input.
    global_peak = float(np.max(np.abs(raw_window)))
    if global_peak > 0:
        raw_window = raw_window / global_peak
    window, channel_peaks = preprocess_window(raw_window)

    with st.spinner("Running epicenter regression …"):
        pred = model.predict(
            {"waveform": window[None], "channel_peaks": channel_peaks[None]},
            verbose=0,
        )
        dist_z_pred = float(np.asarray(pred[0]).ravel()[0])
        sin_pred    = float(np.asarray(pred[1]).ravel()[0])
        cos_pred    = float(np.asarray(pred[2]).ravel()[0])

    dist_pred = float(decode_distance(np.array([dist_z_pred]), mu, sigma)[0])
    az_pred   = float(decode_azimuth(np.array([sin_pred]), np.array([cos_pred]))[0])

    # ── Ground truth ────────────────────────────────────────────────────────
    dist_true = float(row["source_distance_km"])
    az_true   = float(row["back_azimuth_deg"])
    recv_lat  = float(row["receiver_latitude"])
    recv_lon  = float(row["receiver_longitude"])
    src_lat   = float(row["source_latitude"])
    src_lon   = float(row["source_longitude"])

    # ── Spherical geometry → predicted epicenter coordinates ────────────────
    pred_lat, pred_lon = destination_point(recv_lat, recv_lon, az_pred, dist_pred)
    pred_lat = float(pred_lat); pred_lon = float(pred_lon)
    loc_err_km = float(haversine_km(pred_lat, pred_lon, src_lat, src_lon))

    # ── Metrics ─────────────────────────────────────────────────────────────
    cols = st.columns(4)
    cols[0].metric("Distance — true", f"{dist_true:.1f} km")
    cols[1].metric("Distance — predicted", f"{dist_pred:.1f} km",
                   delta=f"{dist_pred - dist_true:+.1f} km")
    cols[2].metric("Back-az — true", f"{az_true:.1f}°")
    az_diff = ((az_pred - az_true + 180) % 360) - 180
    cols[3].metric("Back-az — predicted", f"{az_pred:.1f}°",
                   delta=f"{az_diff:+.1f}°")

    st.metric("Localization error (Haversine)", f"{loc_err_km:.1f} km",
              help="Great-circle distance between catalogued source and "
                   "epicenter computed from the model's distance + back-azimuth.")

    # ── Map ─────────────────────────────────────────────────────────────────
    st.markdown("**Map** — receiver, true epicenter, predicted epicenter")
    map_df = pd.DataFrame([
        {"lat": recv_lat, "lon": recv_lon, "kind": "receiver",
         "color": [52, 152, 219], "size": 90},
        {"lat": src_lat,  "lon": src_lon,  "kind": "true epicenter",
         "color": [46, 204, 113], "size": 130},
        {"lat": pred_lat, "lon": pred_lon, "kind": "predicted epicenter",
         "color": [231, 76, 60], "size": 130},
    ])
    st.map(map_df, latitude="lat", longitude="lon",
           color="color", size="size", zoom=5)

    legend = " · ".join([
        "🔵 receiver",
        "🟢 true epicenter",
        "🔴 predicted epicenter",
    ])
    st.caption(legend)

    # ── Window plot ─────────────────────────────────────────────────────────
    st.markdown("**5-second window the model saw** "
                "(per-trace peak-normalized, P-arrival at t = 1.00 s)")
    fig, axes = plt.subplots(3, 1, figsize=(12, 4.5), sharex=True)
    t = np.arange(EPI_WINDOW) / SAMPLE_RATE
    for c, (ax, name, color) in enumerate(zip(axes, CHANNEL_NAMES, CHANNEL_COLORS)):
        ax.plot(t, window[:, c], color=color, lw=0.6)
        ax.axvline(EPI_PRE_ARRIVAL / SAMPLE_RATE, color="black",
                   linestyle="--", lw=0.9, label="P-arrival")
        ax.set_ylabel(name, fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Epicenter model input window", fontweight="bold", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)

    # ── Saved test-set metrics ──────────────────────────────────────────────
    if EPI_METRICS_PATH.exists():
        with st.expander("Saved test-set metrics from training"):
            m = json.loads(EPI_METRICS_PATH.read_text())
            mc = st.columns(4)
            mc[0].metric("n test", f"{m['n_test']:,}")
            mc[1].metric("Distance MAE", f"{m['distance_mae_km']:.1f} km")
            mc[2].metric("Back-az MAE", f"{m['back_azimuth_mae_deg']:.1f}°")
            mc[3].metric("Localization MAE",
                         f"{m['localization_mae_km']:.1f} km",
                         help=f"median {m['localization_median_km']:.1f} km")


# ── UI: Test-set stats tab ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Splitting Phase-1 dataset to recover test set …")
def get_test_split(seed=42, test_size=0.10):
    """Reproduce the exact 80/10/10 split from train_cnn.py."""
    X = np.load(ROOT / "out" / "X_train.npy", mmap_mode="r")
    y = np.load(ROOT / "out" / "y_train.npy")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    return X_test, y_test


def render_test_stats():
    st.subheader("Sampled test-set performance")
    st.caption("Re-runs int8 TFLite inference on a random sample of the "
               "held-out test set (49,714 windows total) so you can see live "
               "what the saved metrics actually mean. Same model, same "
               "calibration, same windows the saved 0.9819 F1 came from.")

    n = st.slider("Sample size", 100, 5_000, 500, step=100,
                  help="Larger = tighter F1 estimate, longer wait")
    seed = st.number_input("Sample seed", 0, 999_999, 7)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                          key="cm_thresh")

    if not st.button("Run sample evaluation", type="primary"):
        st.info("Pick a sample size and hit **Run sample evaluation**.")
        return

    X_test, y_test = get_test_split()
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(X_test), size=n, replace=False)
    X = X_test[idx].astype(np.float32)
    y = y_test[idx]

    interpreter = load_interpreter()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    progress = st.progress(0.0, text="Running inference …")
    probs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        q = np.clip(X[i] / in_scale + in_zp, -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], q[None])
        interpreter.invoke()
        probs[i] = (interpreter.get_tensor(out["index"])[0, 0] - out_zp) * out_scale
        if i % max(1, n // 20) == 0:
            progress.progress((i + 1) / n)
    progress.empty()

    pred = (probs >= threshold).astype(np.uint8)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    f1 = f1_score(y, pred, zero_division=0)
    acc = (pred == y).mean()
    if cm[0].sum():
        far = cm[0, 1] / cm[0].sum()    # false alarm rate
    else:
        far = 0.0
    if cm[1].sum():
        miss = cm[1, 0] / cm[1].sum()   # miss rate
    else:
        miss = 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1", f"{f1:.4f}")
    c2.metric("Accuracy", f"{acc:.4f}")
    c3.metric("False alarm rate", f"{far:.2%}",
              help="Fraction of noise wrongly flagged as earthquake")
    c4.metric("Miss rate", f"{miss:.2%}",
              help="Fraction of earthquakes the model missed")

    fig, (ax_cm, ax_hist) = plt.subplots(1, 2, figsize=(12, 4.2))
    im = ax_cm.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black",
                       fontsize=11)
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["noise", "earthquake"])
    ax_cm.set_yticklabels(["noise", "earthquake"])
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
    ax_cm.set_title(f"Confusion matrix (n={n})")

    bins = np.linspace(0, 1, 41)
    ax_hist.hist(probs[y == 0], bins=bins, alpha=0.65, label="noise",
                 color="#3498db", edgecolor="white", linewidth=0.4)
    ax_hist.hist(probs[y == 1], bins=bins, alpha=0.65, label="earthquake",
                 color="#e74c3c", edgecolor="white", linewidth=0.4)
    ax_hist.axvline(threshold, color="black", linestyle="--", lw=1,
                    label=f"threshold = {threshold:.2f}")
    ax_hist.set_xlabel("Predicted P(earthquake)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Output probability distribution")
    ax_hist.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)


def _waveform_and_probability_plot(waveform, probs, starts,
                                   p_actual, s_actual,
                                   p_det_arrival, threshold):
    n = waveform.shape[0]
    t_wave = np.arange(n) / SAMPLE_RATE
    t_prob = (starts + PRE_ARRIVAL) / SAMPLE_RATE   # plot probs at the
                                                    # P-arrival position the
                                                    # model expects in-window
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.4]})

    # Three channel waveforms
    for ch, (ax, name, color) in enumerate(zip(axes[:3], CHANNEL_NAMES,
                                               CHANNEL_COLORS)):
        ax.plot(t_wave, waveform[:, ch], color=color, lw=0.5)
        ax.set_ylabel(f"{name}\nampl.", fontsize=8)
        ax.grid(alpha=0.25)
        if p_actual is not None:
            ax.axvline(p_actual / SAMPLE_RATE, color="black",
                       linestyle="--", lw=1.0)
        if s_actual is not None:
            ax.axvline(s_actual / SAMPLE_RATE, color="black",
                       linestyle=":", lw=1.0)
        if p_det_arrival is not None:
            ax.axvline(p_det_arrival / SAMPLE_RATE, color="#e67e22",
                       linestyle="-", lw=1.4, alpha=0.85)

    # Probability curve
    ax = axes[3]
    ax.fill_between(t_prob, 0, probs, color="#34495e", alpha=0.18)
    ax.plot(t_prob, probs, color="#2c3e50", lw=1.2)
    ax.axhline(threshold, color="#7f8c8d", linestyle="--", lw=0.9,
               label=f"threshold = {threshold:.2f}")
    if p_actual is not None:
        ax.axvline(p_actual / SAMPLE_RATE, color="black", linestyle="--",
                   lw=1.0, label="actual P-arrival")
    if s_actual is not None:
        ax.axvline(s_actual / SAMPLE_RATE, color="black", linestyle=":",
                   lw=1.0, label="actual S-arrival")
    if p_det_arrival is not None:
        ax.axvline(p_det_arrival / SAMPLE_RATE, color="#e67e22",
                   linestyle="-", lw=1.4, alpha=0.85,
                   label="predicted P-arrival")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, TRACE_LEN_S)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(earthquake)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.suptitle("Waveform (top 3) and model probability over time (bottom)",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.title("〰  Earthquake Early-Warning CNN — CPU Demo")
    st.caption("Final pre-FPGA validation: the int8 TFLite model running on "
               "real STEAD traces. Same model, same int8 weights, same "
               "calibration as what we'll burn into the Artix-7.")

    if not TFLITE_PATH.exists():
        st.error(f"Model not found: {TFLITE_PATH}. Run `train_cnn.py` first.")
        st.stop()
    if not CSV_PATH.exists() or not HDF5_PATH.exists():
        st.error(f"STEAD data missing under {CSV_PATH.parent}.")
        st.stop()

    interpreter = load_interpreter()
    df = load_catalogue()
    row = sidebar_picker(df)

    tab_arch, tab_demo, tab_stats, tab_epi = st.tabs(
        ["Architecture", "Live demo", "Test-set stats", "Epicenter (side project)"])
    with tab_arch:
        render_architecture()
    with tab_demo:
        render_live_demo(row, interpreter)
    with tab_stats:
        render_test_stats()
    with tab_epi:
        render_epicenter(row)


if __name__ == "__main__":
    main()
