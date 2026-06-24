"""
Epicenter Side-Project — Multi-output Regression CNN (v2)
=========================================================
Trains a lightweight CNN to predict, from a 5-second 3-channel waveform
crop around a P-arrival:

    * source_distance_km   (regressed in log-space, z-normalised)
    * sin(back_azimuth)    (unit-circle target — avoids 0/360 wrap)
    * cos(back_azimuth)

v2 changes (vs first attempt that learned distance only weakly and
azimuth not at all):
    * **Wider trunk** — 32/64/128/128 filters with kernel 5×1 over four
      Conv-BN-ReLU-Pool blocks. ~130k params.
    * **Per-channel normalisation** of the waveform input so each
      channel is on the same dynamic range, plus a small **auxiliary
      input** carrying the original per-channel peak amplitudes — that
      preserves the cross-channel ratio that encodes back-azimuth.
    * **Deeper head** — Dense 128 → Dropout 0.2 → Dense 64 → three heads.

This model is NOT pushed to the FPGA; it's a separate software-only
side project. We keep Conv2D-on-reshape so it stays QAT-compatible if
the project ever needs an int8 export.

Inputs
------
  out/epicenter_data.npz   from prepare_epicenter_dataset.py

Outputs
-------
  out/epicenter_cnn.keras            best-val checkpoint (dual-input model)
  out/epicenter_norm.json            distance encoding params (mu, sigma)
  out/epicenter_metrics.json         test-set metrics
  out/epicenter_metrics.txt          human-readable summary
  out/epicenter_training_history.png loss curves
  out/epicenter_predictions.png      scatter + angular error plots

Usage
-----
  source .venv-tf/bin/activate
  TF_USE_LEGACY_KERAS=1 python train_epicenter.py --data ./out/epicenter_data.npz \
      --out-dir ./out
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from epicenter_utils import (
    angular_error_deg,
    decode_azimuth,
    decode_distance,
    destination_point,
    encode_azimuth,
    encode_distance,
    haversine_km,
    preprocess_window,
)

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW = 500
CHANNELS = 3
CONV_FILTERS = (32, 64, 128, 128)   # beefier trunk
KERNEL_SIZE = 5                     # wider temporal context per block
POOL_SIZE = 2
DENSE_UNITS = (128, 64)             # two-layer head with dropout
DROPOUT = 0.2
BATCH_SIZE = 256
EPOCHS = 80
LEARNING_RATE = 1e-3
PATIENCE = 10
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── Data loading ─────────────────────────────────────────────────────────────
def load_dataset(path: Path):
    """Load .npz produced by prepare_epicenter_dataset.py and split 80/10/10.
    The on-disk waveform is already globally peak-normalised — we re-normalise
    per channel here and keep the original per-channel peaks as side input."""
    npz = np.load(path, allow_pickle=True)
    X_raw = npz["X"].astype(np.float32)                  # (N, 500, 3) — global-peak-norm
    dist_km = npz["dist_km"].astype(np.float32)
    back_az = npz["back_az_deg"].astype(np.float32)
    recv_lat = npz["recv_lat"].astype(np.float32)
    recv_lon = npz["recv_lon"].astype(np.float32)
    src_lat = npz["src_lat"].astype(np.float32)
    src_lon = npz["src_lon"].astype(np.float32)
    trace_name = npz["trace_name"]

    # Per-channel renormalisation + extract per-channel peaks as aux input
    X, peaks = preprocess_window(X_raw)

    n = len(X)
    print(f"  Loaded {n:,} samples — X{X.shape} {X.dtype}, "
          f"channel_peaks{peaks.shape}")
    print(f"  Distance: {dist_km.min():.1f}–{dist_km.max():.1f} km  "
          f"(mean {dist_km.mean():.1f})")
    print(f"  Channel-peak ratios — mean: {peaks.mean(axis=0).round(3)}  "
          f"max-of-3 = {peaks.max(axis=1).mean():.3f}")

    idx = np.arange(n)
    idx_tr, idx_te = train_test_split(idx, test_size=0.10, random_state=SEED)
    idx_tr, idx_val = train_test_split(idx_tr, test_size=1 / 9, random_state=SEED)
    print(f"  Split — train {len(idx_tr):,}  val {len(idx_val):,}  test {len(idx_te):,}")

    return {
        "X": X, "peaks": peaks,
        "dist_km": dist_km, "back_az": back_az,
        "recv_lat": recv_lat, "recv_lon": recv_lon,
        "src_lat": src_lat, "src_lon": src_lon,
        "trace_name": trace_name,
        "idx_tr": idx_tr, "idx_val": idx_val, "idx_te": idx_te,
    }


def build_targets(d):
    """Convert raw metadata to model targets. Distance encoder is fit on the
    train split only so val/test never leak in."""
    dist_z_tr, mu, sigma = encode_distance(d["dist_km"][d["idx_tr"]])
    dist_z, _, _ = encode_distance(d["dist_km"], mu=mu, sigma=sigma)
    sin_az, cos_az = encode_azimuth(d["back_az"])
    return dist_z, sin_az, cos_az, mu, sigma


# ── Model ────────────────────────────────────────────────────────────────────
def build_model():
    """Dual-input Conv2D-over-reshape CNN.

    Trunk receives the per-channel-normalised waveform. The auxiliary input
    (3-element per-channel peaks) is concatenated after GAP so the dense
    head sees both the time-domain features AND the cross-channel
    amplitude ratios that carry polarisation / back-azimuth information.
    """
    inp_wave = layers.Input(shape=(WINDOW, CHANNELS), name="waveform")
    inp_aux  = layers.Input(shape=(CHANNELS,), name="channel_peaks")

    x = layers.Reshape((WINDOW, 1, CHANNELS), name="to_2d")(inp_wave)
    for i, f in enumerate(CONV_FILTERS):
        x = layers.Conv2D(f, (KERNEL_SIZE, 1), padding="same",
                          name=f"conv_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.ReLU(name=f"relu_{i+1}")(x)
        x = layers.MaxPooling2D((POOL_SIZE, 1), name=f"pool_{i+1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Mix the conv embedding with the per-channel peak side input
    h = layers.Concatenate(name="concat_aux")([x, inp_aux])
    for i, u in enumerate(DENSE_UNITS):
        h = layers.Dense(u, activation="relu", name=f"dense_{i+1}")(h)
        if i == 0 and DROPOUT > 0:
            h = layers.Dropout(DROPOUT, name=f"drop_{i+1}")(h)

    out_dist = layers.Dense(1, activation="linear", name="dist_z")(h)
    out_sin  = layers.Dense(1, activation="tanh",   name="sin_az")(h)
    out_cos  = layers.Dense(1, activation="tanh",   name="cos_az")(h)
    return models.Model([inp_wave, inp_aux],
                        [out_dist, out_sin, out_cos],
                        name="epicenter_cnn")


# ── Training ─────────────────────────────────────────────────────────────────
def train(model, d, dist_z, sin_az, cos_az, out_dir: Path):
    X, peaks = d["X"], d["peaks"]
    tr, val = d["idx_tr"], d["idx_val"]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss={"dist_z": "mse", "sin_az": "mse", "cos_az": "mse"},
        loss_weights={"dist_z": 1.0, "sin_az": 1.0, "cos_az": 1.0},
        metrics={"dist_z": "mae", "sin_az": "mae", "cos_az": "mae"},
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "epicenter_cnn.keras"),
            monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]

    history = model.fit(
        {"waveform": X[tr], "channel_peaks": peaks[tr]},
        {"dist_z": dist_z[tr], "sin_az": sin_az[tr], "cos_az": cos_az[tr]},
        validation_data=(
            {"waveform": X[val], "channel_peaks": peaks[val]},
            {"dist_z": dist_z[val], "sin_az": sin_az[val], "cos_az": cos_az[val]},
        ),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    return history


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(model, d, mu, sigma, out_dir: Path):
    te = d["idx_te"]
    Xte = d["X"][te]
    Pte = d["peaks"][te]

    pred = model.predict({"waveform": Xte, "channel_peaks": Pte},
                         batch_size=BATCH_SIZE, verbose=1)
    dist_z_pred = pred[0].ravel()
    sin_pred    = pred[1].ravel()
    cos_pred    = pred[2].ravel()

    dist_pred = decode_distance(dist_z_pred, mu, sigma)
    az_pred   = decode_azimuth(sin_pred, cos_pred)

    dist_true = d["dist_km"][te]
    az_true   = d["back_az"][te]

    dist_mae = float(np.mean(np.abs(dist_pred - dist_true)))
    dist_rmse = float(np.sqrt(np.mean((dist_pred - dist_true) ** 2)))
    az_err = angular_error_deg(az_pred, az_true)
    az_mae = float(np.mean(az_err))
    az_median = float(np.median(az_err))

    pred_lat, pred_lon = destination_point(
        d["recv_lat"][te], d["recv_lon"][te], az_pred, dist_pred)
    loc_err_km = haversine_km(pred_lat, pred_lon, d["src_lat"][te], d["src_lon"][te])
    loc_mae = float(np.mean(loc_err_km))
    loc_median = float(np.median(loc_err_km))

    print("\n" + "=" * 60)
    print("  TEST-SET METRICS  (n = {:,})".format(len(te)))
    print("=" * 60)
    print(f"  Distance MAE        : {dist_mae:8.2f} km")
    print(f"  Distance RMSE       : {dist_rmse:8.2f} km")
    print(f"  Back-az MAE         : {az_mae:8.2f}°  (median {az_median:.2f}°)")
    print(f"  Localization MAE    : {loc_mae:8.2f} km  (median {loc_median:.2f} km)")
    print("=" * 60)

    metrics = {
        "n_test": int(len(te)),
        "distance_mae_km": dist_mae,
        "distance_rmse_km": dist_rmse,
        "back_azimuth_mae_deg": az_mae,
        "back_azimuth_median_deg": az_median,
        "localization_mae_km": loc_mae,
        "localization_median_km": loc_median,
        "encoder_mu": float(mu),
        "encoder_sigma": float(sigma),
        "model_variant": "v2_wider_with_aux_peaks",
    }
    (out_dir / "epicenter_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "epicenter_metrics.txt").write_text(
        f"Epicenter regression test-set metrics ({len(te):,} samples) — v2\n"
        f"-------------------------------------------------\n"
        f"Distance MAE        : {dist_mae:8.2f} km\n"
        f"Distance RMSE       : {dist_rmse:8.2f} km\n"
        f"Back-az MAE         : {az_mae:8.2f}°\n"
        f"Back-az median err  : {az_median:8.2f}°\n"
        f"Localization MAE    : {loc_mae:8.2f} km\n"
        f"Localization median : {loc_median:8.2f} km\n"
    )

    plot_predictions(dist_true, dist_pred, az_true, az_pred, loc_err_km, out_dir)
    return metrics


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_history(history, out_dir: Path):
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(h["loss"], label="train")
    axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE (z-units)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    keys = [k for k in h if k.endswith("_mae") and not k.startswith("val_")]
    for k in keys:
        axes[1].plot(h[k], label=f"train {k.replace('_mae','')}")
        if f"val_{k}" in h:
            axes[1].plot(h[f"val_{k}"], "--", label=f"val {k.replace('_mae','')}")
    axes[1].set_title("Per-head MAE (z-units)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.suptitle("Epicenter CNN v2 — Training History", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "epicenter_training_history.png"
    fig.savefig(path, dpi=150)
    print(f"  History plot → {path}")


def plot_predictions(dist_true, dist_pred, az_true, az_pred, loc_err, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(dist_true, dist_pred, s=4, alpha=0.25, color="#2c3e50")
    lim = [max(1.0, dist_true.min()), dist_true.max()]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("True distance (km)"); axes[0].set_ylabel("Predicted (km)")
    axes[0].set_title("Distance"); axes[0].grid(alpha=0.3)

    axes[1].scatter(az_true, az_pred, s=4, alpha=0.25, color="#2980b9")
    axes[1].plot([0, 360], [0, 360], "r--", lw=1)
    axes[1].set_xlim(0, 360); axes[1].set_ylim(0, 360)
    axes[1].set_xlabel("True back-az (°)"); axes[1].set_ylabel("Predicted (°)")
    axes[1].set_title("Back-azimuth"); axes[1].grid(alpha=0.3)

    bins = np.linspace(0, np.quantile(loc_err, 0.99), 50)
    axes[2].hist(loc_err, bins=bins, color="#16a085", edgecolor="white", linewidth=0.4)
    axes[2].axvline(np.median(loc_err), color="black", linestyle="--",
                    label=f"median = {np.median(loc_err):.1f} km")
    axes[2].axvline(np.mean(loc_err), color="red", linestyle="--",
                    label=f"mean   = {np.mean(loc_err):.1f} km")
    axes[2].set_xlabel("Localization error (km)"); axes[2].set_ylabel("Count")
    axes[2].set_title("Haversine localization error")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    fig.suptitle("Epicenter CNN v2 — Test-set Predictions", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "epicenter_predictions.png"
    fig.savefig(path, dpi=150)
    print(f"  Predictions plot → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main(data_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"TF {tf.__version__}")
    print(f"Devices: {[d.device_type for d in tf.config.list_physical_devices()]}\n")

    print("[1/5] Loading data …")
    d = load_dataset(data_path)

    print("\n[2/5] Building targets …")
    dist_z, sin_az, cos_az, mu, sigma = build_targets(d)
    (out_dir / "epicenter_norm.json").write_text(
        json.dumps({"log_dist_mu": mu, "log_dist_sigma": sigma}, indent=2))
    print(f"  Distance encoder — mu={mu:.4f} sigma={sigma:.4f}")

    print("\n[3/5] Building model …")
    model = build_model()

    print("\n[4/5] Training …")
    history = train(model, d, dist_z, sin_az, cos_az, out_dir)
    plot_history(history, out_dir)

    print("\n[5/5] Evaluating …")
    metrics = evaluate(model, d, mu, sigma, out_dir)
    print(f"\n  Metrics JSON → {out_dir / 'epicenter_metrics.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./out/epicenter_data.npz", type=Path)
    p.add_argument("--out-dir", default="./out", type=Path)
    args = p.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data}.  "
                         "Run prepare_epicenter_dataset.py first.")
    main(args.data, args.out_dir)
