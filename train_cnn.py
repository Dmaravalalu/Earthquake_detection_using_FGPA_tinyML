"""
Phase 2 — 1D-CNN Training with Quantization-Aware Training (QAT)
================================================================
Trains a low-latency 1D-CNN on the Phase-1 STEAD subset, applies QAT to
simulate 8-bit weights/activations during training, and exports a fully
int8-quantized .tflite ready for FPGA conversion (Artix-7 / ZedBoard).

Inputs
------
  out/X_train.npy   shape (N, 200, 3)  float16
  out/y_train.npy   shape (N,)         uint8   (1=earthquake, 0=noise)

Outputs
-------
  out/training_history.png   accuracy + loss curves
  out/confusion_matrix.png   test-set CM
  out/eew_cnn_int8.tflite    fully int8-quantized model
  out/eew_cnn_qat.keras      Keras QAT checkpoint (best val_loss)
  out/training_metrics.txt   final F1, accuracy, per-class report

Usage
-----
  python train_cnn.py --data-dir ./out --out-dir ./out
"""

import argparse
import json
import os
from pathlib import Path

# Must be set before any tensorflow import — tfmot 0.8 needs Keras 2 API.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW = 200
CHANNELS = 3
CONV_FILTERS = (16, 32, 64)   # FPGA-friendly powers of two
KERNEL_SIZE = 3
POOL_SIZE = 2
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 5
SEED = 42
F1_TARGET = 0.92

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── Data ─────────────────────────────────────────────────────────────────────
def load_data(data_dir: Path):
    """Load the Phase-1 arrays and split 80/10/10 (train/val/test), stratified."""
    X = np.load(data_dir / "X_train.npy", mmap_mode="r").astype(np.float32)
    y = np.load(data_dir / "y_train.npy")
    print(f"  Loaded X={X.shape} {X.dtype}, y={y.shape} {y.dtype}")
    print(f"  Class balance: {dict(zip(*np.unique(y, return_counts=True)))}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=1 / 9, stratify=y_temp, random_state=SEED)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_dataset(X, y, batch_size, shuffle):
    """tf.data pipeline. Caches in RAM (≈1 GB float32 for full set)."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(50_000, len(X)), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── Model ────────────────────────────────────────────────────────────────────
def build_base_model():
    """1D-CNN realised as Conv2D over a (time, 1, channels) tensor.

    tfmot's default quantize registry doesn't include Conv1D, so we reshape
    the (200, 3) input to (200, 1, 3) and use Conv2D with kernel (3, 1) —
    mathematically identical to Conv1D(kernel=3) over the time axis, fully
    quantizable, and FPGA toolchains (HLS4ML, Vitis AI) prefer Conv2D.

    BN sits between Conv and ReLU so QAT folds it into the preceding Conv.
    GAP keeps the parameter count tiny — critical for FPGA deployment.
    """
    inp = layers.Input(shape=(WINDOW, CHANNELS), name="waveform")
    x = layers.Reshape((WINDOW, 1, CHANNELS), name="to_2d")(inp)
    for i, f in enumerate(CONV_FILTERS):
        x = layers.Conv2D(f, (KERNEL_SIZE, 1), padding="same",
                          name=f"conv_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.ReLU(name=f"relu_{i+1}")(x)
        x = layers.MaxPooling2D((POOL_SIZE, 1), name=f"pool_{i+1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    out = layers.Dense(1, activation="sigmoid", name="prob_eq")(x)
    return models.Model(inp, out, name="eew_cnn")


def apply_qat(model):
    """Wrap the model so all Conv/Dense weights and activations are simulated
    as 8-bit during training. The TFLite converter later reads these
    learned fake-quant params to produce the real int8 model."""
    return tfmot.quantization.keras.quantize_model(model)


# ── Training ─────────────────────────────────────────────────────────────────
def train(model, train_ds, val_ds, out_dir: Path):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "eew_cnn_qat.keras"),
            monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    return history


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, out_dir: Path):
    print("\n  Predicting on test set …")
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1).ravel()
    y_pred = (y_prob >= 0.5).astype(np.uint8)

    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["noise", "earthquake"],
                                   digits=4)

    print("\n" + "=" * 60)
    print("  TEST-SET METRICS")
    print("=" * 60)
    print(report)
    print(f"  Confusion matrix:\n{cm}")
    print(f"  F1 score: {f1:.4f}  (target ≥ {F1_TARGET})")
    print("  ✅ PASS" if f1 >= F1_TARGET else "  ❌ BELOW TARGET — retune")
    print("=" * 60)

    (out_dir / "training_metrics.txt").write_text(
        f"F1: {f1:.4f}\nTarget: {F1_TARGET}\n\n{report}\n"
        f"Confusion matrix:\n{cm}\n")
    plot_confusion(cm, out_dir)
    return f1, cm


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_history(history, out_dir: Path):
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h["accuracy"], label="train")
    axes[0].plot(h["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(h["loss"], label="train")
    axes[1].plot(h["val_loss"], label="val")
    axes[1].set_title("Loss (BCE)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Phase 2 — 1D-CNN QAT Training History", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "training_history.png"
    fig.savefig(path, dpi=150)
    print(f"  History plot → {path}")


def plot_confusion(cm, out_dir: Path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["noise", "earthquake"])
    ax.set_yticklabels(["noise", "earthquake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Test-set Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    print(f"  Confusion-matrix plot → {path}")


# ── TFLite export ────────────────────────────────────────────────────────────
def export_tflite(qat_model, out_dir: Path, calib_data=None,
                  full_int8_io: bool = True):
    """Convert the QAT model to a fully int8-quantized .tflite.

    QAT bakes quantization params into the model for weights and activations,
    but the converter still needs a representative_dataset to calibrate the
    input-tensor scale/zero-point when full int8 IO is requested."""
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if full_int8_io:
        if calib_data is None:
            raise ValueError("full_int8_io requires calib_data for input "
                             "quantization calibration")

        def representative_dataset():
            for i in range(min(200, len(calib_data))):
                yield [calib_data[i:i+1].astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_bytes = converter.convert()
    path = out_dir / "eew_cnn_int8.tflite"
    path.write_bytes(tflite_bytes)
    size_kb = path.stat().st_size / 1024
    print(f"\n  TFLite int8 model → {path}  ({size_kb:.1f} KB)")

    # Quick sanity probe of input/output tensor specs
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    print(f"  Input  spec: {interp.get_input_details()[0]['shape']} "
          f"{interp.get_input_details()[0]['dtype']}")
    print(f"  Output spec: {interp.get_output_details()[0]['shape']} "
          f"{interp.get_output_details()[0]['dtype']}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"TF {tf.__version__}  |  tfmot {tfmot.__version__}")
    print(f"Devices: {[d.device_type for d in tf.config.list_physical_devices()]}\n")

    print("[1/6] Loading data …")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_dir)

    print("\n[2/6] Building base model …")
    base = build_base_model()
    base.summary()

    print("\n[3/6] Wrapping with QAT …")
    qat = apply_qat(base)

    print("\n[4/6] Training …")
    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   BATCH_SIZE, shuffle=False)
    history = train(qat, train_ds, val_ds, out_dir)
    plot_history(history, out_dir)

    print("\n[5/6] Evaluating on held-out test set …")
    f1, cm = evaluate(qat, X_test, y_test, out_dir)

    print("\n[6/6] Exporting int8 TFLite …")
    export_tflite(qat, out_dir, calib_data=X_train, full_int8_io=True)

    summary = {
        "tf_version": tf.__version__,
        "tfmot_version": tfmot.__version__,
        "test_f1": float(f1),
        "f1_target": F1_TARGET,
        "confusion_matrix": cm.tolist(),
    }
    (out_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary JSON → {out_dir / 'phase2_summary.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./out", type=Path)
    p.add_argument("--out-dir",  default="./out", type=Path)
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
