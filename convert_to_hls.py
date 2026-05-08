"""
Phase 3 — Convert trained Keras model to a Vivado HLS project
==============================================================
Reads the float weights produced by `extract_weights.py`, rebuilds the
1D-CNN architecture in pure Keras 3 (the format hls4ml expects), and
generates a synthesizable Vivado HLS C++ project for the ZedBoard
(Zynq-7020 / xc7z020clg484-1).

Pipeline
--------
  1. (subprocess) extract_weights.py runs in Keras-2 mode → .npz of weights
  2. This script (Keras-3 mode) rebuilds the architecture
  3. Loads weights into the rebuilt model
  4. hls4ml.convert_from_keras_model produces the Vivado HLS project tree

Outputs
-------
  out/hls_project/                 Vivado HLS project tree
    firmware/                       Generated HLS C++ for the model
    myproject_test.cpp              Top-level test bench
    project.tcl, build_prj.tcl      Vivado HLS scripts

Usage
-----
  source .venv-tf/bin/activate
  python convert_to_hls.py        # do NOT set TF_USE_LEGACY_KERAS=1
"""

import os
# Force pure Keras 3 mode — hls4ml 1.3 requires it
os.environ.pop("TF_USE_LEGACY_KERAS", None)
os.environ["TF_USE_LEGACY_KERAS"] = "0"

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
WEIGHTS_NPZ      = ROOT / "out" / "eew_cnn_float_weights.npz"
HLS_PROJECT_DIR  = ROOT / "out" / "hls_project"

# Architecture constants (must match train_cnn.py)
WINDOW = 200
CHANNELS = 3
CONV_FILTERS = (16, 32, 64)
KERNEL_SIZE = 3
POOL_SIZE = 2

# HLS target
TARGET_PART      = "xc7z020clg484-1"   # ZedBoard's Zynq-7020
TARGET_BOARD     = "ZedBoard"
CLOCK_PERIOD_NS  = 10                  # 100 MHz fabric clock

# Quantization config (per Phase 2 QAT settings)
DEFAULT_PRECISION = "fixed<16,6>"
WEIGHT_PRECISION  = "fixed<8,1>"
ACCUM_PRECISION   = "fixed<20,10>"


# ── Step 1: ensure weights file exists (run extract_weights.py if not) ────────
def ensure_weights():
    if WEIGHTS_NPZ.exists():
        print(f"[1/4] Weights file present → {WEIGHTS_NPZ}")
        return
    print("[1/4] Weights file missing — running extract_weights.py …")
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    res = subprocess.run([sys.executable, str(ROOT / "extract_weights.py")],
                         env=env)
    if res.returncode != 0:
        raise SystemExit("    ❌ extract_weights.py failed.")
    print(f"    Extracted weights → {WEIGHTS_NPZ}")


# ── Step 2: rebuild architecture in pure Keras 3 ─────────────────────────────
def build_keras3_model():
    print("[2/4] Building float32 model in Keras 3 …")
    import tensorflow as tf
    from tensorflow.keras import layers, models

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
    model = models.Model(inp, out, name="eew_cnn")
    model.summary(print_fn=lambda s: print("      " + s))
    return model


def load_weights_into(model):
    print(f"      Loading weights from {WEIGHTS_NPZ}")
    payload = np.load(WEIGHTS_NPZ)
    loaded = 0
    for layer in model.layers:
        keys = [k for k in payload.files if k.startswith(layer.name + "/")]
        if not keys:
            continue
        keys.sort(key=lambda k: int(k.split("/")[-1]))
        weights = [payload[k] for k in keys]
        layer.set_weights(weights)
        loaded += 1
    print(f"      Loaded weights for {loaded}/{len(model.layers)} layers")

    # Sanity check
    rng = np.random.default_rng(42)
    x = rng.standard_normal((5, 200, 3)).astype(np.float32)
    y = model.predict(x, verbose=0).ravel()
    print(f"      Sanity output range: [{y.min():.4f}, {y.max():.4f}]  "
          f"(expect within [0, 1])")
    if y.min() < 0 or y.max() > 1.0001:
        raise SystemExit("      ❌ outputs outside sigmoid range; weight load failed")


# ── Step 3: build hls4ml config ──────────────────────────────────────────────
def build_hls_config(model):
    print("[3/4] Building hls4ml config …")
    import hls4ml, tensorflow as tf

    config = hls4ml.utils.config_from_keras_model(
        model,
        granularity="name",
        default_precision=DEFAULT_PRECISION,
        default_reuse_factor=1,
    )

    config["Model"]["Strategy"] = "Latency"
    config["Model"]["Precision"] = DEFAULT_PRECISION

    for layer in model.layers:
        name = layer.name
        if name not in config["LayerName"]:
            continue
        cfg = config["LayerName"][name]
        if isinstance(layer, tf.keras.layers.Conv2D):
            cfg["Precision"] = {
                "weight": WEIGHT_PRECISION,
                "bias":   WEIGHT_PRECISION,
                "result": DEFAULT_PRECISION,
                "accum":  ACCUM_PRECISION,
            }
            cfg["ReuseFactor"] = 1
        elif isinstance(layer, tf.keras.layers.Dense):
            cfg["Precision"] = {
                "weight": WEIGHT_PRECISION,
                "bias":   WEIGHT_PRECISION,
                "result": DEFAULT_PRECISION,
                "accum":  ACCUM_PRECISION,
            }
            cfg["ReuseFactor"] = 1
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            cfg["Precision"] = {
                "scale":  WEIGHT_PRECISION,
                "bias":   WEIGHT_PRECISION,
                "result": DEFAULT_PRECISION,
            }
    return config


# ── Step 4: generate Vivado HLS project ──────────────────────────────────────
def generate_hls_project(model, config):
    print("[4/4] Generating Vivado HLS project tree …")
    import hls4ml

    if HLS_PROJECT_DIR.exists():
        shutil.rmtree(HLS_PROJECT_DIR)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(HLS_PROJECT_DIR),
        project_name="eew_cnn",
        backend="Vivado",
        part=TARGET_PART,
        clock_period=CLOCK_PERIOD_NS,
        io_type="io_parallel",
    )
    hls_model.write()

    n_files = sum(1 for _ in HLS_PROJECT_DIR.rglob("*"))
    print(f"\n  HLS project tree → {HLS_PROJECT_DIR}/  ({n_files} files)")
    print(f"  Target part      : {TARGET_PART}  ({TARGET_BOARD})")
    print(f"  Clock period     : {CLOCK_PERIOD_NS} ns ({1000//CLOCK_PERIOD_NS} MHz)")
    print(f"  IO type          : io_parallel")
    print(f"  Strategy         : Latency")
    print(f"  Weight precision : {WEIGHT_PRECISION}")
    print(f"  Accum precision  : {ACCUM_PRECISION}")
    print()
    print("  Next steps (must run on a Linux/Windows host with Vivado HLS):")
    print(f"     cd {HLS_PROJECT_DIR}")
    print( "     vivado_hls -f build_prj.tcl  \"csim=1 synth=1 cosim=1 export=1\"")
    print( "     # or open Vivado HLS GUI and import the project directory")
    print()
    print("  See phase3.md for the full Vivado/ZedBoard deployment guide.")


def main():
    ensure_weights()
    model = build_keras3_model()
    load_weights_into(model)
    config = build_hls_config(model)
    generate_hls_project(model, config)
    print("\n  Phase 3 conversion complete.")


if __name__ == "__main__":
    main()
