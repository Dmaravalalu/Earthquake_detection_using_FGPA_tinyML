"""
Phase 3 helper — Extract trained weights from the QAT Keras checkpoint
======================================================================
Runs in Keras-2 mode (TF_USE_LEGACY_KERAS=1) so it can deserialize the
tfmot-wrapped QAT checkpoint, copies the weights into a fresh float32
base model, and saves them as a plain numpy .npz keyed by layer name.

The companion script `convert_to_hls.py` runs in Keras-3 mode (no env
flag), rebuilds the same architecture, and loads these weights for
hls4ml conversion.

Output
------
  out/eew_cnn_float_weights.npz   — keys:  '<layer_name>/<weight_idx>'
                                    values: numpy arrays
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path
import numpy as np
import tensorflow as tf  # noqa: E402

from train_cnn import build_base_model, apply_qat   # both use Keras 2

ROOT = Path(__file__).parent
QAT_CKPT = ROOT / "out" / "eew_cnn_qat.keras"
OUT_NPZ  = ROOT / "out" / "eew_cnn_float_weights.npz"


def main():
    if not QAT_CKPT.exists():
        raise SystemExit(f"QAT checkpoint not found: {QAT_CKPT}. "
                         "Run train_cnn.py first.")

    print(f"  TF: {tf.__version__}")
    print("  Loading QAT model …")
    base = build_base_model()
    qat = apply_qat(base)
    qat.compile(optimizer="adam", loss="binary_crossentropy")
    qat.load_weights(str(QAT_CKPT))
    print(f"    QAT model: {qat.count_params():,} params")

    print("  Building clean float32 model and copying weights …")
    float_model = build_base_model()

    # Per-layer "real" weight name suffixes — what we want to copy. Anything
    # else in the QAT wrapper (post_activation/min, optimizer_step, etc.)
    # is fake-quant tracking state, not part of the model.
    REAL_NAMES = {
        "Conv2D": ["kernel", "bias"],
        "Dense":  ["kernel", "bias"],
        "BatchNormalization": ["gamma", "beta", "moving_mean", "moving_variance"],
    }

    copied = 0
    for fl in float_model.layers:
        cls = type(fl).__name__
        if cls not in REAL_NAMES:
            continue
        match = next((q for q in qat.layers
                      if q.name == f"quant_{fl.name}" or q.name == fl.name),
                     None)
        if match is None:
            print(f"    ⚠ no QAT match for {fl.name}")
            continue

        # Extract the named weights from the QAT wrapper's variable list
        wanted = REAL_NAMES[cls]
        found = {}
        for w in match.weights:
            # variable names look like 'quant_conv_1/conv_1/kernel:0' or
            # 'quant_conv_1/post_activation/min:0'. We match by suffix.
            short = w.name.rsplit("/", 1)[-1].split(":")[0]
            if short in wanted:
                found[short] = w.numpy()
        if len(found) != len(wanted):
            print(f"    ⚠ {fl.name}: found {list(found)}, expected {wanted}")
            continue
        fl.set_weights([found[name] for name in wanted])
        copied += 1
    print(f"    Copied weights for {copied} layers with weights")

    # Quick sanity probe
    rng = np.random.default_rng(42)
    x = rng.standard_normal((5, 200, 3)).astype(np.float32)
    y = float_model.predict(x, verbose=0).ravel()
    print(f"    Sanity output range: [{y.min():.4f}, {y.max():.4f}]")
    if y.min() < 0 or y.max() > 1:
        raise SystemExit("    ❌ float32 model outputs outside [0,1]; "
                         "weight copy is broken")

    # Dump weights to npz keyed by '<layer_name>/<idx>'
    payload = {}
    for layer in float_model.layers:
        for i, w in enumerate(layer.get_weights()):
            payload[f"{layer.name}/{i}"] = w
    np.savez(OUT_NPZ, **payload)
    print(f"  Saved {len(payload)} weight tensors → {OUT_NPZ}")


if __name__ == "__main__":
    main()
