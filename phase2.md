# Phase 2 — 1D-CNN Training with Quantization-Aware Training (QAT)

This document captures **what was done, what was decided, and what was discovered** during Phase 2 of the Real-Time Earthquake Early Warning System project. It is intended both as a human reference and as context for future Claude Code sessions. **Read `phase1.md` first** for the dataset preparation context.

---

## 1. Goal of Phase 2

Train a low-latency 1D-CNN on the Phase-1 STEAD subset, simulate 8-bit quantization during training (QAT), and export a fully int8-quantized `.tflite` ready for FPGA conversion (Artix-7 / ZedBoard).

| Aim | Constraint |
|---|---|
| Binary classifier (earthquake vs noise) on `(200, 3)` 2-second windows | Matches Phase-1 output |
| 3 Conv layers, kernel 3, ReLU, BatchNorm | FPGA-friendly small kernels |
| GlobalAveragePooling1D before Dense | Eliminates Dense head's parameter explosion |
| QAT via `tfmot.quantization.keras.quantize_model` | Simulates int8 weights/activations during training |
| Adam(lr=1e-3), Binary cross-entropy, EarlyStopping on `val_loss` | Standard recipe |
| F1 ≥ 0.92 on test set | Project requirement |
| Export as fully int8 `.tflite` for FPGA toolchain | Final deliverable |

**Result: F1 = 0.9819 on 49,714 test samples, model size 16.3 KB.**

---

## 2. Environment Setup

The Phase-1 venv uses **Python 3.14** which **TensorFlow does not support** (max is Python 3.13 with TF 2.20). Phase 2 needed a separate venv.

### Required combo (pinned)

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv ~/Desktop/idp/.venv-tf
source ~/Desktop/idp/.venv-tf/bin/activate
pip install \
    "tensorflow==2.16.2" \
    "tf_keras==2.16.0" \
    "tensorflow-metal==1.1.0" \
    "tensorflow-model-optimization==0.8.0" \
    "numpy<2" \
    scikit-learn matplotlib
```

### Why these specific pins (not the latest)

| Package | Version | Why this version |
|---|---|---|
| `tensorflow` | 2.16.2 | Last TF that ships Keras 2 by default; tfmot 0.8 uses Keras-2 APIs internally |
| `tf_keras` | 2.16.0 | Legacy Keras 2 package required when `TF_USE_LEGACY_KERAS=1` is set; must match TF version |
| `tensorflow-metal` | 1.1.0 | Metal plugin lags TF; 1.2.0 was built against TF 2.18 internals and **dlopens fail** with newer TF |
| `tensorflow-model-optimization` | 0.8.0 | Latest tfmot release; expects Keras 2 — incompatible with TF's default Keras 3 |
| `numpy` | <2 | TF 2.16 was released before numpy 2; pinning avoids `_ARRAY_API not found` errors |

### Always set `TF_USE_LEGACY_KERAS=1` before running

```bash
TF_USE_LEGACY_KERAS=1 python train_cnn.py …
```

The training script also sets it via `os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")` before `import tensorflow`. Without this, tfmot fails with `Keras cannot be imported`.

### Apple Silicon GPU (Metal) verification

```bash
source .venv-tf/bin/activate
TF_USE_LEGACY_KERAS=1 python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Expected output:
# Metal device set to: Apple M4
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If GPU isn't listed, check that `tensorflow-metal` is installed and that the TF version matches the metal plugin's compatibility window.

---

## 3. Model Architecture

**8,449 trainable parameters total — 33 KB Keras checkpoint, 16.3 KB int8 TFLite.**

```
waveform                  (200, 3)         — input
to_2d (Reshape)           (200, 1, 3)      — Conv2D-ready
conv_1 (Conv2D 16, 3×1)   (200, 1, 16)     160 params
bn_1                      (200, 1, 16)     64
relu_1                    (200, 1, 16)     0
pool_1 (MaxPool 2×1)      (100, 1, 16)     0
conv_2 (Conv2D 32, 3×1)   (100, 1, 32)     1,568
bn_2                      (100, 1, 32)     128
relu_2                    (100, 1, 32)     0
pool_2 (MaxPool 2×1)      (50, 1, 32)      0
conv_3 (Conv2D 64, 3×1)   (50, 1, 64)      6,208
bn_3                      (50, 1, 64)      256
relu_3                    (50, 1, 64)      0
pool_3 (MaxPool 2×1)      (25, 1, 64)      0
gap (GlobalAveragePool2D) (64,)            0
prob_eq (Dense 1, sigmoid)(1,)             65
                                          ─────
Total                                      8,449
```

### Key design decisions

1. **Conv2D over a `(200, 1, 3)` reshape — not Conv1D.** tfmot's default quantize registry **does not include `Conv1D`**. Reshaping to `(time, 1, channels)` and using `Conv2D(kernel=(3, 1))` is mathematically identical to `Conv1D(kernel=3)` over the time axis but is fully quantizable, and FPGA toolchains (HLS4ML, Vitis AI) prefer Conv2D.

2. **BatchNorm between Conv and ReLU** (not Conv → ReLU → BN). At export time, QAT folds BN into the preceding Conv's weight matrix — works only when BN sits immediately after Conv with no activation in between.

3. **GlobalAveragePooling2D before Dense.** A flatten + Dense head would explode the parameter count from ~8K to ~100K+. GAP gives one scalar per channel, so Dense only sees 64 inputs → 65 params (64 weights + 1 bias).

4. **Channel widths 16 / 32 / 64** are powers of two (FPGA-friendly), chosen to give ~8K total params — enough capacity for the binary task without overfitting on 397k training samples.

5. **3 pool stages → 25 timesteps at GAP**. Each MaxPool halves the time axis: 200 → 100 → 50 → 25. Receptive field after 3 conv-pool blocks ≈ 36 timesteps (~0.36 s) — enough to characterize a P-wave first break.

---

## 4. Quantization-Aware Training (QAT)

```python
import tensorflow_model_optimization as tfmot
qat_model = tfmot.quantization.keras.quantize_model(base_model)
```

This wraps every Conv/Dense/BN/ReLU with **fake-quant ops** that simulate 8-bit rounding during forward passes while keeping float32 weights for the gradient. The fake-quant `min`/`max` parameters are learned as part of training, so the eventual int8 model has **calibrated activation ranges** baked in — no separate post-training calibration needed for the *interior* of the network.

QAT-specific overhead: the wrapped model has **8,703 params** vs the base's 8,449 — the extra 254 are the per-layer min/max + step-counter variables.

### Training recipe

```
optimizer        Adam(lr=1e-3)
loss             binary_crossentropy
metrics          accuracy, precision, recall
batch_size       512
max_epochs       50
EarlyStopping    monitor=val_loss, patience=5, restore_best_weights=True
ReduceLROnPlateau monitor=val_loss, factor=0.5, patience=2, min_lr=1e-5
data split       80 / 10 / 10 (train / val / test), stratified by class
```

Train and validation curves stay overlapping → **no overfitting** despite training to convergence (33 epochs).

---

## 5. Training Behavior

| Epoch | Train acc | Val acc | Val precision | Val recall | LR | Note |
|---|---|---|---|---|---|---|
| 1 | 0.8935 | **0.9145** | 0.9613 | 0.8627 | 1e-3 | Already 91% after one epoch — well-normalized data lets the model learn fast |
| 5 | 0.9650 | 0.9695 | 0.9740 | 0.9645 | 1e-3 | — |
| 15 | 0.9810 | 0.9803 | 0.9854 | 0.9748 | 5e-4 | LR halved (ReduceLROnPlateau) |
| 29 | 0.9836 | **0.9814** | 0.9850 | 0.9774 | 6.25e-5 | Best val_loss = 0.0541 |
| 34 | — | — | — | — | — | EarlyStopping fires; weights restored to epoch 29 |

- Wall time: **~12 min** for 33 epochs at ~20 s/epoch on Apple M4 GPU
- LR schedule (ReduceLROnPlateau): 1e-3 → 5e-4 → 2.5e-4 → 1.25e-4 → 6.25e-5 → 3.125e-5 → 1.5625e-5

---

## 6. Bugs Found & Fixed (Chronological)

### 6.1 Phase-1 normalization bug masked as a model failure

**Symptom:** Initial training run was stuck at ~50% accuracy across all metrics, val precision and recall both 0. Looked like a model / QAT problem.

**Root cause:** `prepare_dataset.py` line 217 (the **synthetic noise** extraction loop) was missing the `normalize_window()` call — only earthquakes and catalogue noise got per-trace peak normalization. 14,574 of 497,131 traces (~3%) carried raw STEAD counts up to ±2,540, and some had overflowed to `inf` during the float16 cast. Mixed scales of 1.0 and 1000s in the same training set made it impossible for the model to learn.

**Fix:** Added `normalize_window()` to line 217 and **re-ran Phase 1** (~25 min). The in-place re-normalization workaround failed because `inf / inf = NaN` corrupted those traces irrecoverably.

**Lesson:** When a model is stuck at random accuracy with a reasonable architecture, **check the data scale first** before reaching for fancier training tricks.

### 6.2 tfmot does not support Conv1D

**Symptom:** `RuntimeError: Layer conv1d:<class 'tf_keras…Conv1D'> is not supported. You can quantize this layer by passing a tfmot.quantization.keras.QuantizeConfig instance to the quantize_annotate_layer API.`

**Root cause:** tfmot's default quantize registry covers Conv2D, DepthwiseConv2D, Dense, etc. — but not Conv1D.

**Fix:** Restructured the model to use `Conv2D(kernel=(3, 1))` over a `Reshape((200, 1, 3))`. Mathematically identical, fully quantizable, also FPGA-friendlier downstream.

### 6.3 TFLite full-int8 export needs `representative_dataset` even with QAT

**Symptom:** `ValueError: For full integer quantization, a representative_dataset must be specified.`

**Root cause:** QAT bakes quantization params for **weights and intermediate activations**, but the input-tensor scale/zero-point still needs calibration when `inference_input_type=tf.int8` is requested. The Context7 docs were misleading here — saying QAT alone is sufficient.

**Fix:** Added a 200-sample callback that yields normalized float32 windows from the training set:
```python
def representative_dataset():
    for i in range(200):
        yield [calib_data[i:i+1].astype(np.float32)]
converter.representative_dataset = representative_dataset
```

### 6.4 tfmot QAT model can't be `load_model()`'d cleanly

**Symptom:** `ValueError: Layer 'quantize_layer' expected 5 variables, but received 3 variables during loading`.

**Root cause:** Known tfmot bug with Keras 2's native `.keras` save format — the QuantizeLayer's variable bookkeeping doesn't round-trip.

**Fix workflow:** Rebuild the architecture deterministically and load **weights only**, not the full model:
```python
base = build_base_model()
qat = apply_qat(base)
qat.compile(optimizer='adam', loss='binary_crossentropy')   # forces variable creation
qat.load_weights('out/eew_cnn_qat.keras')
```

---

## 7. Final Results

### Test-set metrics (49,714 held-out samples, never seen during training)

```
              precision    recall  f1-score   support
       noise     0.9794    0.9851    0.9822     25,000
  earthquake     0.9849    0.9790    0.9819     24,714
    accuracy                         0.9821     49,714
```

### Confusion matrix
```
                  Predicted noise   Predicted earthquake
True noise            24,628                  372         (false alarms)
True earthquake          519               24,195         (missed events)
```

- **False alarm rate** (false positive rate): 372 / 25,000 = **1.49%**
- **Miss rate** (false negative rate): 519 / 24,714 = **2.10%**

For an EEW system, the miss rate is the safety-critical number — every miss is an earthquake the model failed to flag in time. 2.10% on this dataset is acceptable for a prototype; downstream tuning (threshold sweep, ensembling, temporal smoothing) can push it lower in exchange for slightly higher false alarms.

### Plots

- **`out/training_history.png`** — accuracy and loss curves over 33 epochs. Train and val are tightly overlapping → no overfitting. Loss drops from 0.28 → 0.054.
- **`out/confusion_matrix.png`** — visual of the matrix above.

---

## 8. Output Artifacts

```
out/
├── eew_cnn_int8.tflite     # 16.3 KB  Fully int8 TFLite, int8 IO [1, 200, 3] → [1, 1] — FPGA-ready
├── eew_cnn_qat.keras       # 256 KB   QAT Keras checkpoint (full precision weights + fake-quant params)
├── training_history.png    # Acc + loss curves
├── confusion_matrix.png    # Test-set CM
├── training_metrics.txt    # F1, classification report, CM (text)
└── phase2_train.log        # Full stdout/stderr of the training run
```

---

## 9. How to Re-run Phase 2

```bash
cd ~/Desktop/idp
source .venv-tf/bin/activate
TF_USE_LEGACY_KERAS=1 python train_cnn.py --data-dir ./out --out-dir ./out
```

Runtime: **~12 min** to convergence on Apple M4 GPU (33 epochs at ~20 s/epoch). EarlyStopping with patience=5 prevents wasted compute. Deterministic for fixed `SEED=42`.

---

## 10. Sanity-checking the int8 TFLite

After training, verify the int8 model still classifies correctly:

```bash
source .venv-tf/bin/activate
python -c "
import numpy as np, tensorflow as tf
i = tf.lite.Interpreter('out/eew_cnn_int8.tflite'); i.allocate_tensors()
inp, out = i.get_input_details()[0], i.get_output_details()[0]
X = np.load('out/X_train.npy', mmap_mode='r')[:100].astype(np.float32)
y = np.load('out/y_train.npy')[:100]
in_scale, in_zp = inp['quantization']
out_scale, out_zp = out['quantization']
correct = 0
for j in range(100):
    q = np.clip(X[j]/in_scale + in_zp, -128, 127).astype(np.int8)
    i.set_tensor(inp['index'], q[None]); i.invoke()
    p = (i.get_tensor(out['index'])[0,0] - out_zp) * out_scale
    correct += int((p >= 0.5) == y[j])
print('int8 TFLite accuracy on 100 samples:', correct/100)
"
```

Expect ≥ 0.95 — the int8 model is essentially as accurate as the QAT float model because QAT trained the network to be robust to int8 rounding.

---

## 11. What's Next (Phase 3 onward)

- **Phase 3 (FPGA toolchain integration):** Convert `eew_cnn_int8.tflite` to FPGA bitstream. Two main paths:
  - **HLS4ML:** Reads Keras / ONNX / TFLite, emits Vivado HLS C++. Targets FINN-style streaming dataflow.
  - **Vitis AI:** Xilinx's official toolchain. Reads the QAT Keras model directly via the Vitis AI Quantizer.
  - Either way, the int8 weights are already calibrated — no further training needed.
- **Phase 4:** Synthesize on Artix-7 / ZedBoard, validate real-time inference at 100 Hz.
- **Phase 5:** End-to-end latency budget (sensor → decision), false-alarm tuning under field conditions.

Reference paper for the architecture: `ACCESS2947848.pdf` (in repo root).

---

## 12. Constants Cheat-sheet (from `train_cnn.py`)

| Constant | Value | Why |
|---|---|---|
| `WINDOW` | 200 | 2 s @ 100 Hz — matches Phase 1 |
| `CHANNELS` | 3 | Z / N / E |
| `CONV_FILTERS` | (16, 32, 64) | Powers of two, FPGA-friendly |
| `KERNEL_SIZE` | 3 | Small kernel → small MAC count |
| `POOL_SIZE` | 2 | Halves time axis per stage |
| `BATCH_SIZE` | 512 | Fits comfortably in M4 GPU memory |
| `EPOCHS` | 50 | Upper bound; EarlyStopping cuts shorter |
| `LEARNING_RATE` | 1e-3 | Adam default; ReduceLROnPlateau anneals down |
| `PATIENCE` | 5 | EarlyStopping patience on val_loss |
| `SEED` | 42 | Determinism (matches Phase 1) |
| `F1_TARGET` | 0.92 | Project requirement; achieved 0.9819 |

**Do not change these casually** — the Phase 3 FPGA toolchain depends on the model architecture being stable. If you need to retrain, prefer keeping the architecture fixed and tuning only training-time hyperparameters.
