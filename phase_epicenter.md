# Epicenter Side-Project — Distance + Back-Azimuth Regression

This document covers the **independent second model** built alongside the main
earthquake-detection CNN. It is intentionally **decoupled from the FPGA path**.

> **Scope reminder.** The Phase 1 / Phase 2 / Phase 3 classifier (`train_cnn.py`,
> `eew_cnn_int8.tflite`) is the one we push to the Artix-7. *This* model is a
> Keras-only side experiment that estimates **where** an earthquake came from,
> and is shown as a separate Streamlit tab. It does NOT run on hardware and
> there is no QAT / hls4ml / Vivado plan for it.

> **Shipped state.** Distance MAE ≈ 34 km, back-azimuth MAE ≈ 85° (≈ random),
> localization MAE ≈ 105 km. Distance is mildly predictive; back-azimuth is
> essentially random — the Streamlit tab shows a warning banner reflecting
> this. The proposed redesign (band-pass filtering, longer window,
> polarisation features as aux input, training-stability fixes) lives in
> [`epicenter_future_work.md`](epicenter_future_work.md). The shipped model
> is honest demo-grade only.

---

## 1. Goal

From a 5-second 3-channel window cropped around the P-arrival, predict:

- `source_distance_km` — receiver-to-epicenter great-circle distance
- `back_azimuth_deg` — direction from the receiver pointing toward the source

We deliberately **do not regress latitude/longitude directly** — that target is
poorly conditioned because lat/lon are unbounded and station-dependent. Instead
we predict distance + bearing and reconstruct the epicenter geometrically.

---

## 2. Data Preparation — `prepare_epicenter_dataset.py`

| Aspect | Choice |
|---|---|
| Source | STEAD `merge.csv` + `merge.hdf5` (same as Phase 1) |
| Category filter | `trace_category == "earthquake_local"` only |
| Quality filter | `snr_max ≥ 15 dB`, `magnitude ∈ [2.0, 5.0]`, `0 < distance ≤ 300 km`, back-az + lat/lon present, `p_arrival_sample ≥ 100` |
| Crop | **500 samples (5 s @ 100 Hz)** — `100` pre-P + `400` post-P |
| Channels | Z / N / E (untouched STEAD order) |
| Normalisation | per-trace peak so values land in `[-1, 1]` |
| Storage | `float16` (same precision-saving rationale as Phase 1) |
| Sample cap | 200,000 traces (random sample, `SEED = 42`) |

Output: `out/epicenter_data.npz` containing `X` plus all metadata needed for
training and for the Streamlit map (`recv_lat/lon`, `src_lat/lon`,
`dist_km`, `back_az_deg`, `trace_name`).

Why a larger window than the classifier (500 vs 200 samples)? Distance
estimation benefits from S–P time when S arrives within the window
(roughly `dist ≤ 40 km` at Vp/Vs ≈ 1.73), and from amplitude/spectral
content for everything farther. A 5 s window gives both signals room to
appear without bloating disk/compute.

### Run it

```bash
source .venv-tf/bin/activate
python prepare_epicenter_dataset.py \
    --csv  ~/Downloads/archive/merge.csv \
    --hdf5 ~/Downloads/archive/merge.hdf5 \
    --out  ./out
```

Expected runtime: **~10–15 minutes** (HDF5 random reads dominate, ~270
traces/sec).

---

## 3. Model — `train_epicenter.py` (v2)

The first attempt — a tiny 3-block Conv2D trunk (16/32/64 channels, kernel 3,
single Dense 32 head) trained on globally peak-normalised inputs — converged
quickly but **didn't learn back-azimuth at all** (sin/cos MAE plateaued at
`2/π ≈ 0.637`, the value you get by predicting 0 for both). Distance learning
was weak too (RMSE 55 km vs target std 61 km).

Two changes for v2:

1. **Per-channel normalisation + auxiliary cross-channel peaks.** The conv
   trunk now receives each channel renormalised to ≈ `[-1, 1]` so one loud
   channel can't dominate the others; the *original* per-channel peak
   amplitudes are fed in as a tiny `(3,)` auxiliary input concatenated
   after GAP. That preserves the polarisation (back-azimuth) signal while
   letting the conv layers focus on per-channel shape features.
   Implemented as `preprocess_window` in `epicenter_utils.py`.
2. **Wider trunk, deeper head.** Four Conv-BN-ReLU-Pool blocks with
   32/64/128/128 channels and a `5x1` kernel, then a two-layer dense head
   (128 → Dropout 0.2 → 64). Total ≈ 130 k params (vs 8 k in v1) —
   still small enough to train in minutes but with the inductive bias to
   capture cross-channel polarisation.

```
waveform                  (500, 3)               channel_peaks   (3,)
to_2d (Reshape)           (500, 1, 3)
conv_1 (Conv2D  32, 5x1)+BN+ReLU+MaxPool(2,1)   (250, 1,  32)
conv_2 (Conv2D  64, 5x1)+BN+ReLU+MaxPool(2,1)   (125, 1,  64)
conv_3 (Conv2D 128, 5x1)+BN+ReLU+MaxPool(2,1)   ( 62, 1, 128)
conv_4 (Conv2D 128, 5x1)+BN+ReLU+MaxPool(2,1)   ( 31, 1, 128)
gap (GlobalAveragePool2D)                       (128,)
concat([gap, channel_peaks])                    (131,)
dense_1 (Dense 128, ReLU) → Dropout(0.2)        (128,)
dense_2 (Dense 64,  ReLU)                       ( 64,)
├─ dist_z  (Dense 1, linear)  → standardised log10(distance_km + 1)
├─ sin_az  (Dense 1, tanh)    → sin(back_azimuth)
└─ cos_az  (Dense 1, tanh)    → cos(back_azimuth)
```

The Conv2D-on-reshape trick is kept (rather than switching to Conv1D)
so the architecture stays QAT-compatible if int8 export ever becomes
useful.

### Targets

- **Distance**: `z = (log10(d + 1) − μ) / σ`, with `μ`, `σ` fit on the
  training split only and saved to `out/epicenter_norm.json` so the
  encoder is reproducible at demo time. Log-space tames the long right
  tail; z-normalisation keeps MSE well-behaved.
- **Back-azimuth**: predicted as `(sin θ, cos θ)` jointly, avoiding the
  0° / 360° wrap. Decoded at inference via `atan2`. Tanh activations
  keep predictions on the unit circle.

### Loss / training recipe

```
optimizer        Adam(lr=1e-3)
loss             MSE on each of {dist_z, sin_az, cos_az}; equal weights
metrics          MAE per head
batch_size       256
max_epochs       80
EarlyStopping    monitor=val_loss, patience=8, restore_best_weights=True
ReduceLROnPlateau  factor=0.5, patience=3, min_lr=1e-5
split            80 / 10 / 10  (deterministic via SEED=42)
```

### Run it

```bash
source .venv-tf/bin/activate
TF_USE_LEGACY_KERAS=1 python train_epicenter.py \
    --data ./out/epicenter_data.npz \
    --out-dir ./out
```

Expected runtime on Apple M4 GPU: roughly **15–25 min** on 200k samples,
similar to the classifier.

---

## 4. Epicenter Calculation — `epicenter_utils.py`

Once the model outputs `(distance, back_azimuth)`, we walk from the
receiver along that bearing on the great circle:

```python
from epicenter_utils import destination_point, haversine_km
pred_lat, pred_lon = destination_point(
    recv_lat, recv_lon, back_azimuth_deg=az_pred, distance_km=dist_pred
)
loc_err_km = haversine_km(pred_lat, pred_lon, src_lat, src_lon)
```

Both formulas are standard spherical geometry (Earth radius
`6371.0088 km`), wrap longitudes to `[-180, 180]`, and broadcast cleanly
over numpy arrays. Localization error is reported as the Haversine
distance between the geometrically-derived predicted epicenter and the
catalogued `(source_latitude, source_longitude)`.

---

## 5. Outputs

```
out/
├── epicenter_data.npz                X + metadata (≈400 MB for 200k samples)
├── epicenter_cnn.keras               trained Keras model
├── epicenter_norm.json               distance encoder mu, sigma
├── epicenter_metrics.json            test-set metrics (machine-readable)
├── epicenter_metrics.txt             same, human-readable
├── epicenter_training_history.png    loss curves
└── epicenter_predictions.png         distance scatter, back-az scatter, loc-error histogram
```

---

## 6. Streamlit Tab (`app.py`)

A fourth tab, **"Epicenter (side project)"**, has been added to the demo.
For a randomly picked earthquake trace it:

1. Crops the same 500-sample window the model trained on.
2. Runs the Keras epicenter model in float (no int8 quantization).
3. Decodes distance + back-azimuth.
4. Walks the destination-point formula from the receiver.
5. Plots **receiver / true epicenter / predicted epicenter** on `st.map`.
6. Reports the Haversine localization error in km.
7. Surfaces the saved test-set metrics in an expander.

The tab degrades gracefully if `epicenter_cnn.keras` is missing — it
prints the commands to train it instead of erroring out. It also refuses
to run on noise traces, since they have no source coordinates.

---

## 7. Why This Stays Off the FPGA

- The classifier's bitstream is the deliverable. Adding a second model
  to the Artix-7 burns DSPs and BRAM we already budgeted (~70 of 90 DSPs
  used by the classifier alone).
- Regression accuracy from 200k STEAD samples is acceptable for a
  software demo but not for safety-critical edge use.
- The architecture is kept Conv2D-on-reshape so that, if it ever became
  worth deploying, we could wrap it with `tfmot.quantize_model` and run
  it through hls4ml without restructuring.

---

## 8. Constants Cheat-sheet

| Constant | File | Value | Why |
|---|---|---|---|
| `WINDOW` | both | 500 | 5 s @ 100 Hz |
| `PRE_ARRIVAL` | both | 100 | 1 s of ambient before P |
| `POST_ARRIVAL` | prep | 400 | 4 s after P — captures S for nearby quakes |
| `SNR_THRESHOLD` | prep | 15.0 | Looser than the classifier's 20 — regression needs range |
| `DIST_MAX_KM` | prep | 300.0 | Limit to local quakes |
| `TARGET_N` | prep | 200_000 | Cap for runtime |
| `CONV_FILTERS` | train | (32, 64, 128, 128) | Wider v2 trunk |
| `KERNEL_SIZE` | train | 5 | Larger temporal receptive field per block |
| `DENSE_UNITS` | train | (128, 64) | Two-layer head |
| `DROPOUT` | train | 0.2 | After the first dense layer |
| `BATCH_SIZE` | train | 256 | Smaller than classifier (regression needs noisier gradient less) |
| `PATIENCE` | train | 10 | More patience than classifier — regression converges slower |
| `SEED` | both | 42 | Determinism |
