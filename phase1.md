# Phase 1 — Dataset Preparation (Pre-processing)

This document captures **what was done, what was decided, and what was discovered** during Phase 1 of the Real-Time Earthquake Early Warning System project. It is intended both as a human reference and as context for future Claude Code sessions.

---

## 1. Goal of Phase 1

Build a **balanced, ~500,000-instance training subset** of the STEAD dataset, optimized for a 1D-CNN that will eventually run on Artix-7 / ZedBoard FPGA hardware.

| Aim | Constraint |
|---|---|
| 250k earthquake windows + 250k noise windows | 50/50 class balance |
| Each window = 200 samples (2 s @ 100 Hz), 3 channels (Z/N/E) | Matches CNN input shape |
| Earthquake window aligned 50 samples *before* P-arrival, 150 samples *after* | Model must learn to fire **just after** P-arrival, not centred on it |
| `float16` storage | Halves disk/RAM for FPGA training |
| Deterministic (`SEED = 42`) | Reproducible reruns |

---

## 2. Source Data

| File | Size | Location | Notes |
|---|---|---|---|
| `merge.csv` | 370 MB | `~/Downloads/archive/merge.csv` | Catalogue: 35 columns × 1,265,657 rows |
| `merge.hdf5` | ~91 GB | `~/Downloads/archive/merge.hdf5` | Waveforms, keyed by `data/<trace_name>` → `(6000, 3)` arrays |

**Raw category counts:**
- `earthquake_local`: 1,030,231
- `noise`: 235,426

These two files are *not* in the repo and should be deleted from disk after Phase 1's outputs are verified.

---

## 3. Filtering Decisions

### Earthquake class
```
trace_category == "earthquake_local"
AND  snr_max          ≥ 20 dB
AND  source_magnitude ∈ [2.0, 5.0]
AND  p_arrival_sample ≥ 50           (so 50-sample pre-window fits inside the trace)
```
After filtering: **247,131 candidates** — slightly under the 250k target, so all of them were used (no further sampling).

### Noise class
Two-source strategy to hit exactly 250k:
1. **All catalogue noise**: 235,426 traces, first 200 samples each.
2. **Synthetic noise**: 14,574 *earthquake* traces with `p_arrival_sample > 250`, taking the first 200 samples (guaranteed pre-event ambient because they sit well before the P-arrival).

Total noise = **250,000**.

### Final dataset
- **497,131 windows** total (247,131 EQ + 250,000 noise) — not exactly 500k because the SNR/magnitude filter capped earthquakes at 247k.
- Saved in `out/X_train.npy` (568.9 MB) and `out/y_train.npy` (0.5 MB).

---

## 4. Bugs Found & Fixed in `prepare_dataset.py`

The original script in the repo had two latent bugs that would have silently corrupted the dataset. Both are fixed in the committed version.

### 4.1 `snr_db` is a 3-channel string, not a float
STEAD stores SNR per channel as a stringified array, e.g. `'[56.79999924 55.40000153 47.40000153]'`. The original filter
```python
df["snr_db"].astype(float) >= SNR_THRESHOLD
```
would crash on this. **Fix:** added `parse_snr_max()` which strips brackets, splits on whitespace, and returns the **max** of the 3 channel SNRs. The earthquake passes the SNR filter if at least one channel exceeds 20 dB.

### 4.2 Raw STEAD counts overflow `float16`
`float16` has a max value of ±65,504. Raw STEAD waveforms are integer counts that routinely exceed this. The original direct cast
```python
X[idx] = crop(...).astype(np.float16)
```
silently produced `inf` values for many traces (caught at runtime as `RuntimeWarning: overflow encountered in cast`). **Fix:** added `normalize_window()` which divides each `(200, 3)` window by `max(|x|)` *before* the cast, putting values safely in `[-1, 1]`. This is also standard practice for amplitude-invariant CNN training across stations with different gains.

### 4.3 Cosmetic: replaced manual `every 50k` print counters with `tqdm` progress bars on all three extraction loops.

---

## 5. Output Structure

```
out/
├── X_train.npy            # (497131, 200, 3) float16   — 568.9 MB
├── y_train.npy            # (497131,)         uint8    —   0.5 MB   (1=earthquake, 0=noise)
├── crop_verification.png  # Sanity plot — see §6
└── run.log                # Full stdout/stderr of the extraction run
```

After shuffling with `np.random.default_rng(SEED).permutation(...)`, the two classes are interleaved randomly throughout the arrays.

---

## 6. Verification Plot (`crop_verification.png`)

The plot has **two rows × three columns**:

| Row | Colour | What it shows |
|---|---|---|
| Top | Red | A single **earthquake** window — the 3 channels (Z, N, E) of one trace, plotted over the 2-second crop |
| Bottom | Green | A single **noise** window — the 3 channels of one randomly chosen noise sample |

### What to look for in the EARTHQUAKE row
- The dashed **blue vertical line at t = 0.5 s** marks the catalogued P-arrival sample.
- For t < 0.5 s the waveform should be **quiet / low-amplitude** (pre-event ambient).
- At t = 0.5 s there should be a **sharp impulse** — the P-wave first break.
- For t > 0.5 s the trace should ring with **higher-amplitude oscillations** (the earthquake signal).
- All three channels (Z/N/E) should show this transition simultaneously, though amplitudes may differ between channels (vertical Z is usually clearest for P-waves).

✅ **If you see a clean step-up at the dashed line, the 50-pre / 150-post offset is aligned correctly** and the dataset is safe to ship to training.

❌ Red flags: P-wave appearing *before* the dashed line (off-by-one in cropping), no visible transient at all (wrong trace or bad SNR), or amplitudes flat-lining at exactly 1.0 (float16 overflow — should not happen since the normalization fix).

### What to look for in the NOISE row
- Stationary low-amplitude background across all 3 channels.
- **No impulsive transient** anywhere in the 2-second window.
- Some periodic content is fine (microseismic noise, cultural noise) — the only requirement is the absence of an event.

### Amplitude scale
Both rows are normalized per-trace to roughly `[-1, 1]`. Absolute amplitudes are *not* preserved — only the waveform shape matters for the CNN.

---

## 7. How to Re-run Phase 1

```bash
cd ~/Desktop/idp
source .venv/bin/activate
python prepare_dataset.py \
    --csv  ~/Downloads/archive/merge.csv \
    --hdf5 ~/Downloads/archive/merge.hdf5 \
    --out  ./out
```

Runtime on this machine: **~21 minutes** (HDF5 random-access at ~270 traces/sec). Deterministic for a fixed `SEED`.

Dependencies (already in `.venv`): `numpy 2.4`, `pandas 3.0`, `h5py 3.16`, `matplotlib`, `tqdm 4.67`.

---

## 8. What's Next (Phase 2 onward)

- **Phase 2:** Define and train the 1D-CNN architecture in PyTorch / TensorFlow. Inputs: `X_train` shape `(N, 200, 3)`. Outputs: binary classifier (earthquake vs noise).
- **Phase 3:** Quantize the trained model to int8/int16 for FPGA deployment.
- **Phase 4:** HLS / Verilog implementation on Artix-7 / ZedBoard, real-time inference at 100 Hz.

Reference paper for the architecture: `ACCESS2947848.pdf` (in repo root).

---

## 9. Constants Cheat-sheet (from `prepare_dataset.py`)

| Constant | Value | Why |
|---|---|---|
| `TARGET_EQ` | 250,000 | EQ class target |
| `TARGET_NOISE` | 250,000 | Noise class target |
| `WINDOW` | 200 | 2 s @ 100 Hz |
| `PRE_ARRIVAL` | 50 | Samples before P-arrival kept in window |
| `POST_ARRIVAL` | 150 | Samples after P-arrival kept in window |
| `PRE_NOISE_LEN` | 200 | First-200-samples slice for noise |
| `MIN_P_FOR_SYNTH` | 250 | EQ trace must have P > 2.5 s in for safe pre-event noise crop |
| `SNR_THRESHOLD` | 20.0 | dB, max-of-3-channels |
| `MAG_LOW`, `MAG_HIGH` | 2.0, 5.0 | Magnitude band |
| `CHANNELS` | 3 | Z / N / E |
| `SEED` | 42 | Determinism |

**Do not change these casually** — the CNN architecture downstream depends on the `(200, 3)` shape, the 50/150 P-arrival offset, and the 50/50 class balance.
