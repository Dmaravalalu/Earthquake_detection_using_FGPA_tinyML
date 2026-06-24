# Epicenter Side-Project — Future Work

Companion to [`phase_epicenter.md`](phase_epicenter.md). Captures what we tried,
what failed, and the principled redesign that should actually move the
numbers. **Nothing here is implemented yet.** Pick this up when the side
project gets revisited.

---

## 1. Shipped state (v2, as of the current `main`)

```
Distance MAE        :  33.80 km        (mean true dist = 87 km)
Back-azimuth MAE    :  85.52°          (uniform random ≈ 90°)
Localization MAE    : 105.36 km        (median 90.5 km)
```

Distance is mildly predictive; back-azimuth is **essentially random** —
the sin/cos heads saturated their `tanh` activations within the first
five epochs and never recovered. The model files (`out/epicenter_cnn.keras`,
`out/epicenter_norm.json`, `out/epicenter_metrics.json`) and the Streamlit
"Epicenter (side project)" tab work end-to-end with a clear warning banner,
so the demo is honest about the limitation.

---

## 2. Why retraining alone won't fix it

We tried two iterations:

| Variant | params | dist MAE | back-az MAE | loc MAE |
|---|---|---|---|---|
| v1 — Conv 16/32/64 k=3, dense 32, global-peak norm | 8 k | 38.45 km | 82.09° | 96.56 km |
| **v2 (shipped)** — Conv 32/64/128/128 k=5, dense 128→64, per-channel norm + aux peaks | 130 k | **33.80 km** | 85.52° | 105.36 km |

Distance improved with the wider trunk; back-azimuth didn't. **The bottleneck
is not capacity.** A generic CNN on per-trace normalised raw waveforms doesn't
have the inductive bias to discover, from scratch, the polarisation geometry
that classical seismology solves with eigenanalysis of a 50-sample
covariance matrix.

Specifically:

- **Single-station back-azimuth has a fundamental ±180° ambiguity** absent
  first-motion polarity (P-wave particle motion is radial — same horizontal
  vector for source at θ or θ+180°).
- **Per-trace peak normalisation drops the absolute amplitude** that is the
  strongest distance cue.
- **No band-pass filtering** leaves microseismic noise (~0.1–0.5 Hz) on
  par with the P-wave energy band (1–10 Hz) in many traces.
- **5-second window** captures S-arrival only for `dist ≲ 40 km`. Mean
  distance in our dataset is 87 km, so most traces have no S-P time in
  the window at all.
- **`tanh` sin/cos heads at LR 1 × 10⁻³ saturated within 5 epochs.** Training
  loss climbed from 0.62 → 1.40 and never came back, even after
  ReduceLROnPlateau halved the LR. Distance survived because its linear
  head can't saturate.

---

## 3. Proposed redesign — what to actually change

The ordering is roughly "biggest payoff first". Items marked **★** are
the ones with the strongest physics-motivated case.

### 3.1 Preprocessing (do this even before touching the model)

1. **★ Polarisation features as auxiliary input.** For each window, compute
   the covariance matrix of `(N, E)` over a 50-sample slice immediately after
   P, take its dominant eigenvector, and feed `(eig_x, eig_y, eig_ratio)` as
   3 auxiliary scalars. This is the canonical seismology answer to
   back-azimuth — handing it to the model removes most of the learning
   burden. *Expected:* back-az MAE → ~30–45° (from ~85°).
2. **★ Bigger window: 10 s (1000 samples) with P at sample 200.** Captures
   S-arrival for `dist ≲ 80 km`, which covers half the dataset. *Expected:*
   distance MAE → ~22–27 km.
3. **★ Band-pass filter 1–10 Hz** per channel before normalisation
   (`scipy.signal.butter` order 4, zero-phase). Removes the
   ~0.1–0.5 Hz microseismic background that drowns the P-wave energy on
   many traces. *Expected:* both heads better-conditioned, ~10–15 % MAE
   improvement on top of the other changes.
4. **First-motion polarity feature.** Sign of the largest Z deflection in
   `[P, P+15]` samples. Resolves the ±180° back-azimuth ambiguity. Adds
   one scalar input.
5. **Per-channel z-score (RMS-based) instead of peak normalisation.**
   Robust to single-sample spikes that dominate peak-norm. Keep the
   per-channel RMS values as aux inputs (like we did with peaks).

### 3.2 Architecture

6. **Multi-branch trunk.** Shared 4-block conv body for low-level features
   (current 32/64/128/128 k=5 is fine), then two separate dense branches:
   `dist_branch` (Dense 128 → 64 → linear) and `az_branch` (Dense 128 → 64
   → 2 linear units for sin/cos). Stops one head's gradient from
   bullying the other.
7. **Drop `tanh` on sin/cos**, use **linear** with no activation. Then
   normalise at inference: `(sin, cos) /= sqrt(sin² + cos²)`. Linear heads
   don't saturate.
8. **Initialize azimuth-head weights small** (`HeNormal(scale=0.1)`) so
   initial output is near 0 rather than at the tanh saturation edge.
9. **GroupNorm or LayerNorm instead of BatchNorm** in the conv trunk if
   training stability remains an issue at low LR. BN's running stats can
   drift badly at small effective batch sizes.

### 3.3 Training recipe

10. **Lower initial LR (`3e-4`) + warmup over 1 epoch + cosine decay.**
    The 1e-3 + ReduceLROnPlateau combo we used let the heads diverge
    before LR was reduced.
11. **Gradient clipping `clipnorm=1.0`.** Stops the early instability
    cold.
12. **Huber loss on distance** (`delta=1.0` in z-units), keep MSE on sin/cos.
    Huber is robust to the long-distance tail.
13. **Larger batch (512 or 1024).** Stabilises gradient estimates.
14. **Loss-weight sweep.** Start `dist:sin:cos = 1:2:2` to compensate for
    distance being on a different scale after z-normalisation. Tune.

### 3.4 Data-side levers (cheap)

15. **Don't cap distance at 300 km** — extend to 500 km. More near-field
    examples for the model.
16. **Filter out STEAD traces with `snr_max < 25 dB`** instead of 15 dB.
    Cleaner azimuth signal at the cost of fewer samples — should still
    leave > 100 k.
17. **Stratify by distance** in train/val/test split so the long tail is
    represented in each set.

---

## 4. Realistic ceiling with full redesign

If all of §3.1 + §3.3 land cleanly, the achievable test-set numbers on this
STEAD subset are roughly:

```
Distance MAE        :  ~20–25 km      (vs 34 now)
Back-azimuth MAE    :  ~30–45°        (vs 85 now)
Localization MAE    :  ~40–60 km      (vs 105 now)
```

Still not production-grade, but **honest demo-grade** — the predicted
epicenter dot would land in the right hemisphere and within a few tens
of km of the truth on most traces.

True production-grade single-station epicenter regression (sub-10°
back-azimuth, sub-10 km distance) requires either:

- Multi-station array data (back-azimuth becomes a triangulation problem)
- Specialised seismology toolchains (PhaseNet for picks + GMPE for
  distance + classical polarisation for back-azimuth)
- Much larger transformer/attention models trained on tens of millions
  of curated waveforms

None of those fit the "side-project tab in the Streamlit demo" scope.

---

## 5. Effort estimate

If revisited as a focused session:

| Step | Effort | Compute |
|---|---|---|
| Implement polarisation features + bandpass + 10 s window in `prepare_epicenter_dataset.py` | ~1 hr | 15 min re-prep |
| Redesign model + training recipe in `train_epicenter.py` | ~1 hr | 15–20 min retrain |
| Wire new aux inputs into Streamlit tab | ~30 min | — |
| Update docs | ~15 min | — |
| **Total** | **~3 hrs work + 30 min compute** | |

---

## 6. What NOT to bother trying

Tried and ruled out, or known dead ends:

- **Just making the CNN deeper / wider on the same inputs.** Capacity wasn't
  the bottleneck. v2 was 16× the params of v1 and back-azimuth got worse.
- **Adding more epochs.** Both v1 (44 epochs) and v2 (44 epochs) early-stopped
  cleanly. Loss had plateaued.
- **Different per-trace normalisation alone.** Peak vs std vs per-channel
  all preserve the same physical information; without polarisation features
  the model can't extract it.
- **Switching to Conv1D from Conv2D-on-reshape.** Mathematically identical
  over the time axis; no expected improvement.
- **Predicting (lat, lon) directly.** Tried mentally — much worse target
  geometry. Stick with (distance, back-az) → destination point.
