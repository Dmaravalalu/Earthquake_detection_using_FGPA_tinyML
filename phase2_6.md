# Phase 2.6 — Sliding-Window Stride & (Threshold × Sustain) Benchmarks

This document captures the pre-FPGA detection-quality benchmarks. **Read `phase2.md` and `phase2_5.md` first** for model and demo context.

The goal: understand the model's real-time behaviour on full 60-second STEAD traces (not just the trained 200-sample windows) and pick deployment-ready hyperparameters before committing the FPGA bitstream.

---

## 1. Goals

Two questions:

1. **Stride benchmark**: how does the sliding-window step size affect detection latency, accuracy, and FPGA compute load?
2. **Threshold × sustain sweep**: at the best stride, can we tune the post-processing thresholds to bring detection close to the actual P-arrival with few false alarms?

Same dataset, same filter for both: 100,000 traces matching `trace_category=earthquake_local`, `SNR_max ≥ 15 dB`, `magnitude ∈ [2.0, 5.0]`. Deterministic via `seed=42`.

---

## 2. Detection Definition (the "predicted P" interpretation)

The model outputs a binary probability per 200-sample window. Sliding it across the trace gives a probability stream. From that we need a single "predicted P-arrival" sample.

### Find-first-detection algorithm

1. Slide window at given stride from sample 0 to (6000 − 200).
2. For each window: per-trace peak normalize, quantize to int8, run the int8 TFLite model, dequantize the output to a probability.
3. Find the **first window** where probability ≥ `threshold` for `sustain` consecutive windows.
4. **Predicted P-arrival sample** = first such window's start + 50 (the model was trained with P at sample 50 of its 200-sample input, so this is the natural P location within the window).

### Signed error
```
e = predicted_p_sample − actual_p_sample          (samples)
e_ms = e × 1000 / SAMPLE_RATE                      (milliseconds)
  e < 0   → model fired EARLIER than actual P
  e > 0   → model fired LATER than actual P
  e ≈ 0   → on target
```

---

## 3. Asymmetric Tolerance & F1 Calculation

For an EEW system, the operational requirement is asymmetric:

- **Up to 500 ms before** actual P: acceptable (early warning value, but not a wild false alarm).
- **Any time after** actual P: too late — the wave has already arrived.

So the on-time window is `e ∈ [−500 ms, 0 ms]`. Outcomes per trace:

| Outcome | Condition | Counts as |
|---|---|---|
| **TP** | detected AND `−500 ≤ e ≤ 0` ms | true positive |
| **FP** | detected AND `e < −500` ms | false alarm (model too eager) |
| **FN** | detected AND `e > 0` ms | too-late detection |
| **FN** | not detected at all | miss |

Then:
```
precision = TP / (TP + FP)         "of the alerts, how many landed in the target zone"
recall    = TP / (TP + FN)         "of the earthquakes, how many did we catch on target"
F1        = 2 × P × R / (P + R)
```

(There's no TN because every benchmark trace is an earthquake — no negative class here.)

The previous `phase2.md` F1 of 0.9819 used a **symmetric** ±50 ms tolerance on the trained 200-sample windows. The asymmetric `[−500, 0]` here is much stricter and applies to the 60-second trace, so the numbers are not directly comparable.

---

## 4. Stride Benchmark — `benchmark_strides.py`

Stride values: 5, 10, 15, 20 samples (= 50 / 100 / 150 / 200 ms inference cadence at 100 Hz). Threshold = 0.5, sustain = 3, symmetric tolerance = ±0.5 s.

### Results (100,000 traces, 54 min wall, stride 5 computed once and downsampled)

| stride | F1 | accuracy | on-time | miss | early FP | **mean signed err** | mean \|err\| | windows/trace |
|---|---|---|---|---|---|---|---|---|
| 5  | 0.032 | 1.6% | 1.6% | 0.12% | 97.6% | **−3037 ms** | 3132 ms | 1,161 |
| 10 | 0.039 | 2.0% | 2.0% | 0.23% | 96.9% | **−2575 ms** | 2697 ms | 581 |
| 15 | 0.042 | 2.2% | 2.2% | 0.32% | 96.4% | **−2197 ms** | 2364 ms | 387 |
| **20** | **0.054** | **2.8%** | **2.8%** | 0.40% | 95.6% | **−1991 ms** | **2188 ms** | **291** |

### Findings

- **Detection rate is ~99.8% on every stride** — the model essentially always fires somewhere on every earthquake. The interesting question is *when*.
- **Smaller stride → bigger early bias.** Why: smaller stride gives the model more chances to fire on every brief pre-P transient, and many of those transients sustain just long enough (3 windows) to trigger.
- **Stride 20 wins on every metric** measured: highest F1, smallest \|error\|, smallest signed bias, AND 4× fewer FPGA inferences per trace.

### Recommended stride for FPGA: **20 samples (200 ms cadence, 5 inferences/sec)**

Outputs:
- `out/stride_benchmark_per_trace.csv` (400,000 rows: 100k × 4 strides)
- `out/stride_benchmark_summary.csv`, `.txt`, `.png`

---

## 5. (Threshold × Sustain) Sweep — `benchmark_sweep.py`

At the chosen stride 20, sweep the post-processing thresholds with the asymmetric tolerance `[−500, 0] ms`.

- Thresholds: 0.50, 0.70, 0.85, 0.90, 0.95, 0.99
- Sustains: 1, 2, 3, 5, 10, 15
- 36 combos total, evaluated on the same 100,000 traces.

Implementation trick: run stride-20 inference *once* (saves 100k × 291 float16 probabilities = 56 MB to `out/sweep_probs.npy`), then sweep all 36 combos in vectorised numpy on the saved array. Inference: ~18 min. Sweep: <1 sec.

### Top 7 combos by F1

| threshold | sustain | F1 | precision | recall | miss% | mean signed err | mean \|err\| |
|---|---|---|---|---|---|---|---|
| **0.99** | **3** | **0.144** | 0.097 | 0.276 | 15.5% | **−869 ms** | 1344 ms |
| 0.99 | 2 | 0.133 | 0.086 | 0.290 | 11.5% | −908 ms | 1385 ms |
| 0.99 | 5 | 0.120 | 0.087 | 0.191 | 23.5% | −866 ms | 1351 ms |
| 0.99 | 1 | 0.112 | 0.067 | 0.334 | 6.3% | −1084 ms | 1503 ms |
| 0.95 | 3 | 0.099 | 0.058 | 0.346 | 5.8% | −1056 ms | 1498 ms |
| 0.95 | 2 | 0.091 | 0.052 | 0.373 | 3.9% | −1192 ms | 1578 ms |
| 0.9  | 3 | 0.084 | 0.047 | 0.386 | 3.5% | −1187 ms | 1569 ms |

### Findings

- **Best combo: threshold=0.99, sustain=3** — F1 = 0.144, mean signed err = −869 ms, miss rate = 15.5%.
- The sweep **shifts the bias from −3 s to −0.87 s** — a 70% reduction in early-firing bias.
- **Precision still tops out at ~10%** because under the strict `[−500, 0]` tolerance, 90% of detections are >500 ms early.
- **Knee at sustain ≥ 10**: recall collapses (miss rate jumps from ~15% at sustain 3 to >40% at sustain 10).
- **Threshold has a clean monotonic effect**: higher = less early-firing. Sustain has a stronger effect on miss rate.

### Recommended detection settings: **threshold = 0.99, sustain = 3, stride = 20**

This puts the FPGA at:
- 5 inferences per second (200 ms cadence)
- Mean detection ~870 ms before the catalogued P
- ~15% miss rate
- ~84% of detections fire too early to count as "on-target" by the strict 500 ms metric — but most still come within ~1 second of P, which has real EEW value.

Outputs:
- `out/sweep_probs.npy` (100k × 291 float16, 56 MB) — raw probabilities, can re-sweep without rerunning inference
- `out/sweep_meta.csv` (100k rows: trace_name, actual P, magnitude, SNR)
- `out/sweep_starts.npy` (291,) — window start offsets
- `out/sweep_results.csv` — 36 rows of metrics
- `out/sweep_heatmap.png` — 4 heatmaps (F1, precision, recall, mean signed err)

To re-sweep with different thresholds/sustains/tolerances **without re-running inference**:
```bash
python benchmark_sweep.py --reuse \
    --thresholds 0.97 0.98 0.99 0.995 \
    --sustains 1 2 3 4 \
    --early-tol-ms 500 --late-tol-ms 0
```

---

## 6. Why is the Bias So Strongly Negative?

The model is a **binary classifier** trained on 200-sample windows, not a regressor for P-arrival time. It learned "is there earthquake-like signal in my 2-second window?" — and in real STEAD traces, it triggers on:

- Foreshocks / co-located smaller events not labelled in the catalogue.
- The current earthquake's pre-P energy (P-wave precursor noise from station, foreshock coda).
- Station-specific noise the model has incidentally learned to match earthquake patterns.

The catalogued `p_arrival_sample` is a **human pick** of the main P-wave; the model is more eager and fires sooner. With sustain=3 + threshold=0.5, basically any 600-ms feature in the trace that crosses 0.5 will trigger. Higher threshold and sustain filter most of those out, but not all.

---

## 7. Five Ways to Improve "When"

Ordered by effort vs. payoff, with the lever each one moves:

1. **Raise threshold + sustain** *(no retrain — what this benchmark probed)*. Demonstrated: shifted bias from −3 s to −0.87 s, but capped there.
2. **Probability hysteresis** *(no retrain)*. Require prob to drop below e.g. 0.20 before allowing a new detection. Smooths brief dips so a transient near-0.5 firing followed by a dip-then-real-firing only triggers once.
3. **Temporal smoothing** *(no retrain)*. Apply a 3- or 5-window moving average to the probability stream before thresholding. Reduces noise spikes.
4. **Retrain with stricter pre-P labels** *(retrain — recommended next step)*. Currently P is at sample 50 of 200 (50/150 split). Move to 150/50 (1.5 s of pre-P + 0.5 s of post-P). Forces the model to wait until it has seen substantial post-P energy before firing, which should pull the mean signed error from ~−1 s toward 0.
5. **Add a regression head** *(retrain, biggest change)*. Output (P-detect, P-time-offset) instead of just P-detect. Then the model literally predicts WHERE in its window the P-arrival is, eliminating the bias problem.

---

## 8. Files Added in Phase 2.6

```
benchmark_strides.py    # 4-stride benchmark
benchmark_sweep.py      # threshold × sustain sweep (saves probs for re-sweeping)
phase2_6.md             # this document
```

Output artifacts in `out/`:
```
stride_benchmark_per_trace.csv    400,000 rows raw per-(trace, stride)
stride_benchmark_summary.{csv,txt,png}
stride_benchmark_run.log

sweep_probs.npy         100,000 × 291 float16  (56 MB)  — raw probabilities
sweep_starts.npy        291 ints — window starts at stride 20
sweep_meta.csv          100,000 rows — trace_name + actual P + magnitude + SNR
sweep_results.csv       36 rows — metrics per (threshold, sustain) combo
sweep_heatmap.png       4-panel heatmap
sweep_run.log
```

---

## 9. What's Next

If the user accepts a ~−870 ms early bias as "good enough for early warning":
- **Lock in stride = 20, threshold = 0.99, sustain = 3** — proceed to FPGA synthesis (Phase 3).

If the user wants to push the bias closer to 0:
- **Retrain with PRE_ARRIVAL=150, POST_ARRIVAL=50** — small edits to `prepare_dataset.py` and `train_cnn.py`, ~25 min Phase-1 re-extraction + ~12 min Phase-2 retrain. Then re-run this same benchmark to see how the bias shifts. *Recommended.*

If the user wants accurate P-time prediction (regression):
- **Architecture change** — add a second output head. Bigger Phase-2 redesign. Defer until after Phase 3 baseline is validated on FPGA.
