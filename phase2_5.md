# Phase 2.5 — Interactive CPU Demo (Pre-FPGA Validation)

This document captures the interactive web demo built between Phase 2 (training) and Phase 3 (FPGA synthesis). It is intended both as a human reference and as context for future Claude Code sessions. **Read `phase1.md` and `phase2.md` first** for the full pipeline context.

---

## 1. Goal of Phase 2.5

Before burning the int8 TFLite model onto the Artix-7, we wanted to **see the model run on real STEAD traces in software** with the same calibration the FPGA will use. The demo gives:

- Visual proof the network learned the right features (per-layer activations).
- A way to show non-technical reviewers what the model does end-to-end.
- A live confusion-matrix sanity check on the actual held-out test set.
- Detection latency in milliseconds vs. the catalogued P-arrival.

If the demo classifies correctly here, the FPGA implementation can only fail for hardware reasons — not for model reasons.

---

## 2. App Stack

| Component | Choice | Why |
|---|---|---|
| Web framework | **Streamlit 1.57** | Single-file Python app, built-in widgets for ML demos, no JS required |
| Inference | **TFLite int8 interpreter** | Same binary that's going to FPGA — bit-exact pre-deployment validation |
| Activations | **Keras QAT model** (loaded via the rebuild-and-load-weights workaround from Phase 2) | Need symbolic graph access for intermediate outputs; TFLite is opaque |
| Plots | **matplotlib** | Already in the venv; high-quality static figures Streamlit renders inline |
| Data | **STEAD HDF5** at `~/Downloads/archive/` | Raw 60-second 3-channel traces |

All deps already in `.venv-tf` (added: `streamlit`). Total install footprint <10 MB.

---

## 3. Running the App

```bash
cd ~/Desktop/idp
source .venv-tf/bin/activate
TF_USE_LEGACY_KERAS=1 streamlit run app.py
```

Then open **http://localhost:8501** in any browser. Streamlit caches the catalogue, model, and activation extractor on first load (~3 s); subsequent interactions are instant.

To stop:
```bash
lsof -ti:8501 | xargs kill
```

---

## 4. Three Tabs

### 4.1 Architecture

Visual reference for what's inside the model:

- **Headline metrics** — 8,225 trainable params, 16.3 KB int8 model size, input shape, output type.
- **Flow diagram** — custom matplotlib drawing showing the layer pipeline as colour-coded blocks (Input → Reshape → 3× Conv blocks → GAP → Dense → P(quake)).
- **Per-layer table** — one row per layer with name, type, output shape, parameter count.

This is the slide you show in a project review. No interaction needed.

### 4.2 Live demo

The **end-to-end inference visualization** for one trace at a time.

**Sidebar — pick a trace** (three modes):
- *Random earthquake* — sliders for min SNR (dB), magnitude range, random seed. The catalogue is filtered before sampling.
- *Random noise* — random `trace_category == "noise"` window.
- *By trace name* — paste a specific `trace_name` from STEAD.

**Main panel layout** (top to bottom):

1. **Trace metadata** — magnitude (with type), depth, distance, SNR_max.

2. **Two interactive controls**:
   - **Stride** (1–50 samples) — sliding-window step. Smaller = finer predicted-P resolution but more inference calls (a stride of 5 = 50 ms resolution and ~1160 inference calls per 60 s trace; sub-second on CPU).
   - **Threshold** (0.05–0.95) — minimum probability for the model to be considered "firing".

3. **Four-panel plot**:
   - Top three panels: Z / N / E channels of the raw 60 s waveform.
   - Bottom panel: P(earthquake) over time from sliding-window inference.
   - Markers everywhere: **dashed** = catalogued P-arrival, **dotted** = catalogued S-arrival, **orange solid** = model's predicted P-arrival (first window where prob ≥ threshold for 3 consecutive steps).

4. **Prediction summary** — Actual P, Predicted P, **Detection latency in ms**, peak probability.
   - *Latency convention*: positive = model fired AFTER the catalogued P (normal); negative = model fired BEFORE the catalogued P (early / false-positive in the lead-up).

5. **Per-layer activations** — what the model saw inside its head for the most-confident window:
   - `conv_1 → ReLU` heatmap, **200 timesteps × 16 filters**
   - `conv_2 → ReLU` heatmap, **100 timesteps × 32 filters**
   - `conv_3 → ReLU` heatmap, **50 timesteps × 64 filters**
   - `GAP` bar chart, **64 channel means** (the actual feature vector the final Dense layer classifies)

   The window used is the one with the highest predicted probability (the model's most confident slice). Brighter = more activation.

   What to look for:
   - **Conv_1** filters are essentially short FIR filters — bright stripes typically align with high-frequency P-wave energy.
   - **Conv_2** activations are sparser; many filters are dead (a healthy sign — the model picked the few filter banks it needs).
   - **Conv_3** has clear vertical bands at the time slice containing the P-arrival — the model has localized its attention.
   - **GAP** — only ~5–10 of the 64 channels are large; those are the features the Dense classifier weighs most.

### 4.3 Test-set stats

Re-runs the int8 TFLite model on a random sample of the **same held-out test set** that produced the saved F1 = 0.9819, so the user can verify the saved metrics aren't lying.

**Controls**: Sample size (100–5,000), sample seed, decision threshold.

**Output (after clicking Run sample evaluation)**:
- Four metrics: F1, accuracy, false-alarm rate, miss rate.
- **Confusion matrix** heatmap.
- **Output probability histogram** — overlapping distributions for noise (blue) and earthquake (red), with the threshold marker. You can visually see how cleanly the two classes separate (well-separated peaks ≈ robust classifier).

The test split is recovered deterministically using the same `train_test_split(..., test_size=0.10, stratify=y, random_state=42)` call as `train_cnn.py`, so the sampled metrics converge to **F1 ≈ 0.98** as the sample size grows.

---

## 5. How the Sliding-Window Inference Works

```
def predict_along_trace(waveform, interpreter, stride):
    for each window_start in range(0, 6000-200+1, stride):
        window     = waveform[start : start+200]               # (200, 3)
        normalized = window / max(|window|)                    # per-trace peak norm
        quantized  = round(normalized / in_scale + in_zp)      # int8
        prob       = (interpreter(quantized) - out_zp) * out_scale
    return all probs
```

This is **bit-identical** to what the FPGA pipeline will do at every 100 Hz sample period:
1. Maintain a rolling 200-sample buffer of the last 2 seconds.
2. On every new sample: peak-normalize the buffer, quantize to int8, run the model, dequantize the output, threshold.
3. If output >= threshold → trigger an alert.

The "predicted P-arrival" returned by `find_first_detection()` requires **3 consecutive windows** above threshold to debounce single-sample noise. Stride 5 means 3 consecutive windows = 15 samples = 0.15 s of sustained confidence — a reasonable false-alarm-vs-latency tradeoff.

---

## 6. Reading the Demo: Three Worked Examples

### 6.1 Clean M4+ earthquake (default settings)
- Probability curve sits near 0 for the pre-event seconds, then jumps to ~1.0 within 100–200 ms of the catalogued P-arrival.
- Predicted P-arrival lands within 50–150 ms of actual.
- Confidence stays high through the S-wave coda, then decays.
- All three Conv layers light up at the P-arrival window; GAP shows ~6–8 strongly active channels.

### 6.2 Low-SNR M2 earthquake (min SNR 20, mag 2.0–2.5)
- Probability curve ramps up more gradually.
- Detection latency commonly 200–500 ms.
- Some filters in conv_1 stay dim — the model has less signal to work with.
- Sometimes detection happens 1–2 windows late or not at all if the trace is borderline; this is expected.

### 6.3 Pure noise window
- Probability curve hugs the 0–0.1 band the entire 60 s.
- "Predicted P-arrival" panel shows **no detection**.
- Conv layers show diffuse, low-amplitude activations across all filters — no localized energy to lock onto.
- If you crank the threshold down to 0.2, you may see scattered false alarms — useful for finding a personal threshold preference.

---

## 7. Files Added in Phase 2.5

```
app.py                     # Streamlit application, ~360 lines
phase2_5.md                # This document
```

`app.py` reuses two functions from `train_cnn.py` (`build_base_model`, `apply_qat`) so the architecture stays single-sourced.

---

## 8. Caveats Worth Knowing

- **Catalogue load is heavy** — the 370 MB `merge.csv` is loaded once and cached for the session. First page load takes ~3 s; subsequent navigation is instant.
- **Activation viz uses the QAT Keras model**, not the int8 TFLite — TFLite interpreters don't expose intermediate tensors. Activations are float32, very close to but not identical to the int8 values the FPGA will compute. The probability curve and confusion matrix DO use the int8 TFLite path.
- **No GPU is needed** — TFLite runs the 8K-param model in microseconds on CPU.
- **The app is read-only** — it doesn't retrain or modify any of the saved artifacts. Safe to leave running.

---

## 9. What's Next

If everything in Phase 2.5 looks right, the model is **green-lit for Phase 3 (FPGA synthesis)**.

- Phase 3a: Convert `out/eew_cnn_int8.tflite` to FPGA bitstream (HLS4ML or Vitis AI).
- Phase 3b: Bring up on-board, validate identical predictions on the same test traces.
- Phase 3c: Wire to a real-time 100 Hz sample stream, measure end-to-end latency.
