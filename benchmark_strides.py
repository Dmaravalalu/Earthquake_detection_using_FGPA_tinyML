"""
Phase 2.6 — Sliding-Window Stride Benchmark
===========================================
Sweeps the sliding-window stride over a sample of STEAD earthquake traces,
measures detection quality + latency at each stride, and recommends the
best stride for FPGA deployment.

Inputs
------
  out/eew_cnn_int8.tflite  (Phase 2 model — same one going to FPGA)
  STEAD merge.csv + merge.hdf5

Outputs
-------
  out/stride_benchmark_per_trace.csv     — raw per-trace results
  out/stride_benchmark_summary.csv       — aggregated metrics per stride
  out/stride_benchmark_summary.png       — comparison plots
  out/stride_benchmark_summary.txt       — printable summary

Usage
-----
  python benchmark_strides.py --n 10000 --strides 5 10 15 20

Flags
-----
  --min-snr      default 15.0
  --min-mag      default 2.0
  --max-mag      default 5.0
  --threshold    default 0.5    decision threshold on model probability
  --sustain      default 3      consecutive windows above threshold
  --tolerance-s  default 0.5    window around actual P-arrival counted as on-time
  --seed         default 42     deterministic trace sampling
"""

import argparse
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

ROOT = Path(__file__).parent
TFLITE_PATH = ROOT / "out" / "eew_cnn_int8.tflite"
CSV_PATH    = Path.home() / "Downloads" / "archive" / "merge.csv"
HDF5_PATH   = Path.home() / "Downloads" / "archive" / "merge.hdf5"

WINDOW = 200
PRE_ARRIVAL = 50
SAMPLE_RATE = 100   # Hz
TRACE_LEN = 6000


# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_snr_max(s):
    if not isinstance(s, str):
        return np.nan
    try:
        vals = [float(x) for x in s.strip("[]").replace(",", " ").split()]
        return max(vals) if vals else np.nan
    except Exception:
        return np.nan


def find_first_detection(probs, threshold, sustain):
    """Index of the first window where probability stays ≥ threshold for
    `sustain` consecutive windows. None if no detection."""
    above = probs >= threshold
    if len(above) < sustain:
        return None
    # rolling AND across `sustain` windows
    if sustain == 1:
        rolling = above
    else:
        rolling = above[:len(above) - sustain + 1].copy()
        for k in range(1, sustain):
            rolling &= above[k:len(above) - sustain + 1 + k]
    hits = np.flatnonzero(rolling)
    return int(hits[0]) if len(hits) else None


# ── Inference (batched) ──────────────────────────────────────────────────────
def predict_full_trace(waveform, interpreter, stride_base=5):
    """Slide a 200-sample window at stride `stride_base` across the trace and
    return (starts, probs) using batched TFLite inference."""
    n = waveform.shape[0]
    starts = np.arange(0, n - WINDOW + 1, stride_base, dtype=np.int32)
    n_win = len(starts)

    # Stack and per-window peak-normalize (mirrors training-time normalization)
    windows = np.stack([waveform[s:s + WINDOW] for s in starts]).astype(np.float32)
    peaks = np.abs(windows).max(axis=(1, 2), keepdims=True)
    peaks = np.where(peaks > 0, peaks, 1.0)
    windows = windows / peaks

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    q = np.clip(windows / in_scale + in_zp, -128, 127).astype(np.int8)

    # Dynamic batch via resize_tensor_input
    if tuple(interpreter.get_input_details()[0]["shape"]) != (n_win, WINDOW, 3):
        interpreter.resize_tensor_input(inp["index"], (n_win, WINDOW, 3))
        interpreter.allocate_tensors()
        # refresh details after resize
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]

    interpreter.set_tensor(inp["index"], q)
    interpreter.invoke()
    out_q = interpreter.get_tensor(out["index"])
    probs = (out_q[:, 0].astype(np.float32) - out_zp) * out_scale
    return starts, probs


# ── Main benchmark ───────────────────────────────────────────────────────────
def benchmark(args):
    print(f"  Loading TFLite model: {TFLITE_PATH}")
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    print(f"  Loading STEAD catalogue: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False,
                     usecols=["trace_name", "trace_category",
                              "p_arrival_sample", "snr_db",
                              "source_magnitude"])
    df["snr_max"] = df["snr_db"].map(parse_snr_max)

    mask = ((df["trace_category"] == "earthquake_local")
            & df["snr_max"].notna() & (df["snr_max"] >= args.min_snr)
            & df["source_magnitude"].notna()
            & (df["source_magnitude"].astype(float) >= args.min_mag)
            & (df["source_magnitude"].astype(float) <= args.max_mag)
            & df["p_arrival_sample"].notna()
            & (df["p_arrival_sample"].astype(float) >= PRE_ARRIVAL)
            & (df["p_arrival_sample"].astype(float) <= TRACE_LEN - WINDOW))
    candidates = df[mask].copy()
    print(f"  {len(candidates):,} traces match filters "
          f"(SNR ≥ {args.min_snr}, mag ∈ [{args.min_mag}, {args.max_mag}])")

    n = min(args.n, len(candidates))
    sample = candidates.sample(n=n, random_state=args.seed).reset_index(drop=True)
    print(f"  Sampling {n:,} traces (seed={args.seed})")

    base_stride = min(args.strides)
    if any(s % base_stride for s in args.strides):
        raise SystemExit(f"All strides must be multiples of the smallest "
                         f"({base_stride}); got {args.strides}.")

    # Per-trace results — one row per (trace, stride)
    rows = []
    tol_samples = int(args.tolerance_s * SAMPLE_RATE)

    print(f"  Running batched inference on {n:,} traces …")
    t0 = time.time()
    with h5py.File(HDF5_PATH, "r") as hf:
        for _, meta in tqdm(sample.iterrows(), total=n, unit="trace"):
            trace_name = meta["trace_name"]
            actual_p = int(meta["p_arrival_sample"])
            try:
                wf = np.array(hf[f"data/{trace_name}"], dtype=np.float32)
            except Exception:
                continue
            if wf.shape[0] != TRACE_LEN:
                continue

            base_starts, base_probs = predict_full_trace(
                wf, interpreter, stride_base=base_stride)

            for stride in args.strides:
                step = stride // base_stride
                starts = base_starts[::step]
                probs  = base_probs[::step]
                hit_idx = find_first_detection(probs, args.threshold, args.sustain)

                if hit_idx is None:
                    detected = False
                    pred_p = np.nan
                    err = np.nan
                else:
                    detected = True
                    # The model was trained with P-arrival at sample 50 of
                    # its 200-sample input, so the natural "predicted P-arrival"
                    # is window_start + 50.
                    pred_p = int(starts[hit_idx]) + PRE_ARRIVAL
                    err = pred_p - actual_p   # signed: + late, - early

                rows.append({
                    "trace_name": trace_name,
                    "actual_p_sample": actual_p,
                    "magnitude": float(meta["source_magnitude"]),
                    "snr_max": float(meta["snr_max"]),
                    "stride": stride,
                    "detected": detected,
                    "predicted_p_sample": pred_p,
                    "signed_err_samples": err,
                    "n_windows": len(probs),
                })

    elapsed = time.time() - t0
    per_trace = pd.DataFrame(rows)
    per_trace_path = ROOT / "out" / "stride_benchmark_per_trace.csv"
    per_trace.to_csv(per_trace_path, index=False)
    print(f"\n  Per-trace CSV → {per_trace_path}  ({len(per_trace):,} rows, "
          f"{elapsed:.1f} s wall)")

    # Aggregate per stride
    summary = aggregate(per_trace, tol_samples)
    summary_path = ROOT / "out" / "stride_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Summary CSV → {summary_path}")

    # Pretty text summary
    txt_path = ROOT / "out" / "stride_benchmark_summary.txt"
    txt_path.write_text(format_summary(summary, args))
    print(txt_path.read_text())

    plot_summary(summary, per_trace, args, ROOT / "out")


def aggregate(per_trace, tol_samples):
    """Aggregate metrics per stride.

    Definitions (per detected trace):
      e_ms = (predicted_p_sample - actual_p_sample) * 1000 / SAMPLE_RATE
        - e > 0  : model fired LATE relative to actual P
        - e < 0  : model fired EARLY relative to actual P
      avg(e)        — signed mean (bias). Negative = systematic early-firing.
      avg(|e|)      — mean absolute error.
      median(|e|), p95(|e|) — robust + tail.

    Categorization (for F1 / accuracy):
      on-time :  |e| ≤ tolerance     — TP
      late    :  e  >  tolerance     — FN
      early   :  e  < -tolerance     — FP
      miss    :  no detection        — FN
    """
    out = []
    for stride, g in per_trace.groupby("stride"):
        n = len(g)
        det = g["detected"]
        n_det = int(det.sum())
        errs = g.loc[det, "signed_err_samples"].astype(float)

        on_time = errs.abs() <= tol_samples
        late    = errs > tol_samples
        early   = errs < -tol_samples
        n_on_time = int(on_time.sum())
        n_late    = int(late.sum())
        n_early   = int(early.sum())
        n_miss    = n - n_det

        tp = n_on_time
        fp = n_early
        fn = n_late + n_miss
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc  = tp / n if n else 0.0

        out.append({
            "stride": int(stride),
            "stride_ms": stride * 1000.0 / SAMPLE_RATE,
            "n_traces": n,
            "detection_rate": n_det / n,
            "on_time_rate": n_on_time / n,
            "early_fp_rate": n_early / n,
            "late_rate": n_late / n,
            "miss_rate": n_miss / n,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            # Error / bias statistics (over detected traces, in ms)
            "mean_signed_err_ms":  float(errs.mean())          * 1000.0 / SAMPLE_RATE if len(errs) else np.nan,
            "mean_abs_err_ms":     float(errs.abs().mean())    * 1000.0 / SAMPLE_RATE if len(errs) else np.nan,
            "median_abs_err_ms":   float(errs.abs().median())  * 1000.0 / SAMPLE_RATE if len(errs) else np.nan,
            "p95_abs_err_ms":      float(np.percentile(errs.abs(), 95)) * 1000.0 / SAMPLE_RATE if len(errs) else np.nan,
            "std_err_ms":          float(errs.std())           * 1000.0 / SAMPLE_RATE if len(errs) else np.nan,
            "windows_per_trace": int(g["n_windows"].iloc[0]),
            "inferences_per_second": SAMPLE_RATE / stride,
        })
    return pd.DataFrame(out).sort_values("stride").reset_index(drop=True)


def format_summary(summary, args):
    lines = []
    lines.append("=" * 76)
    lines.append("  STRIDE BENCHMARK — sliding-window stride sweep")
    lines.append("=" * 76)
    lines.append(f"  filters:  SNR ≥ {args.min_snr}, "
                 f"mag ∈ [{args.min_mag}, {args.max_mag}], "
                 f"trace_category = earthquake_local")
    lines.append(f"  threshold = {args.threshold}, sustain = {args.sustain}, "
                 f"on-time tolerance = ±{args.tolerance_s} s")
    lines.append("")
    cols = ["stride", "f1", "accuracy", "on_time_rate", "miss_rate",
            "early_fp_rate", "mean_signed_err_ms", "mean_abs_err_ms",
            "median_abs_err_ms", "p95_abs_err_ms", "windows_per_trace"]
    lines.append(summary[cols].to_string(index=False, float_format="%.2f"))
    best = summary.sort_values(
        ["f1", "mean_abs_err_ms"], ascending=[False, True]).iloc[0]
    lines.append("")
    lines.append(f"  Best stride by F1: {int(best['stride'])} samples "
                 f"({best['stride_ms']:.0f} ms) "
                 f"— F1 {best['f1']:.4f}, "
                 f"on-time {best['on_time_rate']*100:.2f}%, "
                 f"mean signed err {best['mean_signed_err_ms']:+.1f} ms, "
                 f"mean |err| {best['mean_abs_err_ms']:.1f} ms, "
                 f"{int(best['windows_per_trace'])} inferences/trace.")
    lines.append("=" * 76)
    return "\n".join(lines) + "\n"


def plot_summary(summary, per_trace, args, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Top-left: F1 + accuracy + on-time rate vs stride
    ax = axes[0, 0]
    x = summary["stride"]
    ax.plot(x, summary["f1"],            "o-", label="F1",           color="#2c3e50")
    ax.plot(x, summary["on_time_rate"],  "s-", label="on-time rate", color="#27ae60")
    ax.plot(x, summary["detection_rate"],"^-", label="detection rate",color="#3498db")
    ax.set_xticks(x)
    ax.set_xlabel("stride (samples)")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Detection quality vs stride")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Top-right: error stats vs stride (signed bias + abs)
    ax = axes[0, 1]
    ax.axhline(0, color="black", lw=0.5)
    ax.plot(x, summary["mean_signed_err_ms"], "o-", label="mean signed err (bias)",
            color="#8e44ad")
    ax.plot(x, summary["mean_abs_err_ms"],    "s-", label="mean |err|",
            color="#27ae60")
    ax.plot(x, summary["median_abs_err_ms"],  "^-", label="median |err|",
            color="#3498db")
    ax.plot(x, summary["p95_abs_err_ms"],     "d-", label="95th-pct |err|",
            color="#e67e22")
    ax.set_xticks(x)
    ax.set_xlabel("stride (samples)")
    ax.set_ylabel("error (ms)")
    ax.set_title("Error vs stride  (signed = bias, abs = magnitude)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-left: signed error histogram per stride
    ax = axes[1, 0]
    bins = np.linspace(-2000, 2000, 81)
    for stride in args.strides:
        g = per_trace[(per_trace["stride"] == stride) & per_trace["detected"]]
        errs_ms = g["signed_err_samples"].astype(float) * 1000 / SAMPLE_RATE
        if len(errs_ms) == 0:
            continue
        ax.hist(errs_ms, bins=bins, alpha=0.45, label=f"stride {stride}",
                histtype="stepfilled", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.7)
    ax.axvspan(-args.tolerance_s * 1000, args.tolerance_s * 1000,
               color="grey", alpha=0.10, label="on-time band")
    ax.set_xlabel("signed error (ms,  + = late,  − = early)")
    ax.set_ylabel("count")
    ax.set_title("Signed-error distribution per stride")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-right: outcome breakdown bars
    ax = axes[1, 1]
    width = 0.18
    xs = np.arange(len(summary))
    ax.bar(xs - 1.5*width, summary["on_time_rate"],   width, label="on time",       color="#27ae60")
    ax.bar(xs - 0.5*width, summary["late_rate"],      width, label="late",          color="#f39c12")
    ax.bar(xs + 0.5*width, summary["early_fp_rate"],  width, label="early (false)", color="#c0392b")
    ax.bar(xs + 1.5*width, summary["miss_rate"],      width, label="missed",        color="#7f8c8d")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"stride {s}" for s in summary["stride"]])
    ax.set_ylabel("fraction of traces")
    ax.set_title("Outcome breakdown per stride")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Sliding-window stride benchmark — "
                 f"{summary['n_traces'].iloc[0]:,} earthquake traces "
                 f"(SNR ≥ {args.min_snr}, mag {args.min_mag}–{args.max_mag})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "stride_benchmark_summary.png"
    fig.savefig(path, dpi=150)
    print(f"  Summary plot → {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n",            type=int,   default=100_000)
    p.add_argument("--strides",      type=int,   nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--min-snr",      type=float, default=15.0)
    p.add_argument("--min-mag",      type=float, default=2.0)
    p.add_argument("--max-mag",      type=float, default=5.0)
    p.add_argument("--threshold",    type=float, default=0.5)
    p.add_argument("--sustain",      type=int,   default=3)
    p.add_argument("--tolerance-s",  type=float, default=0.5)
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()
    benchmark(args)
