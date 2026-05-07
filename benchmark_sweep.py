"""
Phase 2.6b — (Threshold × Sustain) Sweep at Stride 20
======================================================
Builds on the stride-20 result from `benchmark_strides.py` and finds the
(threshold, sustain) combo that maximises F1 under an *asymmetric* tolerance:
  - up to 500 ms BEFORE actual P-arrival is acceptable (early warning value)
  - any time AFTER actual P is too late.

Strategy: run stride-20 inference once on the same 100k earthquake traces,
save all 291 probabilities per trace to disk, then sweep the (threshold,
sustain) grid in numpy on the saved probs.

Outputs
-------
  out/sweep_probs.npy                  (100k, 291) float16 probabilities
  out/sweep_starts.npy                 (291,) window starts for stride 20
  out/sweep_meta.csv                   trace_name + actual P + magnitude + SNR
  out/sweep_results.csv                one row per (threshold, sustain) combo
  out/sweep_heatmap.png                F1 heatmap + best-combo summary
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
SAMPLE_RATE = 100
TRACE_LEN = 6000


def parse_snr_max(s):
    if not isinstance(s, str):
        return np.nan
    try:
        vals = [float(x) for x in s.strip("[]").replace(",", " ").split()]
        return max(vals) if vals else np.nan
    except Exception:
        return np.nan


# ── Inference: stride=20, batched ─────────────────────────────────────────────
def run_inference(args):
    """Compute all stride-20 probabilities for the sampled traces; save to disk."""
    print(f"  Loading TFLite model: {TFLITE_PATH}")
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

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
    print(f"  {len(candidates):,} traces match filters")

    n = min(args.n, len(candidates))
    sample = candidates.sample(n=n, random_state=args.seed).reset_index(drop=True)
    print(f"  Sampling {n:,} traces (seed={args.seed})")

    starts = np.arange(0, TRACE_LEN - WINDOW + 1, args.stride, dtype=np.int32)
    n_win = len(starts)
    print(f"  Stride {args.stride}: {n_win} windows per trace")

    # Resize TFLite input once for the per-trace batch
    interpreter.resize_tensor_input(inp["index"], (n_win, WINDOW, 3))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    probs = np.zeros((n, n_win), dtype=np.float16)
    meta_rows = []
    print("  Running batched inference …")
    t0 = time.time()
    with h5py.File(HDF5_PATH, "r") as hf:
        for i, meta in tqdm(sample.iterrows(), total=n, unit="trace"):
            trace_name = meta["trace_name"]
            try:
                wf = np.array(hf[f"data/{trace_name}"], dtype=np.float32)
            except Exception:
                continue
            if wf.shape[0] != TRACE_LEN:
                continue

            # Build window batch + per-window peak normalization
            windows = np.stack([wf[s:s + WINDOW] for s in starts])
            peaks = np.abs(windows).max(axis=(1, 2), keepdims=True)
            peaks = np.where(peaks > 0, peaks, 1.0)
            windows = windows / peaks
            q = np.clip(windows / in_scale + in_zp, -128, 127).astype(np.int8)

            interpreter.set_tensor(inp["index"], q)
            interpreter.invoke()
            out_q = interpreter.get_tensor(out["index"])
            p = (out_q[:, 0].astype(np.float32) - out_zp) * out_scale
            probs[i] = p.astype(np.float16)
            meta_rows.append({
                "trace_name": trace_name,
                "actual_p_sample": int(meta["p_arrival_sample"]),
                "magnitude": float(meta["source_magnitude"]),
                "snr_max": float(meta["snr_max"]),
            })

    elapsed = time.time() - t0
    meta_df = pd.DataFrame(meta_rows)
    np.save(ROOT / "out" / "sweep_probs.npy",  probs)
    np.save(ROOT / "out" / "sweep_starts.npy", starts)
    meta_df.to_csv(ROOT / "out" / "sweep_meta.csv", index=False)
    print(f"\n  Probs saved: {probs.shape} float16 "
          f"({probs.nbytes / (1024**2):.1f} MB), {elapsed:.1f} s wall")
    return probs, starts, meta_df


def load_inference():
    """Load saved probs / starts / meta if present (skip re-inference)."""
    probs   = np.load(ROOT / "out" / "sweep_probs.npy")
    starts  = np.load(ROOT / "out" / "sweep_starts.npy")
    meta_df = pd.read_csv(ROOT / "out" / "sweep_meta.csv")
    return probs, starts, meta_df


# ── Sweep: vectorised per (threshold, sustain) ────────────────────────────────
def first_detection_indices(probs, threshold, sustain):
    """For each row of `probs` (shape N×W), return the column index of the
    first window where prob ≥ threshold for `sustain` consecutive windows.
    Returns -1 for rows with no such window."""
    above = probs >= threshold
    n, w = above.shape
    if sustain <= 1:
        rolling = above
    else:
        rolling = above[:, :w - sustain + 1].copy()
        for k in range(1, sustain):
            rolling &= above[:, k:w - sustain + 1 + k]
    out = np.full(n, -1, dtype=np.int32)
    any_hit = rolling.any(axis=1)
    out[any_hit] = rolling[any_hit].argmax(axis=1)
    return out


def sweep(probs, starts, meta_df, args):
    actual_p = meta_df["actual_p_sample"].values.astype(np.int32)
    early_tol = int(args.early_tol_ms * SAMPLE_RATE / 1000)
    late_tol  = int(args.late_tol_ms  * SAMPLE_RATE / 1000)

    rows = []
    for thr in args.thresholds:
        for sus in args.sustains:
            hit_idx = first_detection_indices(probs.astype(np.float32), thr, sus)
            detected = hit_idx >= 0

            pred_p = np.where(detected, starts[hit_idx] + PRE_ARRIVAL, -1)
            err = np.where(detected, pred_p - actual_p, 0)

            on_time = detected & (err >= -early_tol) & (err <= late_tol)
            early   = detected & (err < -early_tol)
            late    = detected & (err > late_tol)
            miss    = ~detected

            n = len(detected)
            tp = int(on_time.sum())
            fp = int(early.sum())
            fn = int(late.sum() + miss.sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec  = tp / (tp + fn) if tp + fn else 0.0
            f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

            errs_ms = err[detected].astype(np.float32) * 1000.0 / SAMPLE_RATE

            rows.append({
                "threshold": thr,
                "sustain": sus,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "tp": tp, "fp": fp, "fn": fn,
                "detection_rate": detected.mean(),
                "on_time_rate": on_time.mean(),
                "early_fp_rate": early.mean(),
                "late_rate": late.mean(),
                "miss_rate": miss.mean(),
                "mean_signed_err_ms":
                    float(errs_ms.mean()) if len(errs_ms) else np.nan,
                "mean_abs_err_ms":
                    float(np.abs(errs_ms).mean()) if len(errs_ms) else np.nan,
                "median_abs_err_ms":
                    float(np.median(np.abs(errs_ms))) if len(errs_ms) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_sweep(results, args, out_path):
    thrs = sorted(results["threshold"].unique())
    suss = sorted(results["sustain"].unique())
    F = results.pivot(index="sustain", columns="threshold", values="f1")
    P = results.pivot(index="sustain", columns="threshold", values="precision")
    R = results.pivot(index="sustain", columns="threshold", values="recall")
    M = results.pivot(index="sustain", columns="threshold",
                      values="mean_signed_err_ms")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def heatmap(ax, mat, title, fmt="{:.3f}", cmap="viridis"):
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(thrs)))
        ax.set_xticklabels([f"{t:.2f}" for t in thrs])
        ax.set_yticks(range(len(suss)))
        ax.set_yticklabels(suss)
        ax.set_xlabel("threshold")
        ax.set_ylabel("sustain (windows)")
        ax.set_title(title)
        for i, sus in enumerate(suss):
            for j, thr in enumerate(thrs):
                v = mat.loc[sus, thr]
                if pd.notna(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=8,
                            color="white" if v < mat.values.max() * 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    heatmap(axes[0, 0], F, "F1",                     "{:.3f}", "viridis")
    heatmap(axes[0, 1], P, "Precision (1 − FP rate)","{:.3f}", "Blues")
    heatmap(axes[1, 0], R, "Recall  (1 − miss / late)","{:.3f}", "Greens")
    heatmap(axes[1, 1], M, "Mean signed err (ms)",   "{:+.0f}", "RdBu_r")

    fig.suptitle(f"Stride-20 sweep on {len(args.thresholds) * len(args.sustains)} "
                 f"(threshold × sustain) combos  —  100k earthquake traces  —  "
                 f"tolerance: −{args.early_tol_ms:.0f} ms early to "
                 f"+{args.late_tol_ms:.0f} ms late",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f"  Heatmap → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",          type=int,   default=100_000)
    p.add_argument("--stride",     type=int,   default=20)
    p.add_argument("--min-snr",    type=float, default=15.0)
    p.add_argument("--min-mag",    type=float, default=2.0)
    p.add_argument("--max-mag",    type=float, default=5.0)
    p.add_argument("--early-tol-ms", type=float, default=500.0)
    p.add_argument("--late-tol-ms",  type=float, default=0.0)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.50, 0.70, 0.85, 0.90, 0.95, 0.99])
    p.add_argument("--sustains",   type=int,   nargs="+",
                   default=[1, 2, 3, 5, 10, 15])
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--reuse",      action="store_true",
                   help="Reuse saved probs from a previous run; skip inference")
    args = p.parse_args()

    if args.reuse and (ROOT / "out" / "sweep_probs.npy").exists():
        print("  Reusing saved probabilities …")
        probs, starts, meta_df = load_inference()
    else:
        probs, starts, meta_df = run_inference(args)

    print(f"\n  Sweeping {len(args.thresholds)} thresholds × "
          f"{len(args.sustains)} sustains "
          f"= {len(args.thresholds) * len(args.sustains)} combos …")
    results = sweep(probs, starts, meta_df, args)
    results_path = ROOT / "out" / "sweep_results.csv"
    results.to_csv(results_path, index=False)

    print("\n" + "=" * 80)
    print("  TOP 10 COMBOS BY F1")
    print("=" * 80)
    top = results.sort_values("f1", ascending=False).head(10)
    cols = ["threshold", "sustain", "f1", "precision", "recall",
            "on_time_rate", "early_fp_rate", "miss_rate",
            "mean_signed_err_ms", "mean_abs_err_ms"]
    print(top[cols].to_string(index=False, float_format="%.3f"))
    best = top.iloc[0]
    print("\n" + "=" * 80)
    print(f"  Best combo: threshold = {best['threshold']:.2f}, "
          f"sustain = {int(best['sustain'])}")
    print(f"    F1            = {best['f1']:.4f}")
    print(f"    Precision     = {best['precision']:.4f}  "
          f"(of all detections, fraction landing in [-{args.early_tol_ms:.0f}, "
          f"+{args.late_tol_ms:.0f}] ms)")
    print(f"    Recall        = {best['recall']:.4f}  "
          f"(of all earthquakes, fraction caught on-time)")
    print(f"    On-time rate  = {best['on_time_rate']:.4f}")
    print(f"    Early FP rate = {best['early_fp_rate']:.4f}")
    print(f"    Miss rate     = {best['miss_rate']:.4f}")
    print(f"    Mean signed err = {best['mean_signed_err_ms']:+.1f} ms")
    print("=" * 80)

    plot_sweep(results, args, ROOT / "out" / "sweep_heatmap.png")
    print(f"\n  Results CSV → {results_path}")


if __name__ == "__main__":
    main()
