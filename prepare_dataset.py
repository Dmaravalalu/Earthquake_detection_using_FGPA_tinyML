"""
Real-Time Earthquake Early Warning System — Dataset Preparation
===============================================================
Builds a balanced 500k-instance training set from the STEAD dataset
(merge.csv + merge.hdf5) optimized for a 1D-CNN on Artix-7 / ZedBoard.

Output
------
  X_train.npy  — shape (500000, 200, 3), dtype float16
  y_train.npy  — shape (500000,),        dtype uint8  (0=noise, 1=earthquake)

Usage
-----
  python prepare_dataset.py --csv /path/to/merge.csv --hdf5 /path/to/merge.hdf5
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_EQ = 250_000          # earthquake samples
TARGET_NOISE = 250_000       # total noise samples
WINDOW = 200                 # samples per crop (2 s @ 100 Hz)
PRE_ARRIVAL = 50             # samples before P-arrival to keep
POST_ARRIVAL = 150           # samples after P-arrival to keep
PRE_NOISE_LEN = 200          # first 200 samples for synthetic noise
MIN_P_FOR_SYNTH = 250        # p_arrival must be > this for safe pre-arrival crop
SNR_THRESHOLD = 20.0
MAG_LOW, MAG_HIGH = 2.0, 5.0
CHANNELS = 3                 # Vertical, North, East
SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_trace(hdf5_file, trace_name):
    """Return a (6000, 3) float32 array for the given trace name."""
    data = np.array(hdf5_file.get("data/" + str(trace_name)), dtype=np.float32)
    return data  # shape (6000, 3)


def crop_earthquake(trace, p_arrival):
    """Extract a 200-sample window centred around the P-arrival with offset."""
    start = int(p_arrival) - PRE_ARRIVAL
    end = start + WINDOW
    return trace[start:end, :]


def crop_pre_arrival_noise(trace):
    """Take the first 200 samples (pure ambient noise)."""
    return trace[:PRE_NOISE_LEN, :]


def normalize_window(window):
    """Per-trace peak normalization: divide by max(|x|) so values land in [-1, 1].
    Required before float16 cast — raw STEAD counts routinely exceed float16's
    65,504 max and would otherwise saturate to inf."""
    peak = np.max(np.abs(window))
    if peak > 0:
        window = window / peak
    return window


def parse_snr_max(s):
    """STEAD stores snr_db as a 3-channel string like '[56.8 55.4 47.4]'.
    Return the max of the three channels, or NaN if unparseable."""
    if not isinstance(s, str):
        return np.nan
    try:
        vals = [float(x) for x in s.strip("[]").replace(",", " ").split()]
        return max(vals) if vals else np.nan
    except Exception:
        return np.nan


# ── Main pipeline ────────────────────────────────────────────────────────────
def main(csv_path: str, hdf5_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load catalogue ────────────────────────────────────────────────────
    print("[1/6] Loading catalogue …")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"      Total rows in merge.csv: {len(df):,}")

    # ── 2. Filter earthquake candidates ──────────────────────────────────────
    print("[2/6] Filtering earthquake candidates …")
    df["snr_max"] = df["snr_db"].map(parse_snr_max)
    eq_mask = (
        (df["trace_category"] == "earthquake_local")
        & (df["snr_max"].notna())
        & (df["snr_max"] >= SNR_THRESHOLD)
        & (df["source_magnitude"].notna())
        & (df["source_magnitude"].astype(float) >= MAG_LOW)
        & (df["source_magnitude"].astype(float) <= MAG_HIGH)
        & (df["p_arrival_sample"].notna())
        & (df["p_arrival_sample"].astype(float) >= PRE_ARRIVAL)  # room for offset
    )
    eq_df = df[eq_mask].copy()
    print(f"      Earthquake candidates after SNR/mag filter: {len(eq_df):,}")

    if len(eq_df) < TARGET_EQ:
        print(f"  ⚠  Only {len(eq_df):,} quake traces available — using all of them.")
    eq_df = eq_df.sample(n=min(TARGET_EQ, len(eq_df)), random_state=SEED)
    actual_eq = len(eq_df)
    print(f"      Sampled earthquakes: {actual_eq:,}")

    # ── 3. Prepare noise roster ──────────────────────────────────────────────
    print("[3/6] Preparing noise roster …")
    # 3a. Catalogue noise traces
    noise_df = df[df["trace_category"] == "noise"].copy()
    n_catalogue_noise = len(noise_df)
    print(f"      Catalogue noise traces: {n_catalogue_noise:,}")

    # 3b. Synthetic noise from earthquake pre-arrivals
    synth_mask = (
        (df["trace_category"] == "earthquake_local")
        & (df["p_arrival_sample"].notna())
        & (df["p_arrival_sample"].astype(float) > MIN_P_FOR_SYNTH)
    )
    synth_df = df[synth_mask].copy()
    n_synth_needed = TARGET_NOISE - n_catalogue_noise
    if n_synth_needed < 0:
        n_synth_needed = 0
        noise_df = noise_df.sample(n=TARGET_NOISE, random_state=SEED)
    else:
        if len(synth_df) < n_synth_needed:
            print(f"  ⚠  Only {len(synth_df):,} synthetic noise candidates — using all.")
        synth_df = synth_df.sample(n=min(n_synth_needed, len(synth_df)),
                                   random_state=SEED)
    n_synth = len(synth_df) if n_synth_needed > 0 else 0
    actual_noise = len(noise_df) + n_synth
    print(f"      Synthetic (pre-arrival) noise: {n_synth:,}")
    print(f"      Total noise instances:         {actual_noise:,}")

    total = actual_eq + actual_noise

    # ── 4. Allocate output arrays ────────────────────────────────────────────
    print(f"[4/6] Allocating arrays for {total:,} samples …")
    X = np.zeros((total, WINDOW, CHANNELS), dtype=np.float16)
    y = np.zeros(total, dtype=np.uint8)

    # ── 5. Extract waveforms from HDF5 ──────────────────────────────────────
    print("[5/6] Extracting waveforms (this may take a while) …")
    idx = 0
    skipped = 0
    sample_eq_idx = None   # for verification plot
    sample_noise_idx = None

    with h5py.File(hdf5_path, "r") as hf:
        # -- Earthquakes --
        for _, row in tqdm(eq_df.iterrows(), total=actual_eq,
                           desc="earthquakes   ", unit="trace"):
            trace_name = row["trace_name"]
            p_arr = float(row["p_arrival_sample"])
            start = int(p_arr) - PRE_ARRIVAL
            end = start + WINDOW
            try:
                raw = load_trace(hf, trace_name)
            except Exception:
                skipped += 1
                continue
            if end > raw.shape[0] or start < 0:
                skipped += 1
                continue
            X[idx] = normalize_window(crop_earthquake(raw, p_arr)).astype(np.float16)
            y[idx] = 1
            if sample_eq_idx is None:
                sample_eq_idx = idx
            idx += 1

        eq_written = idx
        print(f"      Earthquakes written: {eq_written:,}  (skipped {skipped})")
        skipped = 0

        # -- Catalogue noise --
        for _, row in tqdm(noise_df.iterrows(), total=len(noise_df),
                           desc="catalogue noise", unit="trace"):
            trace_name = row["trace_name"]
            try:
                raw = load_trace(hf, trace_name)
            except Exception:
                skipped += 1
                continue
            if raw.shape[0] < PRE_NOISE_LEN:
                skipped += 1
                continue
            X[idx] = normalize_window(crop_pre_arrival_noise(raw)).astype(np.float16)
            y[idx] = 0
            if sample_noise_idx is None:
                sample_noise_idx = idx
            idx += 1

        cat_noise_written = idx - eq_written
        print(f"      Catalogue noise written: {cat_noise_written:,}  (skipped {skipped})")
        skipped = 0

        # -- Synthetic pre-arrival noise --
        if n_synth > 0:
            for _, row in tqdm(synth_df.iterrows(), total=n_synth,
                               desc="synthetic noise", unit="trace"):
                trace_name = row["trace_name"]
                try:
                    raw = load_trace(hf, trace_name)
                except Exception:
                    skipped += 1
                    continue
                if raw.shape[0] < PRE_NOISE_LEN:
                    skipped += 1
                    continue
                X[idx] = normalize_window(crop_pre_arrival_noise(raw)).astype(np.float16)
                y[idx] = 0
                if sample_noise_idx is None:
                    sample_noise_idx = idx
                idx += 1

            synth_written = idx - eq_written - cat_noise_written
            print(f"      Synthetic noise written: {synth_written:,}  (skipped {skipped})")

    # Trim if any traces were skipped
    X = X[:idx]
    y = y[:idx]

    # Shuffle
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(idx)
    X = X[perm]
    y = y[perm]
    # Remap sample indices after shuffle
    if sample_eq_idx is not None:
        sample_eq_idx = int(np.where(perm == sample_eq_idx)[0][0])
    if sample_noise_idx is not None:
        sample_noise_idx = int(np.where(perm == sample_noise_idx)[0][0])

    # ── 6. Save & verify ─────────────────────────────────────────────────────
    print("[6/6] Saving .npy files …")
    x_path = out / "X_train.npy"
    y_path = out / "y_train.npy"
    np.save(x_path, X)
    np.save(y_path, y)

    x_mb = x_path.stat().st_size / (1024 ** 2)
    y_mb = y_path.stat().st_size / (1024 ** 2)
    eq_count = int((y == 1).sum())
    noise_count = int((y == 0).sum())

    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Total samples : {len(y):>10,}")
    print(f"  Earthquake (1): {eq_count:>10,}")
    print(f"  Noise      (0): {noise_count:>10,}")
    print(f"  X_train shape : {X.shape}")
    print(f"  X_train dtype : {X.dtype}")
    print(f"  y_train dtype : {y.dtype}")
    print(f"  X_train size  : {x_mb:>10.1f} MB")
    print(f"  y_train size  : {y_mb:>10.1f} MB")
    print(f"  Total disk    : {x_mb + y_mb:>10.1f} MB")
    print(f"  Est. RAM      : {X.nbytes / (1024**2):>10.1f} MB")
    print("=" * 60)

    # ── Verification plot ────────────────────────────────────────────────────
    plot_verification(X, y, sample_eq_idx, sample_noise_idx, out)


def plot_verification(X, y, eq_idx, noise_idx, out_dir):
    """Plot one earthquake and one noise crop side-by-side."""
    if eq_idx is None or noise_idx is None:
        print("  (skipping plot — not enough samples)")
        return

    ch_labels = ["Vertical (Z)", "North (N)", "East (E)"]
    time = np.arange(X.shape[1]) / 100.0  # seconds at 100 Hz

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)

    for c in range(3):
        # Earthquake
        axes[0, c].plot(time, X[eq_idx, :, c].astype(np.float32),
                        linewidth=0.6, color="crimson")
        axes[0, c].axvline(x=PRE_ARRIVAL / 100.0, color="blue",
                           linestyle="--", linewidth=0.8, label="P-arrival")
        axes[0, c].set_title(f"Earthquake — {ch_labels[c]}", fontsize=9)
        if c == 0:
            axes[0, c].set_ylabel("Amplitude")
            axes[0, c].legend(fontsize=7)

        # Noise
        axes[1, c].plot(time, X[noise_idx, :, c].astype(np.float32),
                        linewidth=0.6, color="seagreen")
        axes[1, c].set_title(f"Noise — {ch_labels[c]}", fontsize=9)
        if c == 0:
            axes[1, c].set_ylabel("Amplitude")
        axes[1, c].set_xlabel("Time (s)")

    fig.suptitle("Offset-Crop Verification (50 pre / 150 post P-arrival)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plot_path = Path(out_dir) / "crop_verification.png"
    fig.savefig(plot_path, dpi=150)
    print(f"  Verification plot saved to: {plot_path}")
    plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare STEAD subset for FPGA earthquake early-warning CNN")
    parser.add_argument("--csv",  required=True, help="Path to merge.csv")
    parser.add_argument("--hdf5", required=True, help="Path to merge.hdf5")
    parser.add_argument("--out",  default=".", help="Output directory (default: cwd)")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        sys.exit(f"CSV not found: {args.csv}")
    if not Path(args.hdf5).exists():
        sys.exit(f"HDF5 not found: {args.hdf5}")

    main(args.csv, args.hdf5, args.out)
