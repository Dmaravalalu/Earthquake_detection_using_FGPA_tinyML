"""
Visualize a single STEAD trace
==============================
Plots the 3-channel waveform (Z / N / E) of one trace from the raw STEAD
HDF5, with vertical markers at the catalogued P-arrival and S-arrival
samples (when present). Useful for eyeballing what the model actually
sees before / during / after a seismic event.

Usage
-----
  # Pick a random earthquake trace
  python visualize_trace.py

  # Pick a random noise trace
  python visualize_trace.py --category noise

  # Plot a specific trace by name
  python visualize_trace.py --trace-name <trace_name>

  # Choose a different earthquake by SNR/magnitude bracket
  python visualize_trace.py --min-snr 30 --min-mag 4.0

  # Save the plot to a file (default: just show)
  python visualize_trace.py --out ./out/trace_view.png
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CSV  = Path.home() / "Downloads" / "archive" / "merge.csv"
DEFAULT_HDF5 = Path.home() / "Downloads" / "archive" / "merge.hdf5"
SAMPLE_RATE_HZ = 100
CHANNEL_NAMES = ("Vertical (Z)", "North-South (N)", "East-West (E)")
CHANNEL_COLORS = ("#c0392b", "#27ae60", "#2980b9")


def parse_snr_max(s):
    """STEAD stores snr_db as a 3-channel string like '[56.8 55.4 47.4]'."""
    if not isinstance(s, str):
        return np.nan
    try:
        vals = [float(x) for x in s.strip("[]").replace(",", " ").split()]
        return max(vals) if vals else np.nan
    except Exception:
        return np.nan


def pick_trace(df, args):
    """Return one row of df matching the user's filters."""
    if args.trace_name:
        row = df[df["trace_name"] == args.trace_name]
        if len(row) == 0:
            raise SystemExit(f"trace_name {args.trace_name!r} not found in CSV")
        return row.iloc[0]

    mask = df["trace_category"] == args.category
    if args.category == "earthquake_local":
        df["snr_max"] = df["snr_db"].map(parse_snr_max)
        mask &= df["snr_max"].notna() & (df["snr_max"] >= args.min_snr)
        if args.min_mag is not None:
            mask &= df["source_magnitude"].astype(float) >= args.min_mag
        if args.max_mag is not None:
            mask &= df["source_magnitude"].astype(float) <= args.max_mag

    candidates = df[mask]
    if len(candidates) == 0:
        raise SystemExit("No traces matched your filters — loosen them and retry.")
    print(f"  {len(candidates):,} traces match your filters; picking one at random.")
    return candidates.sample(n=1, random_state=args.seed).iloc[0]


def load_waveform(hdf5_path, trace_name):
    with h5py.File(hdf5_path, "r") as hf:
        return np.array(hf["data/" + str(trace_name)], dtype=np.float32)


def fmt_metadata(row):
    """Build a multi-line title string from the catalogue row."""
    bits = [f"trace_name: {row['trace_name']}",
            f"category: {row['trace_category']}"]
    if row["trace_category"] == "earthquake_local":
        mag = row.get("source_magnitude", np.nan)
        mag_type = row.get("source_magnitude_type", "")
        depth = row.get("source_depth_km", np.nan)
        dist  = row.get("source_distance_km", np.nan)
        snr_max = parse_snr_max(row.get("snr_db", ""))
        bits.append(f"M{mag:.1f} {mag_type}  •  depth {depth:.1f} km  •  "
                    f"epicentral dist {dist:.1f} km  •  SNR_max {snr_max:.1f} dB")
        if pd.notna(row.get("p_arrival_sample")):
            bits.append(f"P-arrival sample: {int(row['p_arrival_sample'])}  "
                        f"(t = {row['p_arrival_sample']/SAMPLE_RATE_HZ:.2f} s)")
        if pd.notna(row.get("s_arrival_sample")):
            bits.append(f"S-arrival sample: {int(row['s_arrival_sample'])}  "
                        f"(t = {row['s_arrival_sample']/SAMPLE_RATE_HZ:.2f} s)")
    return "\n".join(bits)


def plot_trace(waveform, row, out_path=None):
    n_samples = waveform.shape[0]
    t = np.arange(n_samples) / SAMPLE_RATE_HZ

    p_sample = row.get("p_arrival_sample")
    s_sample = row.get("s_arrival_sample")
    p_t = float(p_sample) / SAMPLE_RATE_HZ if pd.notna(p_sample) else None
    s_t = float(s_sample) / SAMPLE_RATE_HZ if pd.notna(s_sample) else None

    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    for ch, (ax, name, color) in enumerate(zip(axes, CHANNEL_NAMES, CHANNEL_COLORS)):
        ax.plot(t, waveform[:, ch], color=color, linewidth=0.6)
        ax.set_ylabel(f"{name}\namplitude (counts)", fontsize=9)
        ax.grid(alpha=0.25)

        if p_t is not None:
            ax.axvline(p_t, color="black", linestyle="--", linewidth=1.2,
                       label=f"P-arrival ({p_t:.2f} s)")
        if s_t is not None:
            ax.axvline(s_t, color="black", linestyle=":", linewidth=1.2,
                       label=f"S-arrival ({s_t:.2f} s)")
        if ch == 0 and (p_t is not None or s_t is not None):
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(fmt_metadata(row), fontsize=10, fontweight="bold",
                 ha="left", x=0.06, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"  Saved → {out_path}")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  default=str(DEFAULT_CSV))
    p.add_argument("--hdf5", default=str(DEFAULT_HDF5))
    p.add_argument("--trace-name", default=None,
                   help="Plot this specific trace (overrides random sampling)")
    p.add_argument("--category", default="earthquake_local",
                   choices=["earthquake_local", "noise"])
    p.add_argument("--min-snr", type=float, default=20.0,
                   help="Min SNR_max in dB (earthquake only)")
    p.add_argument("--min-mag", type=float, default=None)
    p.add_argument("--max-mag", type=float, default=None)
    p.add_argument("--seed",    type=int,   default=None,
                   help="Random seed for sampling (default: nondeterministic)")
    p.add_argument("--out",     default=None,
                   help="Save plot to this path (default: just show)")
    args = p.parse_args()

    csv_path  = Path(args.csv).expanduser()
    hdf5_path = Path(args.hdf5).expanduser()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not hdf5_path.exists():
        raise SystemExit(f"HDF5 not found: {hdf5_path}")

    print(f"  Loading catalogue from {csv_path} …")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Total rows: {len(df):,}")

    row = pick_trace(df, args)
    print(f"  Selected trace: {row['trace_name']}")

    waveform = load_waveform(hdf5_path, row["trace_name"])
    print(f"  Loaded waveform shape={waveform.shape} dtype={waveform.dtype}")

    plot_trace(waveform, row, out_path=args.out)


if __name__ == "__main__":
    main()
