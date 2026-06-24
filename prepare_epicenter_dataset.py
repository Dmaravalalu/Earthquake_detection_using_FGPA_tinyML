"""
Epicenter Side-Project — Dataset Preparation
============================================
Extracts a 5-second (500-sample @ 100 Hz) window around the catalogued
P-arrival from STEAD `earthquake_local` traces, together with the metadata
needed to evaluate epicenter localization (receiver lat/lon, source lat/lon,
distance, back-azimuth).

This is a SEPARATE, SOFTWARE-ONLY model — it will not run on the FPGA.
The main classifier dataset (X_train.npy / y_train.npy) is untouched.

Output
------
  out/epicenter_data.npz
    X            (N, 500, 3) float16   — Z/N/E waveform crops, per-trace peak-normalized
    dist_km      (N,)        float32   — source_distance_km
    back_az_deg  (N,)        float32   — back_azimuth_deg
    recv_lat     (N,)        float32   — receiver_latitude
    recv_lon     (N,)        float32   — receiver_longitude
    src_lat      (N,)        float32   — source_latitude
    src_lon      (N,)        float32   — source_longitude
    trace_name   (N,)        str       — for debugging / Streamlit demo

Usage
-----
  python prepare_epicenter_dataset.py \
      --csv  ~/Downloads/archive/merge.csv \
      --hdf5 ~/Downloads/archive/merge.hdf5 \
      --out  ./out
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW = 500              # 5 s @ 100 Hz — larger context than the classifier's 200
PRE_ARRIVAL = 100         # samples of ambient before P
POST_ARRIVAL = 400        # samples after P  (captures S for nearby quakes)
CHANNELS = 3

TARGET_N = 200_000        # cap on samples kept
SNR_THRESHOLD = 15.0      # slightly looser than the classifier (we need range)
MAG_LOW, MAG_HIGH = 2.0, 5.0
DIST_MAX_KM = 300.0       # restrict to "local" events — keeps the regression tractable
SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_snr_max(s):
    """STEAD stores snr_db as '[a b c]'; return max of the three channels."""
    if not isinstance(s, str):
        return np.nan
    try:
        vals = [float(x) for x in s.strip("[]").replace(",", " ").split()]
        return max(vals) if vals else np.nan
    except Exception:
        return np.nan


def normalize_window(window):
    """Per-trace peak normalize → values in [-1, 1]. Required before float16
    cast (raw STEAD counts overflow float16's ±65,504 limit)."""
    peak = float(np.max(np.abs(window)))
    if peak > 0:
        window = window / peak
    return window


def load_trace(hf, trace_name):
    return np.array(hf.get("data/" + str(trace_name)), dtype=np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────
def main(csv_path: str, hdf5_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Catalogue
    print("[1/4] Loading catalogue …")
    cols_needed = [
        "trace_name", "trace_category", "p_arrival_sample",
        "snr_db", "source_magnitude",
        "source_distance_km", "back_azimuth_deg",
        "receiver_latitude", "receiver_longitude",
        "source_latitude", "source_longitude",
    ]
    df = pd.read_csv(csv_path, usecols=cols_needed, low_memory=False)
    print(f"      Rows in catalogue: {len(df):,}")

    # 2. Filter
    print("[2/4] Filtering candidates …")
    df["snr_max"] = df["snr_db"].map(parse_snr_max)

    mask = (
        (df["trace_category"] == "earthquake_local")
        & df["snr_max"].notna() & (df["snr_max"] >= SNR_THRESHOLD)
        & df["source_magnitude"].notna()
        & (df["source_magnitude"].astype(float) >= MAG_LOW)
        & (df["source_magnitude"].astype(float) <= MAG_HIGH)
        & df["p_arrival_sample"].notna()
        & (df["p_arrival_sample"].astype(float) >= PRE_ARRIVAL)
        & df["source_distance_km"].notna()
        & (df["source_distance_km"].astype(float) > 0)
        & (df["source_distance_km"].astype(float) <= DIST_MAX_KM)
        & df["back_azimuth_deg"].notna()
        & df["receiver_latitude"].notna()
        & df["receiver_longitude"].notna()
        & df["source_latitude"].notna()
        & df["source_longitude"].notna()
    )
    cand = df[mask].copy()
    print(f"      Candidates after quality filter: {len(cand):,}")

    if len(cand) > TARGET_N:
        cand = cand.sample(n=TARGET_N, random_state=SEED)
    cand = cand.reset_index(drop=True)
    n = len(cand)
    print(f"      Selected: {n:,}")

    # 3. Extract
    print(f"[3/4] Extracting {n:,} waveforms …")
    X = np.zeros((n, WINDOW, CHANNELS), dtype=np.float16)
    dist_km = np.zeros(n, dtype=np.float32)
    back_az = np.zeros(n, dtype=np.float32)
    recv_lat = np.zeros(n, dtype=np.float32)
    recv_lon = np.zeros(n, dtype=np.float32)
    src_lat = np.zeros(n, dtype=np.float32)
    src_lon = np.zeros(n, dtype=np.float32)
    trace_names = np.empty(n, dtype=object)

    idx = 0
    skipped = 0
    with h5py.File(hdf5_path, "r") as hf:
        for _, row in tqdm(cand.iterrows(), total=n, unit="trace"):
            try:
                raw = load_trace(hf, row["trace_name"])
            except Exception:
                skipped += 1
                continue

            p = int(float(row["p_arrival_sample"]))
            start = p - PRE_ARRIVAL
            end = start + WINDOW
            if start < 0 or end > raw.shape[0]:
                skipped += 1
                continue

            window = raw[start:end, :]                          # (500, 3)
            window = normalize_window(window)
            X[idx] = window.astype(np.float16)
            dist_km[idx] = float(row["source_distance_km"])
            back_az[idx] = float(row["back_azimuth_deg"])
            recv_lat[idx] = float(row["receiver_latitude"])
            recv_lon[idx] = float(row["receiver_longitude"])
            src_lat[idx] = float(row["source_latitude"])
            src_lon[idx] = float(row["source_longitude"])
            trace_names[idx] = str(row["trace_name"])
            idx += 1

    # Trim
    X = X[:idx]
    dist_km = dist_km[:idx]
    back_az = back_az[:idx]
    recv_lat = recv_lat[:idx]
    recv_lon = recv_lon[:idx]
    src_lat = src_lat[:idx]
    src_lon = src_lon[:idx]
    trace_names = trace_names[:idx]

    # Deterministic shuffle
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(idx)
    X = X[perm]; dist_km = dist_km[perm]; back_az = back_az[perm]
    recv_lat = recv_lat[perm]; recv_lon = recv_lon[perm]
    src_lat = src_lat[perm]; src_lon = src_lon[perm]
    trace_names = trace_names[perm]

    # 4. Save
    print("[4/4] Saving …")
    out_path = out / "epicenter_data.npz"
    np.savez(
        out_path,
        X=X,
        dist_km=dist_km, back_az_deg=back_az,
        recv_lat=recv_lat, recv_lon=recv_lon,
        src_lat=src_lat, src_lon=src_lon,
        trace_name=trace_names.astype(str),
    )

    size_mb = out_path.stat().st_size / (1024 ** 2)
    print("\n" + "=" * 60)
    print("  EPICENTER DATASET SUMMARY")
    print("=" * 60)
    print(f"  Saved              : {out_path}  ({size_mb:.1f} MB)")
    print(f"  Samples (final)    : {len(X):,}  (skipped {skipped})")
    print(f"  Window             : {WINDOW} samples = {WINDOW/100:.1f} s")
    print(f"  Pre / Post P       : {PRE_ARRIVAL} / {POST_ARRIVAL}")
    print(f"  Distance range     : {dist_km.min():.1f} – {dist_km.max():.1f} km")
    print(f"  Mean distance      : {dist_km.mean():.1f} ± {dist_km.std():.1f} km")
    print(f"  Back-az range      : {back_az.min():.1f} – {back_az.max():.1f}°")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--out",  default="./out")
    args = p.parse_args()

    if not Path(args.csv).exists():
        sys.exit(f"CSV not found: {args.csv}")
    if not Path(args.hdf5).exists():
        sys.exit(f"HDF5 not found: {args.hdf5}")

    main(args.csv, args.hdf5, args.out)
