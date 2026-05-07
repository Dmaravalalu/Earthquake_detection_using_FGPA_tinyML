# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Real-time **earthquake early-warning** system. The eventual target is a 1D-CNN deployed on **Artix-7 / ZedBoard** FPGA hardware; this directory currently contains only the **dataset preparation** stage that feeds that model.

Reference material lives alongside the code:
- `synopsis-2.pdf` — project synopsis
- `ACCESS2947848.pdf` — IEEE Access reference paper
- `dataset.md` — placeholder for dataset documentation (not yet filled in)

## Running the pipeline

`prepare_dataset.py` is the only executable. It consumes the STEAD dataset (`merge.csv` + `merge.hdf5`, not in repo — must be supplied) and emits training arrays:

```bash
python prepare_dataset.py --csv /path/to/merge.csv --hdf5 /path/to/merge.hdf5 --out ./out
```

Outputs (in `--out`):
- `X_train.npy` — `(N, 200, 3)` `float16` — 2 s windows at 100 Hz, 3 channels (Z/N/E)
- `y_train.npy` — `(N,)` `uint8` — `1`=earthquake, `0`=noise
- `crop_verification.png` — sanity plot of one quake + one noise window

Dependencies: `numpy`, `pandas`, `h5py`, `matplotlib`. There is no requirements file, lockfile, test suite, or linter configured.

## Pipeline design (what's load-bearing)

The constants at the top of `prepare_dataset.py` encode the modelling decisions and should not be changed casually:

- **500k balanced set**: `TARGET_EQ = TARGET_NOISE = 250_000`. The CNN expects this shape/balance.
- **Window = 200 samples (2 s @ 100 Hz)** with **50 pre / 150 post** P-arrival offset. The offset is deliberate — the model must learn to fire *just after* P-arrival, not centred on it. Changing `PRE_ARRIVAL` / `POST_ARRIVAL` invalidates the verification plot's interpretation.
- **Quality filter**: `snr_db ≥ 20`, magnitude in `[2.0, 5.0]`. Tightening cuts sample count; loosening pulls in dirty traces.
- **Two-source noise strategy**: catalogue noise (`trace_category == "noise"`) is used first; the shortfall to reach 250k is filled with **synthetic noise** = pre-arrival segments of earthquake traces where `p_arrival_sample > 250` (guarantees the first 200 samples are genuinely pre-event). This is why the script reads earthquake rows twice.
- **`float16` storage** is intentional — halves disk/RAM footprint for FPGA-bound training. Cast to `float32` for plotting/analysis.
- **`SEED = 42`** drives every sample/shuffle call; reruns are deterministic.

Skipped traces (HDF5 read failures, out-of-bounds crops) are counted and the output arrays are trimmed via `X = X[:idx]` before the final shuffle, so the realised `N` may be slightly below `TARGET_EQ + TARGET_NOISE`.
