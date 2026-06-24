"""
Epicenter Side-Project — Shared utilities
=========================================
Spherical-geometry helpers (Haversine + destination point), label
transforms, and angle encoding/decoding used by both `train_epicenter.py`
and the Streamlit demo.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


# ── Per-channel normalisation + auxiliary features ──────────────────────────
def preprocess_window(window: np.ndarray, eps: float = 1e-6):
    """Per-channel renormalisation with cross-channel peaks kept as a side
    input.

    The dataset on disk is already globally peak-normalised (max |x| = 1
    across the whole 3-channel window). For the bigger v2 model we want:

    1. each channel scaled INDEPENDENTLY to roughly [-1, 1] so the conv
       trunk can learn shape features without one loud channel
       overpowering the others;
    2. the *original* per-channel peak amplitudes preserved as a small
       auxiliary input — these encode the relative N/E/Z magnitudes that
       carry the polarisation (back-azimuth) signal.

    Accepts a single window (T, 3) OR a batch (N, T, 3). Returns
    `(renormalised, channel_peaks)` with matching leading dims.
    """
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim == 2:
        peaks = np.max(np.abs(arr), axis=0)            # (3,)
        safe = np.maximum(peaks, eps)
        renorm = arr / safe[None, :]
    elif arr.ndim == 3:
        peaks = np.max(np.abs(arr), axis=1)            # (N, 3)
        safe = np.maximum(peaks, eps)
        renorm = arr / safe[:, None, :]
    else:
        raise ValueError(f"expected (T, 3) or (N, T, 3); got {arr.shape}")
    return renorm.astype(np.float32), peaks.astype(np.float32)


# ── Distance transforms ─────────────────────────────────────────────────────
# Distance is heavily right-skewed (lots of nearby quakes, few far). We
# train the model on a log-scaled, z-normalised target so MSE is well-behaved.

def encode_distance(dist_km: np.ndarray, mu: float | None = None,
                    sigma: float | None = None):
    """log10(dist + 1) → standardised. Returns (encoded, mu, sigma)."""
    log_d = np.log10(np.asarray(dist_km, dtype=np.float32) + 1.0)
    if mu is None:
        mu = float(log_d.mean())
    if sigma is None:
        sigma = float(log_d.std()) or 1.0
    return (log_d - mu) / sigma, mu, sigma


def decode_distance(z: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Invert encode_distance back to km."""
    log_d = np.asarray(z, dtype=np.float32) * sigma + mu
    return np.power(10.0, log_d) - 1.0


# ── Azimuth transforms ──────────────────────────────────────────────────────
def encode_azimuth(deg: np.ndarray):
    """Degrees → (sin, cos) on the unit circle. Avoids the 0/360 discontinuity."""
    rad = np.deg2rad(np.asarray(deg, dtype=np.float32))
    return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)


def decode_azimuth(sin_val: np.ndarray, cos_val: np.ndarray) -> np.ndarray:
    """(sin, cos) → degrees in [0, 360)."""
    deg = np.rad2deg(np.arctan2(sin_val, cos_val))
    return np.mod(deg, 360.0).astype(np.float32)


def angular_error_deg(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    """Smallest absolute angular difference in degrees, in [0, 180]."""
    diff = np.mod(np.asarray(pred_deg) - np.asarray(true_deg) + 180.0, 360.0) - 180.0
    return np.abs(diff).astype(np.float32)


# ── Spherical geometry ──────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance between (lat1, lon1) and (lat2, lon2) in km.
    All inputs in degrees, broadcastable shapes."""
    lat1 = np.deg2rad(np.asarray(lat1, dtype=np.float64))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=np.float64))
    lon1 = np.deg2rad(np.asarray(lon1, dtype=np.float64))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=np.float64))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return (EARTH_RADIUS_KM * c).astype(np.float32)


def destination_point(lat_deg, lon_deg, bearing_deg, distance_km):
    """Given a starting point, a bearing (degrees, clockwise from North)
    and a distance along the great circle, return the destination
    (lat_dest_deg, lon_dest_deg).

    Used to place the predicted epicenter: starting at the receiver,
    walking `back_azimuth_deg` for `source_distance_km` lands on the
    estimated source location.
    """
    lat1 = np.deg2rad(np.asarray(lat_deg,   dtype=np.float64))
    lon1 = np.deg2rad(np.asarray(lon_deg,   dtype=np.float64))
    brng = np.deg2rad(np.asarray(bearing_deg, dtype=np.float64))
    d_over_r = np.asarray(distance_km, dtype=np.float64) / EARTH_RADIUS_KM

    lat2 = np.arcsin(np.sin(lat1) * np.cos(d_over_r)
                     + np.cos(lat1) * np.sin(d_over_r) * np.cos(brng))
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(d_over_r) * np.cos(lat1),
        np.cos(d_over_r) - np.sin(lat1) * np.sin(lat2),
    )
    # Wrap longitude to [-180, 180]
    lon2 = np.mod(lon2 + np.pi, 2 * np.pi) - np.pi
    return np.rad2deg(lat2).astype(np.float32), np.rad2deg(lon2).astype(np.float32)
