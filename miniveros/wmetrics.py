"""Wasserstein-1 metrics on horizontal-mean temperature profiles.

For a run, the state distribution is summarised by the profile of horizontal means over water cells,
one value per level. Per level, the 1-D Wasserstein-1 distance is computed between the generated values
(n samples) and ALL true snapshots of the window (m snapshots) through the quantile functions,
    W1 = int_0^1 |F_g^{-1}(q) - F_t^{-1}(q)| dq ,
so unequal sample sizes are fine. A point prediction x (baseline) gives W1 = mean_i |x - t_i|.
The levels are combined with thickness weights dz_k / H (a thickness-weighted sum, in K).
The floor is the same distance between the first and the second half of the true window (the drift).
"""
from __future__ import annotations

import numpy as np

_Q = np.linspace(0.5 / 200, 1 - 0.5 / 200, 200)


def level_thickness(zt: np.ndarray) -> np.ndarray:
    """Approximate layer thicknesses from level centres (zt ascending, negative, index 0 = bottom)."""
    zt = np.asarray(zt, dtype=float)
    edges = np.empty(len(zt) + 1)
    edges[1:-1] = 0.5 * (zt[:-1] + zt[1:])
    edges[-1] = 0.0                                   # surface
    edges[0] = 2 * zt[0] - edges[1]                   # bottom, mirrored
    return np.diff(edges)


def profile_means(fields: np.ndarray, water: np.ndarray) -> np.ndarray:
    """(n, Z, Y, X) -> (n, Z) horizontal means over water cells per level (NaN-safe)."""
    f = np.where(water[None], fields, np.nan)
    return np.nanmean(f, axis=(2, 3))


def w1_samples(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-level W1 between sample sets a (n, Z) and b (m, Z) via quantile functions -> (Z,)."""
    return np.abs(np.quantile(a, _Q, axis=0) - np.quantile(b, _Q, axis=0)).mean(0)


def w1_point(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-level W1 between a point prediction x (Z,) and samples b (m, Z) -> (Z,)."""
    return np.abs(b - x[None]).mean(0)


def weighted_sum(w1_levels: np.ndarray, dz: np.ndarray) -> float:
    """Thickness-weighted sum over levels, weights dz_k / H (result in K)."""
    return float(np.sum(w1_levels * dz) / np.sum(dz))


def floor_w1(truth_profiles: np.ndarray) -> np.ndarray:
    """Per-level W1 between the first and second half of the (time-ordered) true window."""
    m = truth_profiles.shape[0] // 2
    return w1_samples(truth_profiles[:m], truth_profiles[m:])
