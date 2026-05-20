r"""Evaluation metrics, derived from first principles.

We avoid sklearn's `metrics` module here on purpose. Computing each metric
ourselves makes the math visible in the code and matches the way the
report should present them.

Metrics implemented
-------------------
- accuracy
- binary cross-entropy (log-loss): -1/n sum y log p + (1-y) log(1-p)
- Brier score: 1/n sum (p - y)^2  (plus a Murphy decomposition)
- AUC via Mann-Whitney U
- Expected calibration error (ECE) with equal-width bins
- A reliability diagram (matplotlib) for the writeup

Brier decomposition (Murphy 1973)
---------------------------------
With predictions bucketed into K bins of size n_k, mean prediction p_bar_k,
and empirical rate y_bar_k, and overline y the global rate,

    BS = REL - RES + UNC
    REL = (1/n) sum_k n_k (p_bar_k - y_bar_k)^2     # smaller is better
    RES = (1/n) sum_k n_k (y_bar_k - y_bar)^2       # larger is better
    UNC = y_bar (1 - y_bar)                          # fixed by the data

REL measures calibration, RES measures skill, UNC is the irreducible
variance of the labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


_EPS = 1e-12


def _check(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError("y and p shapes must match")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be binary {0,1}")
    return y, p


def accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    y, p = _check(y, p)
    return float(np.mean((p >= threshold).astype(int) == y))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean BCE. Inputs are probabilities, clipped for numerical safety."""
    y, p = _check(y, p)
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2))


@dataclass
class BrierDecomposition:
    reliability: float   # REL — calibration error, smaller is better
    resolution: float    # RES — skill, larger is better
    uncertainty: float   # UNC — irreducible
    brier: float         # REL - RES + UNC


def brier_decomposition(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> BrierDecomposition:
    """Murphy's three-component decomposition of the Brier score."""
    y, p = _check(y, p)
    n = len(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins; clip 1.0 into the last bin
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, n_bins - 1)
    y_bar = y.mean()

    rel = 0.0
    res = 0.0
    for k in range(n_bins):
        mask = idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        p_bar_k = p[mask].mean()
        y_bar_k = y[mask].mean()
        rel += nk * (p_bar_k - y_bar_k) ** 2
        res += nk * (y_bar_k - y_bar) ** 2

    rel /= n
    res /= n
    unc = y_bar * (1 - y_bar)
    return BrierDecomposition(rel, res, unc, rel - res + unc)


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    r"""Area under the ROC curve via the Mann-Whitney U identity.

    AUC = Pr(p(X_+) > p(X_-)) + 1/2 Pr(p(X_+) = p(X_-))
        = U / (n_+ n_-),  with U from rank statistics.
    """
    y, p = _check(y, p)
    pos_mask = y == 1
    n_pos = int(pos_mask.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC undefined when one class is empty")
    # average ranks handle ties
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks across ties
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg
        i = j + 1
    rank_sum_pos = ranks[pos_mask].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE = sum_k (n_k / n) |y_bar_k - p_bar_k|."""
    y, p = _check(y, p)
    n = len(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, n_bins - 1)
    ece = 0.0
    for k in range(n_bins):
        mask = idx == k
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def reliability_curve(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (bin_centers, empirical_rates, bin_counts) for plotting."""
    y, p = _check(y, p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, n_bins - 1)
    rates = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        mask = idx == k
        counts[k] = int(mask.sum())
        if counts[k] > 0:
            rates[k] = y[mask].mean()
    return centers, rates, counts


def report(y: np.ndarray, p: np.ndarray) -> dict:
    """One-call summary used by `scripts/evaluate.py`."""
    bd = brier_decomposition(y, p)
    return {
        "n": int(len(y)),
        "base_rate": float(np.mean(y)),
        "accuracy": accuracy(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier_score(y, p),
        "brier_reliability": bd.reliability,
        "brier_resolution": bd.resolution,
        "brier_uncertainty": bd.uncertainty,
        "auc": roc_auc(y, p),
        "ece": expected_calibration_error(y, p),
    }
