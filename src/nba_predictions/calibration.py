r"""Probability calibration.

A classifier can have high AUC and still be poorly calibrated — outputting
0.9 on games it only wins 70% of. To make the predictions usable as
probabilities (and comparable to market prices) we post-process them.

Two methods, with their math:

Platt scaling
-------------
Fit a 1-D logistic regression on the model's *logits* z:

    P(Y = 1 | z) = sigma(A z + B).

A and B are the MLE of two scalars under the same BCE loss we already use
elsewhere. We optimise via `scipy.optimize.minimize` with BFGS.

Practical note (Platt 1999): to reduce overfitting on small validation
sets, replace the labels y_i by smoothed targets

    t_i = (N_+ + 1)/(N_+ + 2)  if y_i = 1,
    t_i = 1/(N_- + 2)           if y_i = 0,

where N_+, N_- are the positive/negative counts. We support this via
`use_platt_smoothing`.

Isotonic regression (pool-adjacent-violators)
---------------------------------------------
Fit a non-decreasing step function g(z) minimising

    sum_i w_i (g(z_i) - y_i)^2     s.t. g is non-decreasing.

The pool-adjacent-violators (PAV) algorithm solves this exactly in O(n).
It is more flexible than Platt but needs more validation data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import optimize

from .logistic_regression import bce_loss, log1pexp, sigmoid


def _logit(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------
@dataclass
class PlattScaler:
    A: float = 1.0
    B: float = 0.0
    use_platt_smoothing: bool = True

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattScaler":
        p = np.asarray(p, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        z = _logit(p)

        if self.use_platt_smoothing:
            n_pos = (y == 1).sum()
            n_neg = (y == 0).sum()
            t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
        else:
            t = y

        def objective(params):
            a, b = params
            logits = a * z + b
            # use weighted BCE-from-logits with smoothed targets
            loss = float(np.mean(log1pexp(logits) - t * logits))
            p_hat = sigmoid(logits)
            ga = float(np.mean((p_hat - t) * z))
            gb = float(np.mean(p_hat - t))
            return loss, np.array([ga, gb])

        res = optimize.minimize(
            objective, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B"
        )
        self.A, self.B = float(res.x[0]), float(res.x[1])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        z = _logit(np.asarray(p, dtype=float))
        return sigmoid(self.A * z + self.B)


# ---------------------------------------------------------------------------
# Isotonic regression via PAV
# ---------------------------------------------------------------------------
def pool_adjacent_violators(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Solve min sum_i w_i (g_i - y_i)^2 s.t. g_1 <= g_2 <= ... <= g_n.

    Linear-time PAV. Returns the fitted values g_i.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if w is None:
        w = np.ones(n)
    else:
        w = np.asarray(w, dtype=float)

    # block-merge using stacks
    block_vals: list[float] = []
    block_wts: list[float] = []
    block_starts: list[int] = []
    block_lens: list[int] = []
    for i in range(n):
        block_vals.append(y[i])
        block_wts.append(w[i])
        block_starts.append(i)
        block_lens.append(1)
        # merge backwards while monotonicity is violated
        while len(block_vals) >= 2 and block_vals[-2] >= block_vals[-1]:
            v2, w2 = block_vals.pop(), block_wts.pop()
            s2, l2 = block_starts.pop(), block_lens.pop()
            v1, w1 = block_vals.pop(), block_wts.pop()
            s1, l1 = block_starts.pop(), block_lens.pop()
            new_w = w1 + w2
            new_v = (w1 * v1 + w2 * v2) / new_w
            block_vals.append(new_v)
            block_wts.append(new_w)
            block_starts.append(s1)
            block_lens.append(l1 + l2)

    out = np.empty(n)
    for v, s, l in zip(block_vals, block_starts, block_lens):
        out[s : s + l] = v
    return out


@dataclass
class IsotonicCalibrator:
    """Maps raw probabilities to calibrated probabilities via PAV."""

    x_knots: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y_knots: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        p = np.asarray(p, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        order = np.argsort(p, kind="mergesort")
        p_sorted = p[order]
        y_sorted = y[order]
        g = pool_adjacent_violators(y_sorted)
        # keep only one knot per unique x (last value wins)
        uniq, idx_last = np.unique(p_sorted, return_index=False), None
        # build a thinned representation: take the first occurrence of each block
        keep = np.concatenate([[True], np.diff(g) > 0])
        # also keep block boundaries on x
        keep_x = np.concatenate([[True], np.diff(p_sorted) > 0])
        mask = keep_x | keep
        self.x_knots = p_sorted[mask]
        self.y_knots = g[mask]
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.x_knots.size == 0:
            raise RuntimeError("IsotonicCalibrator is not fitted")
        p = np.asarray(p, dtype=float)
        return np.interp(p, self.x_knots, self.y_knots, left=self.y_knots[0], right=self.y_knots[-1])


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def calibrate(
    p_train: np.ndarray,
    y_train: np.ndarray,
    p_eval: np.ndarray,
    method: Literal["platt", "isotonic"] = "platt",
):
    if method == "platt":
        cal = PlattScaler().fit(p_train, y_train)
    elif method == "isotonic":
        cal = IsotonicCalibrator().fit(p_train, y_train)
    else:
        raise ValueError(f"unknown calibration method {method!r}")
    return cal, cal.transform(p_eval)
