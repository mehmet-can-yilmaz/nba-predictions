"""Sanity tests for the metric implementations."""
from __future__ import annotations

import numpy as np
import pytest

from nba_predictions.metrics import (
    accuracy, brier_decomposition, brier_score,
    expected_calibration_error, log_loss, roc_auc,
)


def test_perfect_predictor():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = y.astype(float)
    assert accuracy(y, p) == 1.0
    assert log_loss(y, p) < 1e-6
    assert brier_score(y, p) == pytest.approx(0.0)
    assert roc_auc(y, p) == pytest.approx(1.0)
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


def test_constant_predictor_equals_base_rate():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1])
    p = np.full_like(y, 0.5, dtype=float)
    # log-loss should equal -(y_bar log 0.5 + (1-y_bar) log 0.5) = log 2
    assert log_loss(y, p) == pytest.approx(np.log(2))
    assert brier_score(y, p) == pytest.approx(0.25)


def test_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    p = rng.uniform(size=500)
    assert roc_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-9)


def test_auc_handles_ties():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5])  # all tied
    assert roc_auc(y, p) == pytest.approx(0.5)


def test_brier_decomposition_identity():
    rng = np.random.default_rng(7)
    p = rng.beta(2, 2, size=2000)
    y = (rng.uniform(size=2000) < p).astype(int)
    bd = brier_decomposition(y, p, n_bins=20)
    # REL - RES + UNC == BS within binning resolution
    bs = brier_score(y, p)
    assert bd.brier == pytest.approx(bs, abs=5e-3)
    assert bd.uncertainty == pytest.approx(np.mean(y) * (1 - np.mean(y)))
