"""Tests for Platt and isotonic calibration."""
from __future__ import annotations

import numpy as np
import pytest

from nba_predictions.calibration import (
    IsotonicCalibrator, PlattScaler, pool_adjacent_violators,
)


def test_pav_already_sorted():
    y = np.array([0.1, 0.2, 0.3, 0.4])
    g = pool_adjacent_violators(y)
    assert np.allclose(g, y)


def test_pav_merges_violators():
    y = np.array([0.1, 0.5, 0.3, 0.7])
    g = pool_adjacent_violators(y)
    assert g[0] == pytest.approx(0.1)
    # middle two get pooled to their mean (0.4) since 0.5 > 0.3
    assert g[1] == pytest.approx(0.4)
    assert g[2] == pytest.approx(0.4)
    assert g[3] == pytest.approx(0.7)
    assert np.all(np.diff(g) >= -1e-12)


def test_pav_monotone_invariant():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    g = pool_adjacent_violators(y)
    assert np.all(np.diff(g) >= -1e-9)


def test_platt_recovers_identity_when_calibrated():
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    ps = PlattScaler(use_platt_smoothing=False).fit(p, y)
    # With well-calibrated inputs, A ≈ 1 and B ≈ 0.
    assert ps.A == pytest.approx(1.0, abs=0.15)
    assert abs(ps.B) < 0.15


def test_isotonic_fits_monotone_data():
    rng = np.random.default_rng(0)
    p = np.linspace(0.01, 0.99, 500)
    y = (rng.uniform(size=500) < p).astype(int)
    iso = IsotonicCalibrator().fit(p, y)
    out = iso.transform(np.linspace(0, 1, 11))
    assert np.all(np.diff(out) >= -1e-9)
