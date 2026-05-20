"""Sanity tests for odds conversions and Kelly."""
from __future__ import annotations

import numpy as np
import pytest

from nba_predictions.market import (
    american_to_prob, decimal_to_prob, devig_two_way,
    kelly_backtest, kelly_fraction, prob_to_decimal,
)


def test_american_to_prob_known_values():
    # +150 means risk $100 to win $150 → break-even 100/250 = 0.40
    assert american_to_prob(150) == pytest.approx(0.40)
    # -150 means risk $150 to win $100 → break-even 150/250 = 0.60
    assert american_to_prob(-150) == pytest.approx(0.60)


def test_decimal_round_trip():
    p = np.array([0.25, 0.5, 0.75])
    assert np.allclose(decimal_to_prob(prob_to_decimal(p)), p)


def test_devig_sums_to_one():
    ph, pa = devig_two_way(np.array([0.55, 0.60]), np.array([0.50, 0.45]))
    assert np.allclose(ph + pa, 1.0)


def test_kelly_zero_when_break_even():
    # If p exactly equals 1 / odds, edge = 0, Kelly = 0.
    p = 0.5
    odds = 2.0  # break-even
    assert kelly_fraction(p, odds) == pytest.approx(0.0, abs=1e-12)


def test_kelly_positive_with_edge():
    # 60% truth on a fair coin priced at 50/50 (decimal 2.0):
    f = kelly_fraction(0.6, 2.0)
    # closed form: f* = (b p - q) / b = (1*0.6 - 0.4)/1 = 0.20
    assert f == pytest.approx(0.20)


def test_kelly_backtest_runs():
    rng = np.random.default_rng(0)
    p_model = rng.uniform(0.3, 0.7, size=200)
    p_mkt = p_model + rng.normal(0, 0.02, size=200)
    y = (rng.uniform(size=200) < p_model).astype(int)
    result = kelly_backtest(p_model, p_mkt, y, min_edge=0.0)
    assert isinstance(result.roi, float)
    assert result.bankroll_path.shape == (201,)
