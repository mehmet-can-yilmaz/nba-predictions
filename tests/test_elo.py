"""ELO tests: derivation correctness and SGD-equivalence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_predictions.elo import EloRating, S_DEFAULT, expected_score


def test_expected_score_symmetric():
    p_ab = expected_score(1600, 1500)
    p_ba = expected_score(1500, 1600)
    assert p_ab + p_ba == pytest.approx(1.0)


def test_400_point_difference_is_90pct():
    # Standard ELO fact: a 400 point edge ≈ 91% win probability.
    assert expected_score(1900, 1500) == pytest.approx(10 / 11, abs=1e-3)


def test_update_is_one_sgd_step():
    """An ELO update equals one stochastic-gradient step on -log-likelihood.

    With H = 0 and the MOV multiplier off, the K-step update should be
    R_A_new = R_A + K (y - p)."""
    elo = EloRating(k=20.0, home_advantage=0.0, mov_multiplier=False)
    elo.ratings["A"] = 1600
    elo.ratings["B"] = 1500
    p_home_before = expected_score(1600, 1500)
    elo.update("A", "B", 110, 100)  # A wins
    expected_a = 1600 + 20.0 * (1.0 - p_home_before)
    assert elo.ratings["A"] == pytest.approx(expected_a, abs=1e-9)
    assert elo.ratings["B"] == pytest.approx(1500 - 20.0 * (1.0 - p_home_before), abs=1e-9)


def test_zero_sum_property():
    """Total ELO is conserved by a single update (sans MOV)."""
    elo = EloRating(home_advantage=0.0, mov_multiplier=False)
    elo.ratings["A"] = 1550
    elo.ratings["B"] = 1450
    total_before = sum(elo.ratings.values())
    elo.update("A", "B", 102, 99)
    total_after = sum(elo.ratings.values())
    assert total_before == pytest.approx(total_after)


def test_fit_predict_smoke():
    games = pd.DataFrame({
        "game_id": [f"g{i}" for i in range(6)],
        "date":    pd.date_range("2024-01-01", periods=6, freq="D"),
        "season":  [2024] * 6,
        "home":    ["A", "B", "A", "C", "B", "C"],
        "away":    ["B", "A", "C", "A", "C", "B"],
        "home_score": [110, 100, 105, 99, 120, 95],
        "away_score": [100, 110, 100, 110, 100, 96],
    })
    out = EloRating().fit_predict(games)
    assert {"elo_home_pre", "elo_away_pre", "p_home_elo"}.issubset(out.columns)
    assert np.all((out["p_home_elo"] >= 0) & (out["p_home_elo"] <= 1))


def test_s_constant():
    assert S_DEFAULT == pytest.approx(400 / np.log(10))
