r"""Prediction-market and sportsbook math.

Implied probability from quoted odds
------------------------------------
- Decimal odds o (e.g. 1.80): break-even probability is 1 / o.
- American odds m:
    if m > 0:  p_imp = 100 / (m + 100)        # e.g. +150 -> 0.40
    if m < 0:  p_imp = (-m) / (-m + 100)      # e.g. -150 -> 0.60
- Prediction-market price q in [0, 1] (Kalshi YES / Polymarket): p_imp = q.

For a two-outcome market the bookmaker / market embeds a vig: p_home + p_away
typically exceeds 1. We remove it by *normalising* (the simplest rule):

    p_home_devig = p_home / (p_home + p_away),
    p_away_devig = p_away / (p_home + p_away).

This is equivalent to a *log-additive* devig only in the limit of small vig;
more sophisticated devig methods (Shin 1993, power, additive) exist but
normalisation is the standard baseline in sports analytics.

Comparing the model to the market
---------------------------------
Once both are calibrated probabilities for the same event, we can:

1. Report log-loss for each side over the same set of games (CRPS-style
   skill comparison restricted to binary outcomes).
2. Compute the Kelly-optimal bet size for the model:

       f* = (p_model * b - (1 - p_model)) / b,      b = decimal_odds - 1.

   The expected log-growth of bankroll under repeated Kelly betting is
   E[log(1 + f X)] which is maximised at f = f*. Negative f* means do not
   bet (or bet the opposite side if the market is symmetric).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Odds conversion
# ----------------------------------------------------------------------
def american_to_prob(m: float | np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    return np.where(m > 0, 100.0 / (m + 100.0), (-m) / (-m + 100.0))


def decimal_to_prob(o: float | np.ndarray) -> np.ndarray:
    return 1.0 / np.asarray(o, dtype=float)


def prob_to_decimal(p: float | np.ndarray) -> np.ndarray:
    """Fair decimal odds for probability p."""
    return 1.0 / np.asarray(p, dtype=float)


def devig_two_way(p_home: np.ndarray, p_away: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicative (normalisation) devig for a two-outcome market."""
    p_home = np.asarray(p_home, dtype=float)
    p_away = np.asarray(p_away, dtype=float)
    s = p_home + p_away
    return p_home / s, p_away / s


# ----------------------------------------------------------------------
# Kelly criterion
# ----------------------------------------------------------------------
def kelly_fraction(p_model: float | np.ndarray, decimal_odds: float | np.ndarray) -> np.ndarray:
    """f* = (b p - q) / b, with b = decimal_odds - 1, q = 1 - p."""
    p = np.asarray(p_model, dtype=float)
    o = np.asarray(decimal_odds, dtype=float)
    b = o - 1.0
    f = (b * p - (1.0 - p)) / b
    return np.where(np.isfinite(f), f, 0.0)


# ----------------------------------------------------------------------
# Backtest harness
# ----------------------------------------------------------------------
@dataclass
class BacktestResult:
    n_bets: int
    roi: float
    final_bankroll: float
    log_growth: float
    bankroll_path: np.ndarray


def kelly_backtest(
    p_model: np.ndarray,
    market_prob: np.ndarray,
    y: np.ndarray,
    *,
    fraction_of_kelly: float = 0.5,   # half-Kelly is the standard practical default
    min_edge: float = 0.02,           # only bet when |edge| exceeds 2%
    starting_bankroll: float = 1.0,
) -> BacktestResult:
    """Simulate sequential Kelly betting against quoted market probabilities.

    Bets are placed only when |p_model - market_prob| > min_edge. We bet
    on whichever side the model favours; the market price of that side
    determines the decimal odds (assumed already devigged).
    """
    p_model = np.asarray(p_model, dtype=float)
    market_prob = np.asarray(market_prob, dtype=float)
    y = np.asarray(y, dtype=float)

    bankroll = starting_bankroll
    path = [bankroll]
    n_bets = 0
    for pm, qm, yi in zip(p_model, market_prob, y):
        # always bet on the side the model prefers; outcome is its payoff
        if pm > qm:
            p_bet, q_market, outcome = pm, qm, yi
        else:
            p_bet, q_market, outcome = 1 - pm, 1 - qm, 1 - yi
        edge = p_bet - q_market
        if edge <= min_edge:
            path.append(bankroll)
            continue
        odds = 1.0 / q_market
        f = max(0.0, kelly_fraction(p_bet, odds)) * fraction_of_kelly
        stake = f * bankroll
        payoff = stake * (odds - 1.0) if outcome == 1 else -stake
        bankroll += payoff
        path.append(bankroll)
        n_bets += 1

    path_arr = np.asarray(path)
    return BacktestResult(
        n_bets=n_bets,
        roi=float(bankroll / starting_bankroll - 1.0),
        final_bankroll=float(bankroll),
        log_growth=float(np.log(bankroll / starting_bankroll)) if bankroll > 0 else float("-inf"),
        bankroll_path=path_arr,
    )


# ----------------------------------------------------------------------
# DataFrame convenience
# ----------------------------------------------------------------------
def attach_market_probs(games: pd.DataFrame) -> pd.DataFrame:
    """Take a games frame with `home_decimal`, `away_decimal` columns,
    add `p_home_market`, `p_away_market` (devigged)."""
    if not {"home_decimal", "away_decimal"}.issubset(games.columns):
        raise ValueError("games must have home_decimal and away_decimal columns")
    out = games.copy()
    ph = decimal_to_prob(out["home_decimal"].to_numpy())
    pa = decimal_to_prob(out["away_decimal"].to_numpy())
    ph_d, pa_d = devig_two_way(ph, pa)
    out["p_home_market"] = ph_d
    out["p_away_market"] = pa_d
    return out
