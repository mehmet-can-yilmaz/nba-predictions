r"""Synthetic NBA-shaped data, for end-to-end tests and report figures.

We sample a season of games from a *known* generative process so that we
can verify the pipeline finds something close to the true probabilities.
True data-generating model:

    latent strength s_i ~ N(0, sigma_s^2),  i = 1..n_teams
    home advantage   H  ~ constant (default 0.30 on the logit scale)
    margin_g ~ N( (s_h - s_a) + H, sigma_m^2 )
    label    = 1 if margin_g > 0 else 0

We then simulate box-score four-factor stats (FGA, FTA, OREB, TOV, points)
so the feature module has something to chew on. Market prices are simulated
by adding mean-zero noise to the true probability and re-adding a small vig.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .market import devig_two_way, prob_to_decimal


@dataclass
class SyntheticConfig:
    n_teams: int = 30
    n_seasons: int = 3
    games_per_team_per_season: int = 82
    sigma_strength: float = 0.35
    home_advantage_logit: float = 0.30   # ~57.4% home win base rate
    sigma_margin: float = 11.0
    seed: int = 42
    market_noise_logit: float = 0.20
    vig: float = 0.04


def _possessions(rng, n: int) -> np.ndarray:
    return rng.normal(99, 4, size=n).clip(80, 120)


def simulate_season(cfg: SyntheticConfig = SyntheticConfig()) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    teams = [f"T{i:02d}" for i in range(cfg.n_teams)]
    strengths = rng.normal(0, cfg.sigma_strength, size=cfg.n_teams)

    # round-robin-ish schedule per season
    rows = []
    game_id = 0
    base_date = pd.Timestamp("2022-10-18")
    for season in range(cfg.n_seasons):
        season_start = base_date + pd.Timedelta(days=365 * season)
        # build a schedule: each team plays N games, half home half away
        pairings = []
        for _ in range(cfg.games_per_team_per_season // 2):
            for i in range(cfg.n_teams):
                j = rng.integers(0, cfg.n_teams)
                while j == i:
                    j = rng.integers(0, cfg.n_teams)
                pairings.append((i, j))
        rng.shuffle(pairings)
        # one game per day per team budget — but we just spread evenly
        for k, (h, a) in enumerate(pairings):
            date = season_start + pd.Timedelta(days=k // 10)   # ~10 games/day
            true_logit = (strengths[h] - strengths[a]) + cfg.home_advantage_logit
            p_true_home = 1.0 / (1.0 + np.exp(-true_logit))
            margin = rng.normal(true_logit * 10.0, cfg.sigma_margin)  # rough scale to points
            home_score = int(round(105 + margin / 2))
            away_score = int(round(105 - margin / 2))
            # (possessions are not used downstream — feature module builds its own)

            rows.append({
                "game_id":  f"G{game_id:06d}",
                "date":     date,
                "season":   2022 + season,
                "home":     teams[h],
                "away":     teams[a],
                "home_score": home_score,
                "away_score": away_score,
                # four-factor counts (rough)
                "fga_home": rng.integers(80, 100),
                "fga_away": rng.integers(80, 100),
                "fta_home": rng.integers(15, 30),
                "fta_away": rng.integers(15, 30),
                "oreb_home": rng.integers(7, 14),
                "oreb_away": rng.integers(7, 14),
                "tov_home": rng.integers(8, 18),
                "tov_away": rng.integers(8, 18),
                "p_true_home": p_true_home,
            })
            game_id += 1
    games = pd.DataFrame(rows)
    return games, strengths


def simulate_market_prices(games: pd.DataFrame, cfg: SyntheticConfig = SyntheticConfig()) -> pd.DataFrame:
    """Attach simulated home/away decimal odds. Adds vig and logit-noise."""
    rng = np.random.default_rng(cfg.seed + 1)
    p_true = games["p_true_home"].to_numpy()
    logit = np.log(p_true / (1 - p_true)) + rng.normal(0, cfg.market_noise_logit, size=len(games))
    p_market = 1.0 / (1.0 + np.exp(-logit))
    q_market = 1.0 - p_market
    # add vig of cfg.vig split evenly
    p_market_v = p_market + cfg.vig / 2
    q_market_v = q_market + cfg.vig / 2
    out = games.copy()
    out["home_decimal"] = prob_to_decimal(p_market_v)
    out["away_decimal"] = prob_to_decimal(q_market_v)
    return out
