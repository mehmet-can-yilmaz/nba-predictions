r"""Feature engineering for the pre-game model.

We build a feature table where every row is one game from the *home* team's
point of view. All features are computed using only information that was
available **strictly before** tip-off — no leakage.

Feature definitions
-------------------
Let G_t(team) denote the t-th completed game for `team` chronologically.
For window W (default 10):

    OffRtg_W(team, t)  = mean over the last W games of points scored per 100 poss
    DefRtg_W(team, t)  = mean over the last W games of points allowed per 100 poss
    NetRtg_W(team, t)  = OffRtg_W - DefRtg_W
    Pace_W(team, t)    = mean possessions per 48 minutes
    Form_W(team, t)    = mean (margin / spread) over the last W games

For each game we compute differentials (home - away) for these stats. The
single ELO feature is computed by `elo.elo_logit_feature`. Rest days are
days since each team's previous game (capped at 7).

The conventional possessions estimator (Oliver, *Basketball on Paper*):

    POSS = FGA + 0.44 * FTA - OREB + TOV.

Sum home and away to get the *game* possessions; use that to normalise per
100. We attribute 0.44 * FTA to possessions to credit a free throw trip as
~0.44 of a possession (and-1s, technicals, etc).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POSS_FTA_COEF = 0.44


def estimate_possessions(fga: float, fta: float, oreb: float, tov: float) -> float:
    """Oliver's possessions estimator."""
    return float(fga + POSS_FTA_COEF * fta - oreb + tov)


def _team_long(boxscores: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide games frame (home/away columns) to a long per-team frame.

    Expected columns include {game_id, date, season, home, away, home_score,
    away_score, plus four-factor counts for each side suffixed _home / _away}.
    """
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        rec = pd.DataFrame({
            "game_id":  boxscores["game_id"].to_numpy(),
            "date":     boxscores["date"].to_numpy(),
            "season":   boxscores["season"].to_numpy(),
            "team":     boxscores[side].to_numpy(),
            "opp":      boxscores[other].to_numpy(),
            "is_home":  int(side == "home"),
            "pts_for":  boxscores[f"{side}_score"].to_numpy(),
            "pts_against": boxscores[f"{other}_score"].to_numpy(),
            "fga":      boxscores.get(f"fga_{side}",  pd.Series(np.nan, index=boxscores.index)).to_numpy(),
            "fta":      boxscores.get(f"fta_{side}",  pd.Series(np.nan, index=boxscores.index)).to_numpy(),
            "oreb":     boxscores.get(f"oreb_{side}", pd.Series(np.nan, index=boxscores.index)).to_numpy(),
            "tov":      boxscores.get(f"tov_{side}",  pd.Series(np.nan, index=boxscores.index)).to_numpy(),
        })
        sides.append(rec)
    long = pd.concat(sides, ignore_index=True)
    long["margin"] = long["pts_for"] - long["pts_against"]
    long["win"] = (long["margin"] > 0).astype(int)
    # possessions; if any counting stats are missing we fall back to a
    # simple proxy of league-average 100 possessions
    poss = long["fga"] + POSS_FTA_COEF * long["fta"] - long["oreb"] + long["tov"]
    long["poss"] = np.where(np.isfinite(poss), poss, 100.0)
    long["off_rtg"] = 100.0 * long["pts_for"] / long["poss"].clip(lower=1)
    long["def_rtg"] = 100.0 * long["pts_against"] / long["poss"].clip(lower=1)
    return long.sort_values(["team", "date"]).reset_index(drop=True)


def _rolling_prior(s: pd.Series, window: int) -> pd.Series:
    """Trailing-window mean *not including the current row* (no leakage)."""
    return s.shift(1).rolling(window=window, min_periods=1).mean()


def _rest_days(dates: pd.Series) -> pd.Series:
    diff = dates.diff().dt.days.fillna(7).clip(upper=7)
    return diff


def build_team_rolling(boxscores: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Per-team rolling features as of the moment each game starts."""
    long = _team_long(boxscores)
    long["date"] = pd.to_datetime(long["date"])

    cols = ["off_rtg", "def_rtg", "poss", "margin", "win"]
    grouped = long.groupby("team", group_keys=False)
    for c in cols:
        long[f"{c}_roll{window}"] = grouped[c].apply(lambda s: _rolling_prior(s, window))
    long[f"net_rtg_roll{window}"] = long[f"off_rtg_roll{window}"] - long[f"def_rtg_roll{window}"]
    long["rest_days"] = grouped["date"].apply(_rest_days)
    long["back_to_back"] = (long["rest_days"] <= 1).astype(int)
    return long


def build_game_features(
    boxscores: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """Build one feature row per game, from the home team's perspective."""
    long = build_team_rolling(boxscores, window=window)
    home = long[long["is_home"] == 1].copy()
    away = long[long["is_home"] == 0].copy()
    merged = home.merge(
        away,
        on="game_id",
        suffixes=("_h", "_a"),
        validate="one_to_one",
    )

    feats = pd.DataFrame({
        "game_id": merged["game_id"],
        "date":    merged["date_h"],
        "season":  merged["season_h"],
        "home":    merged["team_h"],
        "away":    merged["team_a"],
        f"diff_off_rtg_{window}": merged[f"off_rtg_roll{window}_h"] - merged[f"off_rtg_roll{window}_a"],
        f"diff_def_rtg_{window}": merged[f"def_rtg_roll{window}_h"] - merged[f"def_rtg_roll{window}_a"],
        f"diff_net_rtg_{window}": merged[f"net_rtg_roll{window}_h"] - merged[f"net_rtg_roll{window}_a"],
        f"diff_pace_{window}":    merged[f"poss_roll{window}_h"]    - merged[f"poss_roll{window}_a"],
        f"diff_margin_{window}":  merged[f"margin_roll{window}_h"]  - merged[f"margin_roll{window}_a"],
        f"diff_winpct_{window}":  merged[f"win_roll{window}_h"]     - merged[f"win_roll{window}_a"],
        "rest_diff":       merged["rest_days_h"]   - merged["rest_days_a"],
        "home_b2b":        merged["back_to_back_h"],
        "away_b2b":        merged["back_to_back_a"],
        "label_home_win":  (merged["pts_for_h"] > merged["pts_for_a"]).astype(int),
    })
    return feats


def chronological_split(
    feats: pd.DataFrame,
    holdout_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date — never shuffle time-series data."""
    feats = feats.sort_values("date").reset_index(drop=True)
    cut = int(len(feats) * (1.0 - holdout_frac))
    return feats.iloc[:cut].copy(), feats.iloc[cut:].copy()


@dataclass
class FeatureMatrix:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


def to_design_matrix(
    feats: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label_home_win",
    standardize: bool = True,
    train_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[FeatureMatrix, tuple[np.ndarray, np.ndarray]]:
    """Z-score features using *training* mean/std. Returns updated stats."""
    X = feats[feature_cols].to_numpy(dtype=float)
    # impute missing (early-season rows where rolling window isn't full)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    if standardize:
        if train_stats is None:
            mu = X.mean(axis=0)
            sd = X.std(axis=0)
            sd[sd == 0] = 1.0
        else:
            mu, sd = train_stats
        X = (X - mu) / sd
        train_stats = (mu, sd)
    else:
        train_stats = (np.zeros(X.shape[1]), np.ones(X.shape[1]))
    y = feats[label_col].to_numpy(dtype=int)
    return FeatureMatrix(X=X, y=y, feature_names=feature_cols), train_stats
