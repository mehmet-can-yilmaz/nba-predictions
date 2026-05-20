"""Kaggle adapter.

Kaggle distributes several NBA datasets via the `kaggle` CLI. We assume the
user has run e.g.

    kaggle datasets download -d nathanlauga/nba-games -p data/raw/kaggle/

and unzipped it. This loader normalises the resulting CSVs into our schema.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_games(raw_dir: str | Path) -> pd.DataFrame:
    """Load the nathanlauga/nba-games `games.csv` and normalise."""
    raw_dir = Path(raw_dir)
    csv = raw_dir / "games.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"Expected {csv}. Run `kaggle datasets download -d nathanlauga/nba-games`."
        )
    df = pd.read_csv(csv, parse_dates=["GAME_DATE_EST"])
    out = pd.DataFrame({
        "game_id":  df["GAME_ID"].astype(str),
        "date":     df["GAME_DATE_EST"],
        "season":   df["SEASON"].astype(int),
        "home":     df["HOME_TEAM_ID"].astype(str),     # numeric NBA team ids
        "away":     df["VISITOR_TEAM_ID"].astype(str),
        "home_score": df["PTS_home"],
        "away_score": df["PTS_away"],
        "fga_home": df.get("FGA_home", float("nan")),
        "fga_away": df.get("FGA_away", float("nan")),
        "fta_home": df.get("FTA_home", float("nan")),
        "fta_away": df.get("FTA_away", float("nan")),
        "oreb_home": df.get("OREB_home", float("nan")),
        "oreb_away": df.get("OREB_away", float("nan")),
        "tov_home":  df.get("TOV_home",  float("nan")),
        "tov_away":  df.get("TOV_away",  float("nan")),
    })
    return out
