"""Cross-source ID joins (the painful but unavoidable part).

Each of our four sources uses different identifiers:

  basketball-reference :  3-letter team codes, e.g. 'LAL', plus their own
                          slug-based game ids like '202310240LAL'.
  pbpstats             :  numeric NBA.com team ids (e.g. 1610612747) and
                          NBA.com game ids ('0022300001').
  NBA.com (stats)      :  numeric team ids, 10-digit game ids.
  Kaggle               :  numeric NBA team ids; game ids match NBA.com.

This module exposes a canonical 3-letter team code as the primary key, and
a canonical (date, home, away) -> canonical_game_id mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# ---- team id ↔ canonical 3-letter code -----------------------------------
TEAM_CODE_BY_NBA_ID: dict[str, str] = {
    "1610612737": "ATL", "1610612738": "BOS", "1610612751": "BKN", "1610612766": "CHA",
    "1610612741": "CHI", "1610612739": "CLE", "1610612742": "DAL", "1610612743": "DEN",
    "1610612765": "DET", "1610612744": "GSW", "1610612745": "HOU", "1610612754": "IND",
    "1610612746": "LAC", "1610612747": "LAL", "1610612763": "MEM", "1610612748": "MIA",
    "1610612749": "MIL", "1610612750": "MIN", "1610612740": "NOP", "1610612752": "NYK",
    "1610612760": "OKC", "1610612753": "ORL", "1610612755": "PHI", "1610612756": "PHX",
    "1610612757": "POR", "1610612758": "SAC", "1610612759": "SAS", "1610612761": "TOR",
    "1610612762": "UTA", "1610612764": "WAS",
}

# basketball-reference uses 3-letter codes that differ slightly from NBA.com
# (Brooklyn pre-2012, Charlotte/Bobcats, NewOrleans/Hornets/Pelicans, ...).
# Handle current names here; historical aliases can be appended.
BREF_TO_CANONICAL: dict[str, str] = {
    "BRK": "BKN", "CHO": "CHA", "PHO": "PHX",
}


def to_canonical_team(code: str) -> str:
    """Map any source's team identifier to the canonical 3-letter code."""
    if code in TEAM_CODE_BY_NBA_ID:
        return TEAM_CODE_BY_NBA_ID[code]
    if code in BREF_TO_CANONICAL:
        return BREF_TO_CANONICAL[code]
    if len(code) == 3 and code.isalpha():
        return code.upper()
    raise KeyError(f"unknown team identifier: {code!r}")


# ---- game id mapping -----------------------------------------------------
@dataclass
class GameKey:
    """Composite key that is robust across sources: (date, home, away)."""
    date: pd.Timestamp
    home: str
    away: str

    def __post_init__(self) -> None:
        self.home = to_canonical_team(self.home)
        self.away = to_canonical_team(self.away)

    def as_str(self) -> str:
        return f"{self.date.date().isoformat()}_{self.away}@{self.home}"


def attach_canonical_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `canonical_game_id` column built from (date, home, away).

    Works for any source as long as the frame has `date`, `home`, `away`
    columns. Useful before joining frames from different sources.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["home"] = out["home"].astype(str).map(to_canonical_team)
    out["away"] = out["away"].astype(str).map(to_canonical_team)
    out["canonical_game_id"] = [
        GameKey(d, h, a).as_str()
        for d, h, a in zip(out["date"], out["home"], out["away"])
    ]
    return out


def join_market_to_games(
    games: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join market odds onto our games table by canonical game id.

    Games without a matching market row keep NaN odds — those rows are
    excluded from the "vs-market" evaluation set but stay in the
    train/test for the model itself.
    """
    g = attach_canonical_id(games)
    m = attach_canonical_id(market)
    cols = ["canonical_game_id"] + [c for c in m.columns if c not in g.columns]
    return g.merge(m[cols], on="canonical_game_id", how="left")
