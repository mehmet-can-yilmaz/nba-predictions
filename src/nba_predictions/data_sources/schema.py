"""Canonical schemas shared by all data adapters.

Every `fetch_*` function returns a DataFrame whose columns are a subset of
these — extra columns are allowed but the listed ones must be present.
"""
from __future__ import annotations

GAMES_COLUMNS = [
    "game_id",      # canonical (post-join) game id
    "date",         # game date (UTC)
    "season",       # int, e.g. 2024 for the 2023-24 season
    "home",         # canonical team id (3-letter abbreviation)
    "away",
    "home_score",   # final score
    "away_score",
    "fga_home", "fga_away",
    "fta_home", "fta_away",
    "oreb_home", "oreb_away",
    "tov_home",  "tov_away",
]

PBP_COLUMNS = [
    "game_id",
    "period",          # 1-4 plus 5+ for OT
    "seconds_left",    # seconds remaining in the period
    "score_home",
    "score_away",
    "event_type",      # 'shot', 'rebound', 'turnover', 'foul', 'sub', 'ft', ...
    "team_id",         # team that performed the event (canonical)
    "player_id",       # canonical player id, nullable
    "description",
]

MARKET_COLUMNS = [
    "game_id",
    "source",          # 'kalshi' | 'polymarket'
    "snapshot_time",   # when the price was observed
    "home_decimal",    # decimal odds for home, after our internal devig
    "away_decimal",
]
