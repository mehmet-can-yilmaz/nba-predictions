"""NBA.com Stats adapter.

The public stats.nba.com endpoints require browser-like headers and rate
limiting; we use them carefully and cache aggressively. This file is a
*stub* that demonstrates the request shape — replace `_get` with real HTTP
once you have a stable user-agent / proxy story.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests


BASE = "https://stats.nba.com/stats"
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}
RATE_LIMIT_SECONDS = 0.6


def _get(endpoint: str, params: dict, cache_dir: Path | None = None) -> dict:
    """Polite GET with optional on-disk JSON cache."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = endpoint + "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        cache_path = cache_dir / (cache_key.replace("/", "_") + ".json")
        if cache_path.exists():
            import json
            return json.loads(cache_path.read_text())
    url = f"{BASE}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(RATE_LIMIT_SECONDS)
    js = resp.json()
    if cache_dir is not None:
        import json
        cache_path.write_text(json.dumps(js))
    return js


def fetch_games(season: int, cache_dir: Path | None = None) -> pd.DataFrame:
    """Fetch the season schedule + final scores from leaguegamelog.

    Returns a DataFrame indexed by NBA.com game_id.  TODO: map game_id to
    the canonical id via `id_mapping`.
    """
    season_str = f"{season - 1}-{str(season)[-2:]}"   # 2024 -> "2023-24"
    js = _get(
        "leaguegamelog",
        {"Season": season_str, "SeasonType": "Regular Season", "PlayerOrTeam": "T"},
        cache_dir=cache_dir,
    )
    result = js["resultSets"][0]
    df = pd.DataFrame(result["rowSet"], columns=result["headers"])
    # NBA.com returns one row per team-game; pivot to one row per game
    g = df.pivot_table(
        index=["GAME_ID", "GAME_DATE", "SEASON_ID"],
        columns=["MATCHUP"],
        values=["TEAM_ABBREVIATION", "PTS", "FGA", "FTA", "OREB", "TOV"],
        aggfunc="first",
    )
    # NOTE: the actual reshape is more involved because MATCHUP contains both
    # 'LAL vs. BOS' (home) and 'LAL @ BOS' (away). Left as TODO for the team.
    return g.reset_index()
