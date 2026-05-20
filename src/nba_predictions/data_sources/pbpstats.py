"""pbpstats.com adapter stub (Angel's piece).

pbpstats exposes a free JSON API at api.pbpstats.com.  Use it for
possession-level data (each row = one possession with start/end time,
points, lineups, etc.). Endpoints we care about:

    /possessions/team/?Season=2024-25&SeasonType=Regular Season&TeamId=...
    /possessions/game/?GameId=0022400001
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.pbpstats.com"
RATE_LIMIT_SECONDS = 0.5


def fetch_game_possessions(game_id: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """One row per possession for a single NBA.com game id."""
    url = f"{BASE}/get-game-stats"
    params = {"Type": "Possession", "GameId": game_id}
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_dir / f"possessions_{game_id}.json"
        if cp.exists():
            import json
            return pd.DataFrame(json.loads(cp.read_text()))
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(RATE_LIMIT_SECONDS)
    rows = resp.json().get("results", [])
    if cache_dir is not None:
        import json
        cp.write_text(json.dumps(rows))
    return pd.DataFrame(rows)
