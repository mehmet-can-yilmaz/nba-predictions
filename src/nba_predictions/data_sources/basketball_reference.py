"""basketball-reference.com scraper stub (Angel's piece).

Sports Reference asks scrapers to respect a ~3-second crawl-delay and
identify themselves. This stub documents the request shape and parses
their HTML schedule tables; fill in the implementation when the data
volume is finalised.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

BASE = "https://www.basketball-reference.com"
CRAWL_DELAY = 3.0


def schedule_url(season: int, month: str) -> str:
    """e.g. NBA_2024_games-october.html"""
    return f"{BASE}/leagues/NBA_{season}_games-{month}.html"


def fetch_schedule(season: int, cache_dir: Path | None = None) -> pd.DataFrame:
    """Fetch and parse the season schedule from basketball-reference.

    Returns a frame with: date, home, away, home_score, away_score.

    TODO (Angel): pandas.read_html does most of the parsing, but team
    columns need to be cleaned ('Los Angeles Lakers' -> 'LAL'). Use the
    BREF_TO_CANONICAL map in id_mapping.
    """
    months = ["october", "november", "december", "january", "february",
              "march", "april", "may", "june"]
    frames = []
    for m in months:
        try:
            tables = pd.read_html(schedule_url(season, m))
            frames.append(tables[0])
            time.sleep(CRAWL_DELAY)
        except Exception:
            continue
    if not frames:
        raise RuntimeError("basketball-reference returned no schedule data")
    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "date":  pd.to_datetime(raw["Date"]),
        "home":  raw["Home/Neutral"],
        "away":  raw["Visitor/Neutral"],
        "home_score": pd.to_numeric(raw["PTS.1"], errors="coerce"),
        "away_score": pd.to_numeric(raw["PTS"],   errors="coerce"),
    })
    return out.dropna(subset=["home_score", "away_score"])
