"""Download NBA season schedules from basketball-reference.com.

Respects their stated 3-second crawl delay. Writes:
    data/raw/bref_games_<seasons>.csv      — raw scraped HTML tables
    data/processed/games.csv               — cleaned canonical schema

Usage:
    python scripts/fetch_bref.py --seasons 2024 2025
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

BREF_NAME_TO_CODE = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

MONTHS = ["october", "november", "december", "january", "february",
          "march", "april", "may", "june"]


def fetch_season(season: int, crawl_delay: float = 3.5) -> pd.DataFrame:
    """Fetch every monthly schedule page for `season` (e.g. 2024 means 2023-24)."""
    frames = []
    for m in MONTHS:
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games-{m}.html"
        try:
            t = pd.read_html(url)[0]
            t["season"] = season
            frames.append(t)
            print(f"  {season} {m}: {len(t)} games")
            time.sleep(crawl_delay)
        except Exception as e:
            print(f"  {season} {m}: skip ({type(e).__name__})")
    if not frames:
        raise RuntimeError(f"basketball-reference returned no data for season {season}")
    return pd.concat(frames, ignore_index=True)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(subset=["PTS", "PTS.1"]).copy()
    raw["date"] = pd.to_datetime(raw["Date"], format="%a, %b %d, %Y")
    games = pd.DataFrame({
        "game_id":    ["G" + d.strftime("%Y%m%d") + a[:3] + h[:3]
                       for d, a, h in zip(raw["date"], raw["Visitor/Neutral"], raw["Home/Neutral"])],
        "date":       raw["date"],
        "season":     raw["season"].astype(int),
        "home":       raw["Home/Neutral"].map(BREF_NAME_TO_CODE),
        "away":       raw["Visitor/Neutral"].map(BREF_NAME_TO_CODE),
        "home_score": raw["PTS.1"].astype(int),
        "away_score": raw["PTS"].astype(int),
    })
    unmapped = games[games["home"].isna() | games["away"].isna()]
    if len(unmapped):
        raise RuntimeError(f"Unmapped team names: {unmapped[['home','away']].drop_duplicates()}")
    return games.sort_values("date").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--raw-out", default="data/raw")
    ap.add_argument("--processed-out", default="data/processed/games.csv")
    args = ap.parse_args()

    raws = []
    for s in args.seasons:
        raws.append(fetch_season(s))
    raw = pd.concat(raws, ignore_index=True)
    raw_path = Path(args.raw_out) / f"bref_games_{'_'.join(str(s) for s in args.seasons)}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    print(f"\n[info] raw -> {raw_path}")

    games = clean(raw)
    out = Path(args.processed_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(out, index=False)
    print(f"[info] cleaned ({len(games)} games) -> {out}")
    print(f"[info] home win rate: {(games['home_score'] > games['away_score']).mean():.4f}")


if __name__ == "__main__":
    main()
