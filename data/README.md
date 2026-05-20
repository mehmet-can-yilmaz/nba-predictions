# Data layout

This directory is git-tracked structurally, but not for contents.

```
data/
├── raw/             # untouched downloads (Kaggle CSVs, NBA.com JSON, B-Ref HTML)
├── interim/         # one-source-at-a-time cleaned frames
└── processed/       # final feature/label tables + model predictions
```

All three subdirectories are in `.gitignore` so the repo stays small.
Drop your raw downloads under `data/raw/<source>/`.

## How to populate

```bash
# Kaggle (requires `kaggle` CLI configured)
mkdir -p data/raw/kaggle && kaggle datasets download -d nathanlauga/nba-games -p data/raw/kaggle/ --unzip

# NBA.com — slow; cache to data/raw/nba_stats/
python -c "from nba_predictions.data_sources import nba_stats; nba_stats.fetch_games(2024, cache_dir='data/raw/nba_stats')"

# basketball-reference — please respect their 3s crawl delay
python -c "from nba_predictions.data_sources import basketball_reference as br; br.fetch_schedule(2024, cache_dir='data/raw/bref').to_csv('data/interim/bref_2024.csv')"
```
