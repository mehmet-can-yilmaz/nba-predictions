# NBA Game-Winner Predictions

Group project for **Math 17 — Mathematics for Machine Learning**.
Authors: Angel · Mehmet Can · Andrew

## Headline result

On a chronological 20% hold-out of 2,640 real NBA games (2023-24 and
2024-25 seasons + playoffs, scraped from basketball-reference.com), the
L2-regularised logistic regression baseline achieves:

| Metric         | ELO   | LR (L-BFGS) | LR + Platt | NN    | NN + Platt |
|----------------|-------|-------------|------------|-------|------------|
| Accuracy       | 0.659 | 0.672       | 0.672      | 0.672 | **0.676**  |
| Log-loss       | 0.616 | **0.595**   | 0.595      | 0.602 | 0.601      |
| Brier          | 0.212 | **0.205**   | 0.205      | 0.207 | 0.207      |
| AUC            | 0.734 | **0.740**   | 0.740      | 0.734 | 0.734      |
| ECE            | 0.077 | 0.044       | 0.046      | 0.041 | **0.040**  |


## What's in the codebase

| Block                            | Module(s)                                      |
| -------------------------------- | ---------------------------------------------- |
| ELO ratings (with home edge, MOV)| `nba_predictions.elo`                          |
| Pre-game logistic regression     | `nba_predictions.logistic_regression`          |
| FF neural net w/ team embeddings | `nba_predictions.neural_net`                   |
| Rolling-window feature builder   | `nba_predictions.features`                     |
| Evaluation metrics (from scratch)| `nba_predictions.metrics`                      |
| Calibration (Platt + isotonic)   | `nba_predictions.calibration`                  |
| Market odds, devigging, Kelly    | `nba_predictions.market`                       |
| Cross-source ID join             | `nba_predictions.data_sources.id_mapping`      |
| Source adapters                  | `nba_predictions.data_sources.*`               |

## Reproducing the results

```bash
# 0. Setup (macOS needs a venv because the system Python is "externally managed")
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest

# 1. Run unit tests
pytest                                                              # 30 passed

# 2. Re-fetch the data (~1 minute, respects basketball-reference's crawl-delay)
python scripts/fetch_bref.py

# 3. Produce all paper / slide artifacts
PYTHONPATH=src python scripts/produce_report.py --data data/processed/games.csv

# 4. (Optional) Recompile paper + slides
cd paper  && pdflatex paper.tex && pdflatex paper.tex && cd ..
cd slides && pdflatex slides.tex && pdflatex slides.tex && cd ..
```

## Repository layout

```
nba-predictions/
├── README.md
├── MATH.md                        # full derivations
├── paper/
│   ├── paper.tex
│   └── paper.pdf                  # the writeup
├── slides/
│   ├── slides.tex
│   └── slides.pdf                 # presentation
├── pyproject.toml
├── src/nba_predictions/
│   ├── elo.py
│   ├── logistic_regression.py
│   ├── neural_net.py
│   ├── features.py
│   ├── metrics.py
│   ├── calibration.py
│   ├── market.py
│   ├── synthetic.py
│   └── data_sources/
│       ├── schema.py
│       ├── id_mapping.py
│       ├── nba_stats.py
│       ├── kaggle.py
│       ├── basketball_reference.py
│       └── pbpstats.py
├── scripts/
│   ├── fetch_bref.py              # downloads & cleans real schedules
│   ├── produce_report.py          # makes all figures + metrics for paper
│   ├── train_baseline.py
│   ├── train_nn.py
│   └── make_figures.py
├── figures/                       # PNG + PDF figures
├── data/processed/                # cleaned games + model preds
└── tests/                         # 30 unit tests
```

