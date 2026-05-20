# NBA Game-Winner Predictions

Group project for **Math 17 — Mathematics for Machine Learning**.

Authors: Angel · Mehmet Can · Andrew

## What this is

A small, self-contained codebase for predicting NBA game winners from
pre-game features, and (eventually) comparing the predictions to the prices
quoted on Kalshi / Polymarket. The emphasis is *mathematical*: every model
is implemented from primitives (numpy / scipy) with the derivation written
into the module docstring. We use `sklearn` only as an independent
reference in the tests.

The codebase covers all four blocks of the proposal:

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
| Source adapters (stubs)          | `nba_predictions.data_sources.*`               |

All the derivation lives in the docstring at the top of each file and in
[`MATH.md`](MATH.md).

## Quick start

```bash
# 1. Install
python -m pip install -e .

# 2. Run unit tests (gradient checks, sklearn cross-checks, identities)
pytest

# 3. Run end-to-end pipeline on synthetic data
PYTHONPATH=src python scripts/train_baseline.py        # ELO + LR + Platt
PYTHONPATH=src python scripts/train_nn.py              # NN with team embeddings
PYTHONPATH=src python scripts/make_figures.py          # reliability + loss curves
```

On synthetic data (3 seasons, 3690 games) the LR baseline matches the
simulated true probabilities to within Brier reliability ≤ 0.001 and reaches
~60% test accuracy — about what you'd expect when the irreducible
uncertainty term in the Brier decomposition is ≈ 0.24.

## Plugging in real data

Each adapter under `src/nba_predictions/data_sources/` returns DataFrames
that conform to the schema in `data_sources/schema.py`. They are stubs
that document the request shape; once a teammate fills them in, the
training scripts work unchanged:

```bash
PYTHONPATH=src python scripts/train_baseline.py --data data/processed/games_2024.csv
```

`id_mapping.attach_canonical_id` joins frames from different sources using
the composite key `(date, home, away)`, which is robust to NBA.com vs.
basketball-reference vs. pbpstats vs. Kaggle id differences.

## Repository layout

```
nba-predictions/
├── README.md                      # this file
├── MATH.md                        # detailed math derivations for the writeup
├── pyproject.toml                 # `pip install -e .` works
├── requirements.txt
├── src/nba_predictions/
│   ├── elo.py
│   ├── logistic_regression.py
│   ├── neural_net.py
│   ├── features.py
│   ├── metrics.py
│   ├── calibration.py
│   ├── market.py
│   ├── synthetic.py               # fake season for E2E tests
│   └── data_sources/
│       ├── schema.py
│       ├── id_mapping.py          # cross-source joins (Mehmet Can)
│       ├── nba_stats.py           # NBA.com   (Mehmet Can)
│       ├── kaggle.py              # Kaggle    (Mehmet Can)
│       ├── basketball_reference.py# B-Ref     (Angel)
│       └── pbpstats.py            # pbpstats  (Angel)
├── scripts/
│   ├── train_baseline.py
│   ├── train_nn.py
│   └── make_figures.py
└── tests/                         # gradient checks, identities, sklearn cross-checks
```

## Who does what (per proposal)

| Person     | Code surface                                              |
| ---------- | --------------------------------------------------------- |
| Angel      | `data_sources/basketball_reference.py`, `data_sources/pbpstats.py`, `features.py` |
| Mehmet Can | `data_sources/nba_stats.py`, `data_sources/kaggle.py`, `data_sources/id_mapping.py`, `data_sources/schema.py` |
| Andrew     | `logistic_regression.py`, `neural_net.py`, `metrics.py`, `calibration.py`, `market.py`, scripts under `scripts/` |
