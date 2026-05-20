"""Train the team-embedding neural network and compare to the LR baseline.

Same data path as `train_baseline.py`. Run after the baseline so that
predictions are comparable on the same chronological holdout.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nba_predictions import (
    calibration,
    elo as elo_mod,
    features,
    metrics,
    neural_net as nn_mod,
    synthetic,
)


def _load(data_path: str | None) -> pd.DataFrame:
    if data_path is None:
        games, _ = synthetic.simulate_season(synthetic.SyntheticConfig())
        return games
    return pd.read_csv(data_path, parse_dates=["date"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--holdout", type=float, default=0.2)
    args = parser.parse_args()

    games = _load(args.data).sort_values("date").reset_index(drop=True)

    elo_run = elo_mod.EloRating()
    games = elo_run.fit_predict(games)
    games["elo_logit"] = elo_mod.elo_logit_feature(games["elo_home_pre"], games["elo_away_pre"])

    feats = features.build_game_features(games, window=10)
    feats = feats.merge(games[["game_id", "elo_logit"]], on="game_id")

    train, test = features.chronological_split(feats, holdout_frac=args.holdout)

    feature_cols = [
        c for c in feats.columns
        if c.startswith("diff_") or c in ("rest_diff", "home_b2b", "away_b2b", "elo_logit")
    ]
    Xtr, stats = features.to_design_matrix(train, feature_cols)
    Xte, _ = features.to_design_matrix(test, feature_cols, train_stats=stats)

    teams = sorted(set(games["home"]).union(games["away"]))
    team_ix = {t: i for i, t in enumerate(teams)}
    h_tr = train["home"].map(team_ix).to_numpy()
    a_tr = train["away"].map(team_ix).to_numpy()
    h_te = test["home"].map(team_ix).to_numpy()
    a_te = test["away"].map(team_ix).to_numpy()

    # set aside an inner validation split for early stopping
    n_train = len(Xtr.X)
    cut = int(n_train * 0.85)
    model = nn_mod.TeamEmbeddingNN(
        n_teams=len(teams),
        n_features=Xtr.X.shape[1],
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    model.fit(
        Xtr.X[:cut], h_tr[:cut], a_tr[:cut], Xtr.y[:cut],
        X_val=Xtr.X[cut:], home_val=h_tr[cut:], away_val=a_tr[cut:], y_val=Xtr.y[cut:],
        epochs=args.epochs, lr=args.lr, verbose=True,
    )

    p_train = model.predict_proba(Xtr.X, h_tr, a_tr)
    p_test  = model.predict_proba(Xte.X, h_te, a_te)
    _, p_test_cal = calibration.calibrate(p_train, Xtr.y, p_test, method="platt")

    print("\n=== NN (test, raw) ===")
    _print(metrics.report(Xte.y, p_test))
    print("\n=== NN + Platt (test) ===")
    _print(metrics.report(Xte.y, p_test_cal))


def _print(d):
    for k, v in d.items():
        print(f"  {k:>22s} : {v:.4f}" if isinstance(v, float) else f"  {k:>22s} : {v}")


if __name__ == "__main__":
    main()
