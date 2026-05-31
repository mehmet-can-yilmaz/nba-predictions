"""Train the pre-game baseline (ELO + logistic regression) and report.

Usage:

    python scripts/train_baseline.py [--data data/games.csv] [--method lbfgs|irls|gd]

If --data is omitted we generate synthetic games so the pipeline can be
verified end-to-end. The script prints a metrics table and saves the model
predictions to data/processed/baseline_preds.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictions import (
    calibration,
    elo as elo_mod,
    features,
    logistic_regression as lr_mod,
    market as mk,
    metrics,
    synthetic,
)


def _load_games(path: str | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if path is None:
        print("[info] no --data given; using synthetic season")
        cfg = synthetic.SyntheticConfig()
        games, _ = synthetic.simulate_season(cfg)
        games = synthetic.simulate_market_prices(games, cfg)
        return games, games[["game_id", "home_decimal", "away_decimal"]]
    games = pd.read_csv(path, parse_dates=["date"])
    return games, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None,
                        help="path to a games CSV; uses synthetic data if omitted")
    parser.add_argument("--method", choices=["lbfgs", "irls", "gd"], default="lbfgs")
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="data/processed/baseline_preds.csv")
    args = parser.parse_args()

    # 1. Load games -------------------------------------------------------
    games, _ = _load_games(args.data)
    games = games.sort_values("date").reset_index(drop=True)
    print(f"[info] {len(games)} games loaded from {args.data or 'synthetic'}")

    # 2. Run ELO ----------------------------------------------------------
    elo_run = elo_mod.EloRating(k=20.0, home_advantage=100.0)
    games = elo_run.fit_predict(games)
    games["elo_logit"] = elo_mod.elo_logit_feature(games["elo_home_pre"], games["elo_away_pre"])
    print(f"[info] ELO log-loss on full sample: {metrics.log_loss(games['p_home_elo'].to_numpy() > 0, games['p_home_elo']):.4f}")

    # 3. Rolling features -------------------------------------------------
    core_cols = ["game_id", "date", "season", "home", "away", "home_score", "away_score"]
    optional_cols = [
        "fga_home", "fga_away", "fta_home", "fta_away",
        "oreb_home", "oreb_away", "tov_home", "tov_away",
    ]
    keep = core_cols + [c for c in optional_cols if c in games.columns]
    feats = features.build_game_features(games[keep], window=args.window)
    feats = feats.merge(games[["game_id", "elo_logit", "p_home_elo"]], on="game_id")

    train, test = features.chronological_split(feats, holdout_frac=args.holdout)
    print(f"[info] train: {len(train)}   test: {len(test)}")

    feature_cols = [
        c for c in feats.columns
        if c.startswith("diff_") or c in ("rest_diff", "home_b2b", "away_b2b", "elo_logit")
    ]
    print("[info] features:", feature_cols)

    Xtr, stats = features.to_design_matrix(train, feature_cols)
    Xte, _     = features.to_design_matrix(test,  feature_cols, train_stats=stats)

    # 4. Train logistic regression ---------------------------------------
    model = lr_mod.LogisticRegression(l2=args.l2, fit_intercept=True, max_iter=300)
    model.fit(Xtr.X, Xtr.y, method=args.method)
    print(f"[info] LR converged in {len(model.history)} iterations of {args.method.upper()}")

    p_train = model.predict_proba(Xtr.X)
    p_test  = model.predict_proba(Xte.X)

    # 5. Calibrate on the train set, transform the test predictions ------
    cal, p_test_cal = calibration.calibrate(p_train, Xtr.y, p_test, method="platt")
    print(f"[info] Platt scaling fit: A={cal.A:.3f}, B={cal.B:.3f}")

    # 6. Report -----------------------------------------------------------
    print("\n=== LR pre-game model (test) ===")
    print(_fmt(metrics.report(Xte.y, p_test)))
    print("\n=== LR + Platt (test) ===")
    print(_fmt(metrics.report(Xte.y, p_test_cal)))
    print("\n=== ELO alone (test) ===")
    print(_fmt(metrics.report(test["label_home_win"].to_numpy(), test["p_home_elo"].to_numpy())))

    if "home_decimal" in games.columns:
        market = mk.attach_market_probs(games)
        market_test = market.set_index("game_id").loc[test["game_id"]]
        p_market = market_test["p_home_market"].to_numpy()
        print("\n=== Market implied (test, devigged) ===")
        print(_fmt(metrics.report(test["label_home_win"].to_numpy(), p_market)))

        bt = mk.kelly_backtest(p_test_cal, p_market, test["label_home_win"].to_numpy())
        print(f"\nKelly backtest: bets={bt.n_bets}  ROI={bt.roi:+.2%}  "
              f"log-growth={bt.log_growth:+.3f}")

    # 7. Persist predictions ---------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "game_id": test["game_id"].to_numpy(),
        "date":    test["date"].to_numpy(),
        "y":       test["label_home_win"].to_numpy(),
        "p_lr":    p_test,
        "p_lr_calibrated": p_test_cal,
        "p_elo":   test["p_home_elo"].to_numpy(),
    })
    out_df.to_csv(out_path, index=False)
    print(f"[info] wrote {out_path}")


def _fmt(d: dict) -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, int):
            lines.append(f"  {k:>22s} : {v:d}")
        else:
            lines.append(f"  {k:>22s} : {v:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
