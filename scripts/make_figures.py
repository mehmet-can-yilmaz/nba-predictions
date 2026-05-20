"""Generate writeup figures: reliability diagrams, loss curves, ELO traces.

Saves PNGs into `figures/`. Run *after* `train_baseline.py`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nba_predictions import (
    elo as elo_mod,
    features,
    logistic_regression as lr_mod,
    market as mk,
    metrics,
    synthetic,
)


def reliability_plot(y, p, title, path):
    centers, rates, counts = metrics.reliability_curve(y, p, n_bins=10)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    mask = counts > 0
    ax.plot(centers[mask], rates[mask], "-o", label="empirical")
    for c, r, n in zip(centers, rates, counts):
        if n > 0:
            ax.annotate(str(n), (c, r), fontsize=7, alpha=0.6)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical frequency")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def loss_curve(history, title, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(history, "-")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def elo_trace(games, top_n=6, path="figures/elo_trace.png"):
    """Plot the ELO trajectories of the top-N teams by final rating."""
    elo_run = elo_mod.EloRating()
    teams = sorted(set(games["home"]).union(games["away"]))
    traces = {t: [] for t in teams}
    games = games.sort_values("date").reset_index(drop=True)
    last_season = None
    for _, row in games.iterrows():
        if last_season is not None and row.season != last_season:
            elo_run.regress_to_mean()
        last_season = row.season
        for t in teams:
            traces[t].append(elo_run.get(t))
        elo_run.update(row.home, row.away, row.home_score, row.away_score)
    final = {t: traces[t][-1] for t in teams}
    top = sorted(final, key=final.get, reverse=True)[:top_n]
    fig, ax = plt.subplots(figsize=(7, 4))
    for t in top:
        ax.plot(traces[t], label=f"{t} ({final[t]:.0f})", linewidth=1)
    ax.axhline(1500, color="black", linestyle=":", linewidth=0.6)
    ax.set_xlabel("game index"); ax.set_ylabel("ELO rating")
    ax.set_title(f"ELO trajectories — top {top_n} by final rating")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--out", type=str, default="figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.data is None:
        games, _ = synthetic.simulate_season(synthetic.SyntheticConfig())
        games = synthetic.simulate_market_prices(games, synthetic.SyntheticConfig())
    else:
        games = pd.read_csv(args.data, parse_dates=["date"])

    # ELO trace
    elo_trace(games, path=str(out / "elo_trace.png"))

    # Train baseline LR; capture loss curve
    elo_run = elo_mod.EloRating()
    games = elo_run.fit_predict(games)
    games["elo_logit"] = elo_mod.elo_logit_feature(games["elo_home_pre"], games["elo_away_pre"])
    feats = features.build_game_features(games, window=10).merge(
        games[["game_id", "elo_logit", "p_home_elo"]], on="game_id"
    )
    train, test = features.chronological_split(feats, holdout_frac=0.2)
    cols = [c for c in feats.columns if c.startswith("diff_") or c in ("rest_diff", "home_b2b", "away_b2b", "elo_logit")]
    Xtr, stats = features.to_design_matrix(train, cols)
    Xte, _ = features.to_design_matrix(test, cols, train_stats=stats)

    for method in ["lbfgs", "irls", "gd"]:
        m = lr_mod.LogisticRegression(l2=1e-3, max_iter=200).fit(Xtr.X, Xtr.y, method=method)
        loss_curve(m.history, f"LR loss curve — {method.upper()}", out / f"loss_{method}.png")

    # Reliability of model vs ELO vs market
    m = lr_mod.LogisticRegression(l2=1e-3, max_iter=300).fit(Xtr.X, Xtr.y, method="lbfgs")
    p_te = m.predict_proba(Xte.X)
    reliability_plot(Xte.y, p_te, "LR — reliability (test)", out / "reliability_lr.png")
    reliability_plot(test["label_home_win"].to_numpy(), test["p_home_elo"].to_numpy(),
                     "ELO — reliability (test)", out / "reliability_elo.png")

    if "home_decimal" in games.columns:
        market = mk.attach_market_probs(games).set_index("game_id").loc[test["game_id"]]
        reliability_plot(test["label_home_win"].to_numpy(),
                         market["p_home_market"].to_numpy(),
                         "Market — reliability (test)", out / "reliability_market.png")

    print(f"[info] figures written to {out.resolve()}")


if __name__ == "__main__":
    main()
