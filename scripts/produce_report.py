"""Produce all artifacts referenced by the paper and slides.

Outputs to:
    figures/         — PNG/PDF plots
    data/processed/  — metrics_<model>.csv, model_comparison.csv
    data/processed/baseline_preds.csv

Usage:
    PYTHONPATH=src python scripts/produce_report.py --data data/processed/games.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nba_predictions import (
    calibration,
    elo as elo_mod,
    features,
    logistic_regression as lr_mod,
    metrics,
    neural_net as nn_mod,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    # also save pdf for the paper
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_elo_trace(games: pd.DataFrame, out: Path, top_n: int = 6) -> None:
    elo_run = elo_mod.EloRating()
    teams = sorted(set(games["home"]).union(games["away"]))
    traces = {t: [] for t in teams}
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
    bot = sorted(final, key=final.get)[:3]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for t in top:
        ax.plot(traces[t], label=f"{t} ({final[t]:.0f})", linewidth=1.1)
    for t in bot:
        ax.plot(traces[t], label=f"{t} ({final[t]:.0f})", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.axhline(1500, color="black", linestyle=":", linewidth=0.6)
    ax.set_xlabel("game index (chronological)")
    ax.set_ylabel("ELO rating")
    ax.set_title(f"ELO trajectories — top {top_n} and bottom 3")
    ax.legend(loc="best", ncol=2)
    _save(fig, out)


def fig_loss_curves_combined(Xtr, ytr, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)
    titles = {"lbfgs": "L-BFGS (superlinear)",
              "irls":  "IRLS / Newton (quadratic)",
              "gd":    "Gradient Descent (linear)"}
    for ax, method in zip(axes, ["lbfgs", "irls", "gd"]):
        m = lr_mod.LogisticRegression(l2=1e-3, max_iter=200).fit(Xtr, ytr, method=method, lr=0.5)
        ax.plot(m.history, color="C0")
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss" if method == "lbfgs" else "")
        ax.set_title(titles[method])
        ax.grid(True, alpha=0.3)
    fig.suptitle("Same convex loss, three optimisers reach the same optimum", y=1.04)
    _save(fig, out)


def fig_reliability_combined(y, preds: dict[str, np.ndarray], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect", linewidth=1)
    colors = ["C0", "C1", "C2", "C3"]
    for (name, p), c in zip(preds.items(), colors):
        centers, rates, counts = metrics.reliability_curve(y, p, n_bins=10)
        mask = counts > 5  # suppress noisy bins
        ax.plot(centers[mask], rates[mask], "-o", color=c, label=name, markersize=5)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical frequency")
    ax.set_title("Reliability diagram (test set)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def fig_model_comparison(metrics_table: pd.DataFrame, out: Path) -> None:
    metrics_to_plot = ["accuracy", "auc", "log_loss", "brier"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, m in zip(axes, metrics_to_plot):
        ax.bar(metrics_table.index, metrics_table[m], color=["C0", "C1", "C2"])
        ax.set_title(m.replace("_", "-"))
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, alpha=0.3, axis="y")
        if m in ("log_loss", "brier"):
            ax.set_ylabel("lower is better")
        else:
            ax.set_ylabel("higher is better")
    fig.suptitle("Held-out test set: model comparison", y=1.04)
    _save(fig, out)


def fig_calibration_demo(y, p, out: Path) -> None:
    """Show the effect of Platt scaling on a deliberately miscalibrated model."""
    # Sharpen the predictions to make them over-confident
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    p_overconf = 1 / (1 + np.exp(-1.6 * logit))
    cal = calibration.PlattScaler().fit(p_overconf, y)
    p_cal = cal.transform(p_overconf)

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect", linewidth=1)
    for label, pp, col in [("raw (deliberately over-confident)", p_overconf, "C3"),
                            ("after Platt scaling",              p_cal,       "C2")]:
        centers, rates, counts = metrics.reliability_curve(y, pp, n_bins=10)
        mask = counts > 5
        ax.plot(centers[mask], rates[mask], "-o", color=col, label=label, markersize=5)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical frequency")
    ax.set_title("Calibration: Platt scaling pulls a miscalibrated model to the diagonal")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-figures", default="figures")
    parser.add_argument("--out-metrics", default="data/processed")
    args = parser.parse_args()

    fig_dir = Path(args.out_figures)
    met_dir = Path(args.out_metrics)
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    games = pd.read_csv(args.data, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    print(f"[info] loaded {len(games)} games, seasons {sorted(games['season'].unique())}")

    # --- ELO ---------------------------------------------------------
    elo_run = elo_mod.EloRating()
    games = elo_run.fit_predict(games)
    games["elo_logit"] = elo_mod.elo_logit_feature(games["elo_home_pre"], games["elo_away_pre"])

    fig_elo_trace(games.copy(), fig_dir / "elo_trace.png")

    # --- Features ----------------------------------------------------
    core = ["game_id", "date", "season", "home", "away", "home_score", "away_score"]
    optional = ["fga_home", "fga_away", "fta_home", "fta_away",
                "oreb_home", "oreb_away", "tov_home", "tov_away"]
    keep = core + [c for c in optional if c in games.columns]
    feats = features.build_game_features(games[keep], window=10)
    feats = feats.merge(games[["game_id", "elo_logit", "p_home_elo"]], on="game_id")

    train, test = features.chronological_split(feats, holdout_frac=0.20)
    feature_cols = [c for c in feats.columns
                    if c.startswith("diff_") or c in ("rest_diff", "home_b2b", "away_b2b", "elo_logit")]
    Xtr, stats = features.to_design_matrix(train, feature_cols)
    Xte, _ = features.to_design_matrix(test, feature_cols, train_stats=stats)
    print(f"[info] train: {len(Xtr.X)}  test: {len(Xte.X)}  features: {len(feature_cols)}")

    # --- LR convergence figure --------------------------------------
    fig_loss_curves_combined(Xtr.X, Xtr.y, fig_dir / "loss_curves.png")

    # --- LR baseline -------------------------------------------------
    lr = lr_mod.LogisticRegression(l2=1e-3, max_iter=300).fit(Xtr.X, Xtr.y, method="lbfgs")
    p_lr_train = lr.predict_proba(Xtr.X)
    p_lr_test  = lr.predict_proba(Xte.X)
    cal_lr, p_lr_test_cal = calibration.calibrate(p_lr_train, Xtr.y, p_lr_test, method="platt")

    # --- NN ----------------------------------------------------------
    teams = sorted(set(games["home"]).union(games["away"]))
    team_ix = {t: i for i, t in enumerate(teams)}
    h_tr = train["home"].map(team_ix).to_numpy()
    a_tr = train["away"].map(team_ix).to_numpy()
    h_te = test["home"].map(team_ix).to_numpy()
    a_te = test["away"].map(team_ix).to_numpy()
    cut = int(len(Xtr.X) * 0.85)
    nn_model = nn_mod.TeamEmbeddingNN(
        n_teams=len(teams), n_features=Xtr.X.shape[1],
        embedding_dim=8, hidden_dim=16, l2=1e-4, l2_emb=1e-3,
    ).fit(
        Xtr.X[:cut], h_tr[:cut], a_tr[:cut], Xtr.y[:cut].astype(float),
        X_val=Xtr.X[cut:], home_val=h_tr[cut:], away_val=a_tr[cut:], y_val=Xtr.y[cut:].astype(float),
        epochs=80, lr=3e-3,
    )
    p_nn_train = nn_model.predict_proba(Xtr.X, h_tr, a_tr)
    p_nn_test  = nn_model.predict_proba(Xte.X, h_te, a_te)
    _, p_nn_test_cal = calibration.calibrate(p_nn_train, Xtr.y, p_nn_test, method="platt")

    # --- ELO baseline (already computed) -----------------------------
    p_elo_test = test["p_home_elo"].to_numpy()
    y_test = test["label_home_win"].to_numpy()

    # --- Metrics table ----------------------------------------------
    rows = {
        "ELO":          metrics.report(y_test, p_elo_test),
        "LR (LBFGS)":   metrics.report(y_test, p_lr_test),
        "LR + Platt":   metrics.report(y_test, p_lr_test_cal),
        "NN (Adam)":    metrics.report(y_test, p_nn_test),
        "NN + Platt":   metrics.report(y_test, p_nn_test_cal),
    }
    df = pd.DataFrame(rows).T
    df.to_csv(met_dir / "model_comparison.csv")
    print("\n=== Held-out test set (chronological 20%) ===")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    # Save calibration scalars
    Path(met_dir / "platt_params.json").write_text(json.dumps({
        "lr_platt_A": cal_lr.A, "lr_platt_B": cal_lr.B,
    }, indent=2))

    # --- Reliability figure (combined) ------------------------------
    fig_reliability_combined(y_test, {
        "ELO":         p_elo_test,
        "LR":          p_lr_test_cal,
        "NN":          p_nn_test_cal,
    }, fig_dir / "reliability_combined.png")

    # --- Calibration demo -------------------------------------------
    fig_calibration_demo(y_test, p_lr_test, fig_dir / "calibration_demo.png")

    # --- Model comparison bar chart ---------------------------------
    bar_df = df.loc[["ELO", "LR + Platt", "NN + Platt"]].copy()
    bar_df.index = ["ELO", "LR", "NN"]
    fig_model_comparison(bar_df, fig_dir / "model_comparison.png")

    # --- Save predictions for follow-up market analysis -------------
    out_preds = pd.DataFrame({
        "game_id": test["game_id"].to_numpy(),
        "date":    test["date"].to_numpy(),
        "home":    test["home"].to_numpy(),
        "away":    test["away"].to_numpy(),
        "y":       y_test,
        "p_elo":   p_elo_test,
        "p_lr":    p_lr_test,
        "p_lr_cal":p_lr_test_cal,
        "p_nn":    p_nn_test,
        "p_nn_cal":p_nn_test_cal,
    })
    out_preds.to_csv(met_dir / "baseline_preds.csv", index=False)
    print(f"\n[info] artifacts written to {fig_dir} and {met_dir}")


if __name__ == "__main__":
    main()
