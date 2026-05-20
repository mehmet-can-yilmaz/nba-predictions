r"""ELO ratings — a logistic regression in disguise.

ELO as a logistic model
-----------------------
Assign each team a latent strength R_i. Model the probability that team A
beats team B as

    P(A beats B) = sigma( (R_A - R_B) / s )           (1)

where sigma is the logistic and s sets the spread. The traditional ELO
parameterisation uses a base-10 logistic with s = 400 / log(10):

    P(A beats B) = 1 / (1 + 10^{-(R_A - R_B) / 400}).

Setting r := (R_A - R_B) / 400 we have P = sigma(r log 10) which is exactly
(1) with s = 400 / log(10) ~= 173.7. So ELO is a *one-parameter* logistic
regression where the only feature is the rating difference.

ELO updates as stochastic gradient descent
------------------------------------------
After observing outcome y in {0, 1} (1 if A wins) the negative log-likelihood is

    L(R) = - y log p - (1 - y) log(1 - p),  with p = sigma((R_A - R_B)/s).

The gradients are

    dL/dR_A = (p - y) / s,
    dL/dR_B = -(p - y) / s.

Doing one SGD step with learning rate K * s on each rating gives

    R_A <- R_A + K (y - p),
    R_B <- R_B - K (y - p),

which is exactly the textbook ELO update. The K factor *is* the learning
rate. Smaller K = slower adaptation, lower variance; larger K = faster
adaptation, higher variance. NBA implementations commonly use K in the
range 20-32.

Home-court advantage
--------------------
Empirically the home team wins ~58% of NBA games. Encode this with a
constant H added to the home team's rating before predicting:

    p_home = sigma( (R_home + H - R_away) / s ).

H ~= 100 ELO points reproduces the historical home win rate (see
FiveThirtyEight's NBA methodology).

Margin of victory multiplier (optional, FiveThirtyEight-style)
--------------------------------------------------------------
Vanilla ELO ignores the score. A common refinement is

    K_eff = K * ln(|margin| + 1) * 2.2 / ((R_winner - R_loser_pre) * 0.001 + 2.2),

which (a) gives more credit to larger wins and (b) tempers updates when
the favourite wins big (autocorrelation correction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

S_DEFAULT = 400.0 / np.log(10.0)   # ~173.7178; spread of the standard ELO logistic


def expected_score(r_a: float, r_b: float, s: float = S_DEFAULT) -> float:
    """P(A beats B) under the base-10 ELO logistic."""
    return float(1.0 / (1.0 + np.exp(-(r_a - r_b) / s)))


@dataclass
class EloRating:
    """Streaming ELO with home-court advantage and an optional MOV multiplier.

    Parameters
    ----------
    k :
        Base learning rate. 20 is conservative, 32 is the chess default.
    home_advantage :
        ELO points added to the home team before prediction.
    mov_multiplier :
        If True, scales K by the FiveThirtyEight margin-of-victory factor.
    initial_rating :
        Rating assigned to a team on first appearance.
    season_regress :
        Between-season mean reversion. 0.25 means each rating moves 25% of
        the way back toward 1500 at season boundaries (handled by the caller
        via `regress_to_mean`).
    """

    k: float = 20.0
    home_advantage: float = 100.0
    mov_multiplier: bool = True
    initial_rating: float = 1500.0
    season_regress: float = 0.25

    ratings: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def get(self, team: str) -> float:
        return self.ratings.setdefault(team, self.initial_rating)

    def predict_home(self, home: str, away: str) -> float:
        return expected_score(self.get(home) + self.home_advantage, self.get(away))

    def update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
    ) -> tuple[float, float]:
        """Update both ratings after a game. Returns (p_home_pre, delta)."""
        r_home = self.get(home)
        r_away = self.get(away)
        p_home = expected_score(r_home + self.home_advantage, r_away)
        y = 1.0 if home_score > away_score else 0.0

        if self.mov_multiplier:
            margin = abs(home_score - away_score)
            winner_diff = (r_home - r_away) * (1 if y == 1 else -1)
            mov = np.log(margin + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
            k = self.k * mov
        else:
            k = self.k

        delta = k * (y - p_home)
        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta
        return p_home, delta

    def regress_to_mean(self) -> None:
        """Season-boundary mean reversion. Each rating R moves toward 1500."""
        for t, r in self.ratings.items():
            self.ratings[t] = 1500.0 + (1.0 - self.season_regress) * (r - 1500.0)

    # ------------------------------------------------------------------
    def fit_predict(self, games: pd.DataFrame) -> pd.DataFrame:
        """Run ELO over a chronologically-sorted schedule.

        `games` must contain columns: date, season, home, away, home_score,
        away_score. Returns the same frame with `elo_home_pre`, `elo_away_pre`,
        `p_home_elo` appended.
        """
        required = {"date", "season", "home", "away", "home_score", "away_score"}
        missing = required - set(games.columns)
        if missing:
            raise ValueError(f"games is missing columns: {sorted(missing)}")

        games = games.sort_values("date").reset_index(drop=True).copy()
        elo_home_pre = np.empty(len(games))
        elo_away_pre = np.empty(len(games))
        p_home = np.empty(len(games))

        last_season = None
        for i, row in games.iterrows():
            if last_season is not None and row.season != last_season:
                self.regress_to_mean()
            last_season = row.season

            elo_home_pre[i] = self.get(row.home)
            elo_away_pre[i] = self.get(row.away)
            p_home[i], _ = self.update(row.home, row.away, row.home_score, row.away_score)

        games["elo_home_pre"] = elo_home_pre
        games["elo_away_pre"] = elo_away_pre
        games["p_home_elo"] = p_home
        return games


# ----------------------------------------------------------------------
# Helpers for downstream models — turn ratings into a single ELO feature
# ----------------------------------------------------------------------
def elo_logit_feature(
    elo_home: Iterable[float],
    elo_away: Iterable[float],
    home_advantage: float = 100.0,
    s: float = S_DEFAULT,
) -> np.ndarray:
    """Return (R_home + H - R_away) / s — i.e. the logit of the ELO win prob.

    Including this single feature in a logistic regression lets the model
    rescale/shift ELO without re-deriving it.
    """
    a = np.asarray(list(elo_home), dtype=float)
    b = np.asarray(list(elo_away), dtype=float)
    return (a + home_advantage - b) / s
