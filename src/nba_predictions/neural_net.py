r"""A small feed-forward network with team embeddings.

Architecture
------------
Inputs per game:
  - x:   continuous/diff features (rolling stats, rest, ELO logit), dimension d
  - h, a: integer ids for home and away teams

Embeddings:
  E in R^{T x k} — a lookup table. Each team i has a learned vector e_i in R^k.

Forward pass:
  e_h = E[h], e_a = E[a]
  u   = [x ; e_h - e_a]                        (concatenation, size d + k)
  h_1 = relu(W_1 u + b_1)                       (hidden layer of width m)
  z   = w_2^T h_1 + b_2                         (logit)
  p   = sigma(z)                                (home win probability)

We use (e_h - e_a) rather than [e_h ; e_a] so that team identity enters only
through a *relative* strength vector — this is in the spirit of ELO and
keeps the model invariant to swapping the labels of two teams.

Loss
----
Mean BCE with L2 regularisation on weights (not embeddings) — embeddings
are regularised separately with weight `l2_emb`, which acts as a Gaussian
prior on team strength.

Backprop, in full
-----------------
Let n = batch size, denote 1{...} as the indicator. For each example,

    dL/dz   = p - y                                                  (1)
    dL/dw_2 = h_1 (p - y)                                             (2)
    dL/db_2 = (p - y)                                                 (3)
    dL/dh_1 = w_2 (p - y)                                             (4)
    dL/d(W_1 u + b_1) = dL/dh_1 * 1{W_1 u + b_1 > 0}   (ReLU derivative)
    dL/dW_1 = (dL/d(.)) outer u                                       (5)
    dL/db_1 = dL/d(.)                                                 (6)
    dL/du   = W_1^T dL/d(.)                                           (7)
    dL/d e_h =  dL/du[d:]                                             (8a)
    dL/d e_a = -dL/du[d:]                                             (8b)

Averaging (1)-(8) over the batch and adding the L2 gradients
(lambda W_1, lambda w_2, lambda_emb E) gives the full gradient.

We optimise with Adam (Kingma & Ba 2015). The state per parameter:

    m_t = beta_1 m_{t-1} + (1 - beta_1) g_t
    v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2
    m_hat = m_t / (1 - beta_1^t),   v_hat = v_t / (1 - beta_2^t)
    theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

We use early stopping on a validation split for the nonconvex surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .logistic_regression import log1pexp, sigmoid


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# Adam optimiser
# ---------------------------------------------------------------------------
class Adam:
    def __init__(self, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, g in grads.items():
            if k not in self.m:
                self.m[k] = np.zeros_like(g)
                self.v[k] = np.zeros_like(g)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class TeamEmbeddingNN:
    n_teams: int
    n_features: int
    embedding_dim: int = 8
    hidden_dim: int = 16
    l2: float = 1e-4
    l2_emb: float = 1e-3
    seed: int = 0

    params: dict[str, np.ndarray] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        d = self.n_features + self.embedding_dim
        m = self.hidden_dim
        # He init for ReLU
        self.params = {
            "E":   rng.normal(0, 0.1, size=(self.n_teams, self.embedding_dim)),
            "W1":  rng.normal(0, np.sqrt(2.0 / d), size=(m, d)),
            "b1":  np.zeros(m),
            "w2":  rng.normal(0, np.sqrt(2.0 / m), size=m),
            "b2":  0.0,
        }

    # ---- forward / loss -------------------------------------------------
    def _forward(self, X: np.ndarray, home_ix: np.ndarray, away_ix: np.ndarray):
        E = self.params["E"]
        eh = E[home_ix]
        ea = E[away_ix]
        u = np.concatenate([X, eh - ea], axis=1)
        pre = u @ self.params["W1"].T + self.params["b1"]
        h1 = _relu(pre)
        logits = h1 @ self.params["w2"] + self.params["b2"]
        return logits, h1, pre, u

    def loss(self, X, home_ix, away_ix, y):
        logits, *_ = self._forward(X, home_ix, away_ix)
        bce = float(np.mean(log1pexp(logits) - y * logits))
        reg = 0.5 * self.l2 * (
            float(np.sum(self.params["W1"] ** 2)) + float(np.sum(self.params["w2"] ** 2))
        )
        reg += 0.5 * self.l2_emb * float(np.sum(self.params["E"] ** 2))
        return bce + reg

    # ---- backward -------------------------------------------------------
    def _grads(self, X, home_ix, away_ix, y) -> dict[str, np.ndarray]:
        n = X.shape[0]
        logits, h1, pre, u = self._forward(X, home_ix, away_ix)
        p = sigmoid(logits)

        dlogit = (p - y) / n                              # (n,)
        dw2 = h1.T @ dlogit + self.l2 * self.params["w2"]
        db2 = float(dlogit.sum())
        dh1 = np.outer(dlogit, self.params["w2"])         # (n, m)
        dpre = dh1 * (pre > 0)
        dW1 = dpre.T @ u + self.l2 * self.params["W1"]
        db1 = dpre.sum(axis=0)
        du = dpre @ self.params["W1"]                     # (n, d+k)

        d_diff = du[:, X.shape[1]:]
        dE = np.zeros_like(self.params["E"])
        np.add.at(dE,  home_ix,  d_diff)
        np.add.at(dE,  away_ix, -d_diff)
        dE += self.l2_emb * self.params["E"]

        return {"E": dE, "W1": dW1, "b1": db1, "w2": dw2, "b2": np.array(db2)}

    # ---- training -------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        home_ix: np.ndarray,
        away_ix: np.ndarray,
        y: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        home_val: Optional[np.ndarray] = None,
        away_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 10,
        verbose: bool = False,
    ) -> "TeamEmbeddingNN":
        opt = Adam(lr=lr)
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)
        best_val = np.inf
        best_params = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in self.params.items()}
        bad = 0

        for epoch in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                ix = order[start : start + batch_size]
                grads = self._grads(X[ix], home_ix[ix], away_ix[ix], y[ix])
                # b2 is a scalar; cast for Adam's array handling
                p = {k: (np.asarray(v).copy() if k == "b2" else v) for k, v in self.params.items()}
                opt.step(p, grads)
                # write back, preserving b2 as scalar
                self.params = {k: (float(v) if k == "b2" else v) for k, v in p.items()}

            train_loss = self.loss(X, home_ix, away_ix, y)
            log = {"epoch": epoch, "train_loss": train_loss}
            if X_val is not None:
                val_loss = self.loss(X_val, home_val, away_val, y_val)
                log["val_loss"] = val_loss
                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    best_params = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in self.params.items()}
                    bad = 0
                else:
                    bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"early stop at epoch {epoch}")
                    break
            self.history.append(log)
            if verbose and epoch % 10 == 0:
                print(log)

        if X_val is not None:
            self.params = best_params
        return self

    # ---- inference ------------------------------------------------------
    def predict_proba(self, X: np.ndarray, home_ix: np.ndarray, away_ix: np.ndarray) -> np.ndarray:
        logits, *_ = self._forward(X, home_ix, away_ix)
        return sigmoid(logits)
