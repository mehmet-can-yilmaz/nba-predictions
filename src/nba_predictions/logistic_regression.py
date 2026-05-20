r"""Logistic regression — derived from scratch.

We model the home team's win probability as

    p_i = sigma(x_i^T w + b),   sigma(z) = 1 / (1 + exp(-z)).

Setup (Bernoulli MLE -> binary cross-entropy)
---------------------------------------------
Each label y_i in {0, 1} is Bernoulli(p_i). The likelihood is

    L(w, b) = prod_i p_i^{y_i} (1 - p_i)^{1 - y_i}.

Taking -log and averaging over n examples gives the binary cross-entropy

    J(w, b) = - (1/n) sum_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ].

Adding an isotropic Gaussian prior w ~ N(0, sigma^2 I) (a.k.a. L2):

    J_lambda(w, b) = J(w, b) + (lambda / 2) ||w||_2^2,

where lambda = 1 / (n sigma^2). The bias b is conventionally unpenalised.

Gradient and Hessian
--------------------
Let X be the n x d design matrix and p = sigma(Xw + b 1). Then

    grad_w J_lambda = (1/n) X^T (p - y) + lambda w,
    grad_b J_lambda = (1/n) sum_i (p_i - y_i),
    H_ww          = (1/n) X^T diag(p (1 - p)) X + lambda I.

Because p(1-p) > 0, H_ww is positive semi-definite, so J_lambda is convex
in (w, b). A unique minimiser exists whenever lambda > 0 (it also exists for
lambda = 0 when X has full column rank and the classes are not linearly
separable; otherwise the MLE diverges).

Optimisers implemented here
---------------------------
1. Batch gradient descent — for pedagogy, lets us watch the loss decrease.
2. Iteratively reweighted least squares (IRLS) — Newton's method on
   J_lambda, mathematically equivalent to solving a weighted ridge regression
   at every step. Quadratic convergence near the optimum.
3. L-BFGS via scipy.optimize.minimize — quasi-Newton, the workhorse for
   smooth convex losses with many features.

We expose `LogisticRegression.fit(..., method=...)` so that the report can
compare convergence behaviour empirically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from scipy import optimize


# ---------------------------------------------------------------------------
# Numerically stable primitives
# ---------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigma(z) = 1 / (1 + exp(-z)).

    For large |z| the naive expression overflows; we branch by sign.
    """
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log1pexp(z: np.ndarray) -> np.ndarray:
    """log(1 + exp(z)) computed without overflow (the softplus)."""
    return np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))


def bce_loss(y: np.ndarray, logits: np.ndarray) -> float:
    """Mean binary cross-entropy from *logits*, not probabilities.

    Using logits z = Xw + b directly we have
        -[y log sigma(z) + (1-y) log(1 - sigma(z))]
        = log(1 + exp(z)) - y * z = softplus(z) - y z,
    which avoids ever evaluating log(0).
    """
    return float(np.mean(log1pexp(logits) - y * logits))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class LogisticRegression:
    """Penalised logistic regression, derived from first principles.

    Parameters
    ----------
    l2 :
        Regularisation strength lambda. Penalises the slope, not the bias.
    fit_intercept :
        Whether to learn b.
    max_iter, tol :
        Stopping criteria for iterative solvers.
    """

    l2: float = 1e-3
    fit_intercept: bool = True
    max_iter: int = 200
    tol: float = 1e-7

    w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b: float = 0.0
    history: list[float] = field(default_factory=list)

    # ---- core computations ------------------------------------------------
    def _logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + (self.b if self.fit_intercept else 0.0)

    def _objective_and_grad(self, params: np.ndarray, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        w = params[:d]
        b = params[d] if self.fit_intercept else 0.0
        z = X @ w + b
        # loss: mean softplus(z) - y z, plus L2 on w
        loss = float(np.mean(log1pexp(z) - y * z) + 0.5 * self.l2 * w @ w)
        p = sigmoid(z)
        grad_w = X.T @ (p - y) / n + self.l2 * w
        if self.fit_intercept:
            grad_b = float(np.mean(p - y))
            grad = np.concatenate([grad_w, [grad_b]])
        else:
            grad = grad_w
        return loss, grad

    # ---- public API -------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: Literal["lbfgs", "irls", "gd"] = "lbfgs",
        lr: float = 0.5,
        callback: Callable[[np.ndarray], None] | None = None,
    ) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.history = []

        if method == "lbfgs":
            self._fit_lbfgs(X, y, callback)
        elif method == "irls":
            self._fit_irls(X, y, callback)
        elif method == "gd":
            self._fit_gd(X, y, lr, callback)
        else:
            raise ValueError(f"Unknown method {method!r}")
        return self

    def _pack(self) -> np.ndarray:
        return np.concatenate([self.w, [self.b]]) if self.fit_intercept else self.w.copy()

    def _unpack(self, params: np.ndarray) -> None:
        d = self.w.shape[0]
        self.w = params[:d].copy()
        if self.fit_intercept:
            self.b = float(params[d])

    # ---- L-BFGS -----------------------------------------------------------
    def _fit_lbfgs(self, X, y, callback):
        x0 = self._pack()

        def f(p):
            loss, grad = self._objective_and_grad(p, X, y)
            self.history.append(loss)
            if callback is not None:
                callback(p)
            return loss, grad

        result = optimize.minimize(
            f,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "gtol": self.tol, "ftol": self.tol},
        )
        self._unpack(result.x)

    # ---- IRLS (Newton) ----------------------------------------------------
    def _fit_irls(self, X, y, callback):
        """One IRLS step solves the normal equations of a weighted ridge:

            (X^T W X + n lambda I) delta = X^T (y - p) - n lambda w,

        with W = diag(p(1-p)). This is exactly the Newton update for J_lambda.
        Convergence is quadratic near the optimum, but each step costs O(d^3)
        from the linear solve, so IRLS is best for d on the order of hundreds.
        """
        n, d = X.shape
        if self.fit_intercept:
            X_ = np.hstack([X, np.ones((n, 1))])
            d_ = d + 1
            reg = self.l2 * np.eye(d_)
            reg[-1, -1] = 0.0  # do not penalise bias
        else:
            X_ = X
            d_ = d
            reg = self.l2 * np.eye(d_)

        beta = np.zeros(d_)
        prev_loss = np.inf
        for it in range(self.max_iter):
            z = X_ @ beta
            p = sigmoid(z)
            W = p * (1.0 - p)
            # Guard against W -> 0 (perfect classification driving Hessian singular)
            W = np.clip(W, 1e-9, None)
            grad = X_.T @ (p - y) / n + reg @ beta
            H = (X_.T * W) @ X_ / n + reg
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            beta_new = beta - step
            loss = bce_loss(y, X_ @ beta_new) + 0.5 * self.l2 * beta_new[: d] @ beta_new[: d]
            self.history.append(loss)
            if callback is not None:
                callback(beta_new)
            if abs(prev_loss - loss) < self.tol:
                beta = beta_new
                break
            beta, prev_loss = beta_new, loss

        if self.fit_intercept:
            self.w = beta[:-1]
            self.b = float(beta[-1])
        else:
            self.w = beta
            self.b = 0.0

    # ---- Vanilla gradient descent ----------------------------------------
    def _fit_gd(self, X, y, lr, callback):
        params = self._pack()
        for _ in range(self.max_iter):
            loss, grad = self._objective_and_grad(params, X, y)
            self.history.append(loss)
            if callback is not None:
                callback(params)
            new = params - lr * grad
            if np.linalg.norm(new - params) < self.tol:
                params = new
                break
            params = new
        self._unpack(params)

    # ---- inference --------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self._logits(np.asarray(X, dtype=float)))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
