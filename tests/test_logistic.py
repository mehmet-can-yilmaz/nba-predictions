"""Test the logistic regression solvers against (a) each other and (b)
sklearn as an independent reference for the optimum.
"""
from __future__ import annotations

import numpy as np
import pytest

from nba_predictions.logistic_regression import (
    LogisticRegression,
    bce_loss,
    log1pexp,
    sigmoid,
)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n, d = 400, 5
    X = rng.normal(size=(n, d))
    w_true = np.array([1.5, -2.0, 0.0, 0.5, -0.7])
    b_true = 0.3
    p = 1.0 / (1.0 + np.exp(-(X @ w_true + b_true)))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y, w_true, b_true


def test_sigmoid_stable():
    z = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
    p = sigmoid(z)
    assert np.all(np.isfinite(p))
    assert p[0] == pytest.approx(0.0, abs=1e-12)
    assert p[-1] == pytest.approx(1.0, abs=1e-12)


def test_log1pexp_matches_sigmoid_identity():
    # softplus(z) - z = softplus(-z), i.e. -log sigma(z)
    z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert np.allclose(log1pexp(z) - z, log1pexp(-z))


def test_bce_from_logits_matches_naive():
    rng = np.random.default_rng(1)
    z = rng.normal(scale=3, size=200)
    y = (rng.uniform(size=200) > 0.5).astype(float)
    p = sigmoid(z)
    naive = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert bce_loss(y, z) == pytest.approx(naive, rel=1e-6)


@pytest.mark.parametrize("method", ["lbfgs", "irls", "gd"])
def test_solvers_agree(toy, method):
    X, y, *_ = toy
    m = LogisticRegression(l2=1e-3, max_iter=500, tol=1e-9)
    m.fit(X, y, method=method, lr=0.5)
    # log-loss should be close to L-BFGS reference
    ref = LogisticRegression(l2=1e-3, max_iter=500, tol=1e-9).fit(X, y, method="lbfgs")
    pa = m.predict_proba(X)
    pb = ref.predict_proba(X)
    assert np.mean((pa - pb) ** 2) < 1e-3


def test_matches_sklearn(toy):
    """Our optimum should match sklearn's at the same L2 strength.

    Note: sklearn parameterises regularisation as C = 1 / (n * lambda) when
    using `liblinear` or as 1 / lambda with `lbfgs` on the *sum* (not mean)
    loss. We use lambda on the mean loss; the equivalent sklearn C for our
    lambda is 1 / (n * lambda)."""
    from sklearn.linear_model import LogisticRegression as Sk
    X, y, *_ = toy
    n = len(y)
    lam = 1e-2
    ours = LogisticRegression(l2=lam, max_iter=500, tol=1e-10).fit(X, y, method="lbfgs")
    sk = Sk(C=1.0 / (n * lam), penalty="l2", solver="lbfgs", max_iter=500, tol=1e-10).fit(X, y)
    # compare on prediction agreement, not exact weights (parameterisation can shift)
    assert np.mean(np.abs(ours.predict_proba(X) - sk.predict_proba(X)[:, 1])) < 5e-3


def test_gradient_finite_difference(toy):
    """Numerical check that our analytical gradient matches finite differences."""
    X, y, *_ = toy
    m = LogisticRegression(l2=1e-2, fit_intercept=True)
    m.w = np.zeros(X.shape[1])
    m.b = 0.0
    params = m._pack()
    _, grad = m._objective_and_grad(params, X, y)
    fd = np.empty_like(grad)
    h = 1e-5
    for i in range(len(params)):
        p_plus = params.copy(); p_plus[i] += h
        p_minus = params.copy(); p_minus[i] -= h
        f_plus, _ = m._objective_and_grad(p_plus, X, y)
        f_minus, _ = m._objective_and_grad(p_minus, X, y)
        fd[i] = (f_plus - f_minus) / (2 * h)
    assert np.allclose(grad, fd, atol=1e-6)
