# Math notes for the NBA Game-Winner project

This document is the bridge between the proposal and the code: every
formula here is implemented (and tested) in `src/nba_predictions/`. Equation
numbers are referenced in code docstrings and unit tests.

---

## 1. Setup — the prediction problem as binary classification

Each example is a game (or a game state at a fixed moment in time). Let
$y \in \{0,1\}$ be 1 iff the home team wins. We model

$$
p_i \;=\; \Pr\!\bigl(Y_i = 1 \,\bigm|\, X_i = x_i\bigr) \;=\; \sigma(x_i^\top w + b),
\qquad
\sigma(z) \;=\; \frac{1}{1+e^{-z}}.
$$

Each $Y_i$ is conditionally Bernoulli($p_i$). With independence across
games the joint likelihood is

$$
\mathcal{L}(w,b) \;=\; \prod_{i=1}^{n} p_i^{y_i}\,(1-p_i)^{1-y_i}.
\tag{1.1}
$$

The mean negative log-likelihood — the **binary cross-entropy**, or
**log-loss** — is

$$
J(w,b) \;=\; -\frac{1}{n}\sum_{i=1}^{n}\Bigl[y_i \log p_i + (1-y_i)\log(1-p_i)\Bigr].
\tag{1.2}
$$

We add an isotropic Gaussian prior $w \sim \mathcal N(0, \sigma^2 I)$ on the
weights (not the bias):

$$
J_\lambda(w,b) \;=\; J(w,b) + \tfrac{\lambda}{2}\lVert w \rVert_2^2,
\qquad \lambda = \tfrac{1}{n\sigma^2}.
\tag{1.3}
$$

This is what we minimise.

> **Code:** `logistic_regression.LogisticRegression._objective_and_grad`.
> **Test:** `test_logistic.test_gradient_finite_difference` numerically
> verifies equations (1.4)–(1.6) below.

---

## 2. Gradient, Hessian, convexity

With $p = \sigma(Xw + b\mathbf 1)$,

$$
\nabla_w J_\lambda \;=\; \tfrac{1}{n} X^\top(p - y) + \lambda w
\tag{1.4}
$$

$$
\nabla_b J_\lambda \;=\; \tfrac{1}{n}\mathbf 1^\top(p - y)
\tag{1.5}
$$

$$
H_{ww} \;=\; \nabla_w^2 J_\lambda \;=\; \tfrac{1}{n} X^\top \operatorname{diag}\!\bigl(p \odot (1-p)\bigr) X + \lambda I.
\tag{1.6}
$$

Because $p_i(1-p_i) > 0$, the diagonal matrix is positive definite, so
$H_{ww}$ is **positive semi-definite** for any $X$, and **positive
definite** whenever $\lambda > 0$. Therefore $J_\lambda$ is convex and has
a unique minimiser whenever $\lambda > 0$.

This convexity is why we can use any of L-BFGS, IRLS, or vanilla gradient
descent and get the same answer — only the *speed* differs.

---

## 3. Three optimisers, side by side

| Method                       | Per-step cost | Convergence       | Notes                                                                       |
| ---------------------------- | ------------- | ----------------- | --------------------------------------------------------------------------- |
| **Vanilla gradient descent** | $O(nd)$       | linear            | One $\nabla$ per step. Needs a learning rate.                                |
| **IRLS (Newton)**            | $O(nd^2 + d^3)$ | **quadratic**   | Solves a weighted ridge each step. Best for moderate $d$.                  |
| **L-BFGS**                   | $O(nd)$       | superlinear       | Quasi-Newton; approximates $H^{-1}$ from past gradients. Our default.       |

### IRLS as Newton's method

Newton's update for $J_\lambda$ is $\theta \leftarrow \theta - H^{-1}\nabla$.
Plugging in (1.4) and (1.6) and writing $W = \operatorname{diag}(p(1-p))$, one
Newton step satisfies

$$
\bigl(X^\top W X + n\lambda I\bigr)\,\Delta \;=\; X^\top(y - p) - n\lambda\, w.
\tag{1.7}
$$

This is exactly the normal equation of a *weighted* ridge regression — hence
"iteratively reweighted least squares". Implemented in
`logistic_regression.LogisticRegression._fit_irls`.

### Why L-BFGS for our regime

We expect on the order of $10^4$ training rows and well under $10^3$
features once the four-factor differentials, ELO logit, rest, and B2B
flags are stacked. L-BFGS is the standard choice in this regime —
superlinear convergence at the cost of only stored gradients. We use
`scipy.optimize.minimize(method="L-BFGS-B")`.

> **Test:** `test_logistic.test_solvers_agree` checks all three optimisers
> arrive at the same predictions; `test_matches_sklearn` cross-checks
> against `sklearn.linear_model.LogisticRegression`.

---

## 4. ELO is a one-parameter logistic regression

Assign each team a latent strength $R_i$ and posit

$$
\Pr(A \text{ beats } B) \;=\; \sigma\!\left(\frac{R_A - R_B}{s}\right).
\tag{2.1}
$$

The traditional base-10 form $1 / (1 + 10^{-(R_A-R_B)/400})$ is exactly
(2.1) with $s = 400/\ln 10 \approx 173.72$.

So **ELO is logistic regression with one feature — the rating difference —
and a known scale**. Including the single feature
`(R_home + H - R_away) / s` in our LR (function
`elo.elo_logit_feature`) is then equivalent to letting the LR re-scale and
shift ELO; if the rest of $X$ adds nothing, the LR will simply recover
$w = 1, b = 0$ on this column.

### ELO updates *are* SGD on the log-loss

After observing $y \in \{0,1\}$ for one game,

$$
L(R) \;=\; -\bigl[y\log p + (1-y)\log(1-p)\bigr], \quad p = \sigma\!\bigl((R_A - R_B)/s\bigr),
$$

with gradients

$$
\frac{\partial L}{\partial R_A} = \frac{p - y}{s}, \qquad
\frac{\partial L}{\partial R_B} = -\frac{p - y}{s}.
$$

One SGD step with learning rate $Ks$ on each rating gives

$$
R_A \leftarrow R_A + K\,(y - p), \qquad R_B \leftarrow R_B - K\,(y - p),
$$

which is the **textbook ELO update**. The K-factor is literally the
learning rate.

> **Test:** `test_elo.test_update_is_one_sgd_step` checks this identity.

### Home-court advantage

Encoded as a constant $H$ added to the home rating before predicting:

$$
p_{\text{home}} \;=\; \sigma\!\bigl((R_{\text{home}} + H - R_{\text{away}})/s\bigr).
$$

$H \approx 100$ ELO points reproduces the historical ~58% home win rate.

### Margin-of-victory multiplier (FiveThirtyEight)

$$
K_\text{eff} \;=\; K \cdot \frac{\ln(|\text{margin}| + 1)\cdot 2.2}{(R_w - R_\ell)\cdot 0.001 + 2.2}.
$$

The numerator rewards larger wins; the denominator damps updates when the
favourite wins (autocorrelation correction). Implemented in `EloRating.update`.

---

## 5. Neural net with team embeddings

Inputs: continuous features $x \in \mathbb R^d$, home/away team ids $h, a$.
Lookup $e_h, e_a \in \mathbb R^k$ from a learned table $E \in \mathbb R^{T\times k}$.
Use $u = [x ; e_h - e_a]$ — using the **difference** keeps the model invariant
to relabelling teams (matches the ELO worldview).

$$
h_1 = \mathrm{ReLU}(W_1 u + b_1), \qquad z = w_2^\top h_1 + b_2, \qquad p = \sigma(z).
$$

Loss is mean BCE + L2 on $W_1, w_2$ + a separate L2 on $E$ (a Gaussian
prior on team strength). Optimiser is **Adam**:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,
$$
$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t}, \quad
\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.
$$

Backprop is derived in full in the file docstring of `neural_net.py`.
Early stopping on a held-out 15% inner-validation split handles the
non-convex surface, in line with the proposal.

---

## 6. Evaluation metrics

All implemented in `nba_predictions.metrics`, all tested against
`sklearn.metrics` where applicable.

### 6.1 Log-loss

Same formula as the training loss (1.2) — the loss the proposal compares
against the prediction markets.

### 6.2 Brier score

$$
\mathrm{BS} \;=\; \tfrac{1}{n}\sum_i (p_i - y_i)^2 \;\in\; [0,1].
$$

Smaller is better. The mean-squared-error of probabilistic forecasts.

### 6.3 Murphy's three-way decomposition

Bin predictions into $K$ equal-width bins. Let $n_k$ be the bin size,
$\bar p_k$ the mean predicted probability in bin $k$, $\bar y_k$ the
empirical win rate in bin $k$, and $\bar y$ the overall positive rate.
Then

$$
\boxed{\,\mathrm{BS} \;=\; \underbrace{\tfrac{1}{n}\sum_k n_k (\bar p_k - \bar y_k)^2}_{\text{REL (calibration)}}
\;-\; \underbrace{\tfrac{1}{n}\sum_k n_k (\bar y_k - \bar y)^2}_{\text{RES (skill)}}
\;+\; \underbrace{\bar y(1-\bar y)}_{\text{UNC}}\,}
$$

- **REL** — reliability: how badly the bin means deviate from frequencies.
  Smaller is better.
- **RES** — resolution: how much the bin frequencies differ from the base
  rate. Larger is better.
- **UNC** — irreducible variance of $y$; the same for every model.

Implemented in `metrics.brier_decomposition`, tested in
`test_metrics.test_brier_decomposition_identity`.

### 6.4 AUC via Mann-Whitney U

$$
\mathrm{AUC} \;=\; \Pr\!\bigl(p(X_+) > p(X_-)\bigr) + \tfrac12\Pr\!\bigl(p(X_+) = p(X_-)\bigr)
\;=\; \frac{U}{n_+ n_-},
$$

where $U = \sum_{i\in\text{pos}}\text{rank}(p_i) - n_+(n_++1)/2$ and the
ranks are averaged across ties. No threshold needed — it measures the
ordering quality of $p$. Tested against `sklearn.metrics.roc_auc_score`.

### 6.5 Expected calibration error (ECE)

$$
\mathrm{ECE} \;=\; \sum_k \frac{n_k}{n}\,\bigl|\bar y_k - \bar p_k\bigr|.
$$

A scalar summary of how far the reliability curve lies from the diagonal.

---

## 7. Calibration

### 7.1 Platt scaling

Fit a 1-D logistic regression on the model's logits:

$$
\Pr(Y=1 \mid z) \;=\; \sigma(A z + B).
$$

To stop $(A,B)$ from overfitting on small validation sets, Platt (1999)
recommends smoothed targets

$$
t_i = \begin{cases}\frac{N_+ + 1}{N_+ + 2} & y_i = 1\\ \frac{1}{N_- + 2} & y_i = 0\end{cases}
$$

with $N_+, N_-$ the validation class counts. Implemented in
`calibration.PlattScaler` and enabled by default.

### 7.2 Isotonic regression via pool-adjacent-violators (PAV)

We want the optimum of

$$
\min_g \sum_i w_i (g_i - y_i)^2 \quad\text{s.t.}\quad g_1 \le g_2 \le \dots \le g_n,
$$

where $g_i = g(z_i)$ and $z_i$ are sorted scores. PAV builds the solution
left-to-right with a stack: whenever the new block's value is smaller than
the previous block's, merge the two into one block with weight-average
value. The algorithm runs in $O(n)$.

> **Code:** `calibration.pool_adjacent_violators`.
> **Test:** `test_calibration.test_pav_merges_violators` checks the
> textbook merge step.

---

## 8. Markets, devigging, Kelly

### 8.1 Implied probabilities

- Decimal odds $o$: $p_\text{imp} = 1/o$.
- American odds $m$: $p_\text{imp} = \frac{100}{m+100}$ if $m>0$, else $\frac{-m}{-m+100}$.
- Kalshi / Polymarket price $q \in [0,1]$: $p_\text{imp} = q$.

### 8.2 Removing the vig (normalisation)

A two-outcome book typically has $p_\text{home}^{\text{quoted}} + p_\text{away}^{\text{quoted}} > 1$.
The simplest devig divides each by the sum:

$$
p_\text{home}^{\text{devig}} \;=\; \frac{p_\text{home}^{\text{quoted}}}{p_\text{home}^{\text{quoted}} + p_\text{away}^{\text{quoted}}}.
$$

Code: `market.devig_two_way`.

### 8.3 Kelly criterion

For a single bet at decimal odds $o$ on an event the model puts at
probability $p$, with $b = o - 1$ and $q = 1 - p$,

$$
f^{*} \;=\; \frac{bp - q}{b}.
$$

$f^*$ maximises the expected log growth of the bankroll under repeated
i.i.d. bets — i.e. it maximises $E[\log(1 + fX)]$. Negative $f^*$ means
no edge; in practice half-Kelly is the workhorse default because it
trades a small amount of growth for a large reduction in variance. Code:
`market.kelly_fraction` and `kelly_backtest`.

---

## 9. Comparing the model to the market

Once predictions and devigged market probabilities live on the same set
of games we report, side by side:

- **Accuracy** at threshold 0.5
- **Log-loss** — the metric the proposal calls out
- **Brier** + Murphy decomposition (REL vs. RES)
- **AUC**
- **ECE** (calibration)
- **Kelly backtest** ROI and log-growth for whichever side the model
  favours, conditional on edge $> 2$ pp.

If the model's log-loss is *below* the market's on a sufficiently large
common slice, we have empirical evidence that we beat the population —
which is precisely the goal stated in the proposal.

Reliability diagrams (`scripts/make_figures.py`) plot the bin frequencies
against bin centers; a perfectly calibrated forecaster lies on the
diagonal $y = x$.

---

## 10. Threats to validity and what we report honestly

| Threat                       | Mitigation                                                                                   |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| Look-ahead bias              | All rolling features are computed *shifted by one* — never include the current row.          |
| Train/test leakage           | Splits are *chronological* (`features.chronological_split`), never shuffled.                 |
| Overfit on small data        | L2 in LR, L2 + early stopping in the NN, Platt smoothing on calibration.                     |
| Market is incomplete         | Kelly + log-loss are reported only on the intersection set; size of that set is in the table.|
| Multiple testing / model picking | Final reported numbers come from a single chronological holdout fixed up front.          |

We will report log-loss vs. the market exactly once, on this fixed
holdout, and quote whether the difference is plausibly inside the
sampling noise (binomial / Hoeffding bounds on the per-game log-loss
differences).
