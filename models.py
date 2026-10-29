"""Models used for the competition, beyond the six required methods.

Every `fit_*` function takes (tx, y) with y in {0, 1} and returns a score
function tx -> scores, so they plug directly into
model_selection.cross_validate. Predicting class 1 means score >= threshold,
with the threshold tuned for F1 (see model_selection.best_threshold).
"""

import numpy as np

from helpers import sigmoid, compute_logistic_loss
from implementations import least_squares, ridge_regression, reg_logistic_regression


def weighted_reg_logistic_newton(
    y, tx, lambda_, pos_weight=1.0, max_iters=25, tol=1e-4, verbose=False
):
    """Class-weighted, L2-regularized logistic regression via Newton's method.

    Minimizes
        L(w) = (1 / sum_n s_n) * sum_n s_n [log(1 + exp(x_n^T w)) - y_n x_n^T w]
               + lambda_ * ||w_{1:}||^2
    where s_n = pos_weight if y_n = 1 else 1. Up-weighting the rare positive
    class makes the model pay more attention to it. The bias (column 0) is
    not penalized.

    Newton's method uses the Hessian
        H = X^T diag(s * p * (1 - p)) X / sum(s) + 2 * lambda_ * I
    and converges in ~7 iterations, versus thousands for gradient descent,
    with no step size to tune. tx is float32, so step norms bottom out
    around 1e-4 (float32 rounding); `tol` is relative to ||w|| and set to
    stop there instead of running to max_iters.

    Returns:
        w: shape (D,), loss: unweighted, unpenalized logistic loss at w.
    """
    n, d = tx.shape
    s = np.where(y == 1, pos_weight, 1.0)
    s_sum = s.sum()
    reg = np.full(d, 2 * lambda_)
    reg[0] = 0.0  # do not penalize the bias
    w = np.zeros(d)
    dt = tx.dtype  # match tx's dtype in products to avoid copying tx
    for it in range(max_iters):
        p = sigmoid((tx @ w.astype(dt)).astype(float))
        grad = (tx.T @ (s * (p - y)).astype(dt)) / s_sum + reg * w
        hess = _weighted_gram(tx, s * p * (1 - p)) / s_sum
        hess[np.diag_indices(d)] += reg + 1e-10
        step = np.linalg.solve(hess, grad)
        w = w - step
        if verbose:
            print(f"    newton iter {it}: |step| = {np.linalg.norm(step):.2e}")
        if np.linalg.norm(step) < tol * (1 + np.linalg.norm(w)):
            break
    return w, compute_logistic_loss(y, tx, w)


def _weighted_gram(tx, d, chunk=50_000):
    """X^T diag(d) X computed in row chunks to limit memory use."""
    out = np.zeros((tx.shape[1], tx.shape[1]))
    for i in range(0, tx.shape[0], chunk):
        xc = tx[i : i + chunk]
        out += (xc * d[i : i + chunk, None].astype(xc.dtype)).T @ xc
    return out


# ---------------------------------------------------------------------------
# Factories returning fit functions for cross_validate
# ---------------------------------------------------------------------------


def logistic_newton(lambda_=1e-4, pos_weight=1.0):
    def fit(tx, y):
        w, _ = weighted_reg_logistic_newton(y, tx, lambda_, pos_weight)
        return lambda t: t @ w.astype(t.dtype)

    return fit


def logistic_gd(lambda_=1e-4, gamma=0.5, max_iters=300):
    """Required reg_logistic_regression (plain gradient descent)."""

    def fit(tx, y):
        w0 = np.zeros(tx.shape[1], dtype=tx.dtype)
        w, _ = reg_logistic_regression(
            y.astype(tx.dtype), tx, lambda_, w0, max_iters, gamma
        )
        return lambda t: t @ w.astype(t.dtype)

    return fit


def linear_least_squares():
    """Least squares on y in {0, 1}; the prediction is thresholded.

    Some BRFSS columns are exact functions of others (e.g. _RFHLTH is derived
    from GENHLTH), which makes X^T X singular. In that case we fall back to
    the minimum-norm solution with the pseudo-inverse.
    """

    def fit(tx, y):
        tx, y = tx.astype(float), y.astype(float)
        try:
            w, _ = least_squares(y, tx)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(tx.T @ tx) @ (tx.T @ y)
        return lambda t: t @ w

    return fit


def linear_ridge(lambda_=1e-5):
    def fit(tx, y):
        w, _ = ridge_regression(y.astype(float), tx.astype(float), lambda_)
        return lambda t: t @ w

    return fit
