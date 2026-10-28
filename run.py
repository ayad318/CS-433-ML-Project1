"""Produce the AIcrowd submission for the heart-attack (MICHD) prediction task.

Pipeline:
  1. model A: preprocess the BRFSS features (see preprocessing.py) and fit
     L2-regularized logistic regression with Newton's method;
  2. model B: gradient-boosted trees on the raw survey codes (see gbdt.py);
  3. average the two predicted probabilities (the ensemble beats both
     models alone on every CV fold);
  4. choose the decision threshold that maximizes F1 on out-of-fold
     predictions (5-fold cross-validation), since the positive class is
     only ~9% of the data and the default 0.5 threshold gives poor F1;
  5. refit on the full training set and write submission.csv.

Usage: python run.py  (about 20-30 minutes on a laptop)
"""

import numpy as np

from helpers import load_csv_data, create_csv_submission, sigmoid
from gbdt import GradientBoosting
from preprocessing import DROP_COLUMNS, Preprocessor, load_column_names

DATA_PATH = "dataset"
LAMBDA = 1e-5  # chosen by CV; F1 is flat between 1e-6 and 1e-3
N_FOLDS = 5
SEED = 0  # fixes the CV split so the output is reproducible


def reg_logistic_regression_newton(y, tx, lambda_, max_iters=15, tol=1e-6):
    """L2-regularized logistic regression solved with Newton's method.

    Minimizes the same objective as reg_logistic_regression in
    implementations.py, (1/N) * NLL + lambda_ * ||w||^2 (bias not penalized),
    but each step solves H step = grad, which converges in ~10 iterations
    instead of thousands of gradient steps.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), first column is the bias.
        lambda_: float, the regularization parameter.
        max_iters: int, maximum number of Newton steps.
        tol: float, stop when the largest step component is below tol.

    Returns:
        w: numpy array of shape (D,).
    """
    n, d = tx.shape
    reg = 2 * lambda_ * np.eye(d)
    reg[0, 0] = 1e-8  # keeps H invertible without shrinking the bias
    w = np.zeros(d)
    for _ in range(max_iters):
        p = sigmoid(tx @ w)
        # Same gradient as compute_logistic_gradient, plus the penalty term.
        gradient = tx.T @ (p - y) / n + reg @ w
        # H = (1/N) X^T S X + 2 lambda I, with S = diag(p_n (1 - p_n)).
        # Scaling the rows of tx avoids building the N x N matrix S.
        hessian = (tx * (p * (1 - p))[:, None]).T @ tx / n + reg
        step = np.linalg.solve(hessian, gradient)
        w = w - step
        if np.abs(step).max() < tol:
            break
    return w


def f1_score(y, pred):
    """F1 score of boolean predictions pred against labels y in {0, 1}."""
    # F1 = 2 TP / (2 TP + FP + FN) = 2 TP / (#predicted pos + #actual pos)
    tp = np.sum(pred & (y == 1))
    return 2 * tp / (pred.sum() + y.sum())


def best_threshold(y, scores):
    """Return the probability threshold maximizing F1 on (y, scores)."""
    thresholds = np.linspace(0.01, 0.99, 197)
    f1s = [f1_score(y, scores >= t) for t in thresholds]
    return thresholds[int(np.argmax(f1s))]


def fit(x, y, names):
    """Fit both models on raw features x and labels y in {0, 1}.

    Returns:
        model: the tuple (preprocessor, w, trees, kept columns for the trees).
    """
    preprocessor = Preprocessor().fit(x, names)
    tx = np.c_[np.ones(len(x)), preprocessor.transform(x)]
    w = reg_logistic_regression_newton(y, tx, LAMBDA)
    # The trees use the raw codes, minus the same uninformative columns.
    keep = [i for i, n in enumerate(names) if n not in DROP_COLUMNS]
    trees = GradientBoosting(seed=SEED).fit(x[:, keep], y)
    return preprocessor, w, trees, keep


def predict_proba(model, x):
    """Return the ensemble's P(y = 1) for every row of the raw features x."""
    preprocessor, w, trees, keep = model
    tx = np.c_[np.ones(len(x)), preprocessor.transform(x)]
    p_logistic = sigmoid(tx @ w)
    p_trees = trees.predict_proba(x[:, keep])
    p = (p_logistic + p_trees) / 2
    # Real numerical problems still stop the run (see the note in main).
    if not np.all(np.isfinite(p)):
        raise FloatingPointError("non-finite predicted probabilities")
    return p


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    names = load_column_names(f"{DATA_PATH}/x_train.csv")
    # Labels come as {-1, 1}; logistic regression needs {0, 1}.
    y = (y_train == 1).astype(float)

    # Out-of-fold predictions to pick the threshold and estimate test F1.
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(y)), N_FOLDS)
    oof = np.empty(len(y))
    # Each sample is predicted by a model that never saw it. The
    # preprocessor is refitted per fold, so no validation statistics leak.
    for k, val in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != k])
        oof[val] = predict_proba(fit(x_train[train], y[train], names), x_train[val])
        print(f"fold {k + 1}/{N_FOLDS} done")

    # One threshold for all folds, then the per-fold F1 at that threshold.
    threshold = best_threshold(y, oof)
    fold_f1 = [f1_score(y[val], oof[val] >= threshold) for val in folds]
    print(f"threshold = {threshold:.3f}")
    print(f"CV F1 = {np.mean(fold_f1):.4f} +/- {np.std(fold_f1):.4f}")
    print(f"CV accuracy = {np.mean((oof >= threshold) == (y == 1)):.4f}")

    # Final model on all training data, thresholded as chosen above.
    model = fit(x_train, y, names)
    y_pred = np.where(predict_proba(model, x_test) >= threshold, 1, -1)
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print(f"wrote submission.csv ({np.mean(y_pred == 1):.1%} predicted positive)")


if __name__ == "__main__":
    # NumPy 2.0 on macOS (Apple Accelerate BLAS) emits spurious
    # "divide by zero / overflow / invalid value encountered in matmul"
    # warnings even for small, finite inputs. They do not affect the results,
    # so they are silenced here; predict_proba checks that outputs are finite.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        main()
