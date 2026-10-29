"""Baselines, ablation study and hyperparameter search (for the report).

    python experiments.py

Every configuration is evaluated with the same stratified 5-fold split
(seed 1). Preprocessing is re-fitted inside each training fold. For each
configuration we report:
  - F1 (mean +- std over folds) at the F1-optimal threshold,
  - F1 at the "default" threshold (score >= 0, i.e. probability >= 0.5 for
    logistic models, prediction >= 0.5 for linear regression on {0, 1}),
  - precision, recall, accuracy at the tuned threshold.
Results are printed and saved to results/experiments.csv.
"""

import csv
import os
import time

import numpy as np

from model_selection import classification_metrics, cross_validate
from models import linear_least_squares, linear_ridge, logistic_gd, logistic_newton
from preprocessing import Preprocessor, clean_raw, load_data, CROSS_COLS
from run import LAMBDA, POS_WEIGHT, PREPROCESSOR

K_FOLD, SEED = 5, 1


def main():
    x_train, x_test, y_train, _, _, raw_columns = load_data("dataset")
    y = (y_train == 1).astype(int)
    x_all, columns = clean_raw(np.vstack([x_train, x_test]), raw_columns)
    x_clean = x_all[: len(y)].copy()
    del x_all, x_test, x_train
    # The raw matrix is only needed for one ablation; memory-map it from the
    # .npy cache instead of holding 840 MB in RAM for the whole run.
    x_train = np.load("dataset/x_train.npy", mmap_mode="r")

    full = PREPROCESSOR
    prep = lambda **kw: (lambda: Preprocessor(**{**full, **kw}))
    best = logistic_newton(LAMBDA, POS_WEIGHT)

    # (name, x, columns, preprocessor factory, fit function, default threshold)
    configs = [
        # --- Baselines: required methods on the full preprocessing ---------
        (
            "least squares",
            x_clean,
            columns,
            prep(poly_degree=1, cross_cols=None),
            linear_least_squares(),
            0.5,
        ),
        (
            "ridge (lambda=1e-5)",
            x_clean,
            columns,
            prep(poly_degree=1, cross_cols=None),
            linear_ridge(1e-5),
            0.5,
        ),
        (
            "reg. logistic GD (300 it, gamma=0.5)",
            x_clean,
            columns,
            prep(poly_degree=1, cross_cols=None),
            logistic_gd(LAMBDA, 0.5, 300),
            0.0,
        ),
        # --- Final model ---------------------------------------------------
        (
            "FINAL: weighted logistic Newton",
            x_clean,
            columns,
            prep(poly_degree=1, cross_cols=None),
            best,
            0.0,
        ),
        # --- Ablation: remove one component at a time ----------------------
        (
            "- no class weighting (pos_weight=1)",
            x_clean,
            columns,
            prep(poly_degree=1, cross_cols=None),
            logistic_newton(LAMBDA, 1.0),
            0.0,
        ),
        (
            "- no log1p of skewed features",
            x_clean,
            columns,
            prep(log_skewed=False, poly_degree=1, cross_cols=None),
            best,
            0.0,
        ),
        (
            "- no missingness indicators",
            x_clean,
            columns,
            prep(missing_indicators=False, poly_degree=1, cross_cols=None),
            best,
            0.0,
        ),
        (
            "- no one-hot (all continuous)",
            x_clean,
            columns,
            prep(max_categories=0, poly_degree=1, cross_cols=None),
            best,
            0.0,
        ),
        (
            "- no clean_raw (all 321 raw columns)",
            x_train,
            raw_columns,
            prep(),
            best,
            0.0,
        ),
        (
            "- keep columns <= 50% missing",
            x_clean,
            columns,
            prep(max_nan_frac=0.5, poly_degree=1, cross_cols=None),
            best,
            0.0,
        ),
        (
            "- keep columns <= 80% missing",
            x_clean,
            columns,
            prep(max_nan_frac=0.8, poly_degree=1, cross_cols=None),
            best,
            0.0,
        ),
    ]
    # --- Hyperparameters ----------------------------------------------------
    for lam in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        configs.append(
            (
                f"lambda={lam:g}",
                x_clean,
                columns,
                prep(poly_degree=1, cross_cols=None),
                logistic_newton(lam, POS_WEIGHT),
                0.0,
            )
        )

    for deg in [2, 3, 4]:
        configs.append(
            (
                f"Polynomial deg={deg}, cross_cols=None",
                x_clean,
                columns,
                prep(cross_cols=None, poly_degree=deg),
                best,
                0,
            )
        )
    configs.append(
        (
            f"Polynomial deg=4, cross_cols hand selected",
            x_clean,
            columns,
            prep(cross_cols=CROSS_COLS, poly_degree=4),
            best,
            0,
        )
    )

    os.makedirs("results", exist_ok=True)
    rows = []
    header = f"{'configuration':42s} {'F1 tuned':>15s} {'F1 @default':>11s} {'prec':>6s} {'recall':>6s} {'acc':>6s} {'time':>6s}"
    print(header)
    for name, x, cols, make_prep, fit, default_t in configs:
        t0 = time.time()
        cv = cross_validate(x, y, cols, make_prep, fit, K_FOLD, SEED, verbose=False)
        f1_default = classification_metrics(
            y, (cv["oof_scores"] >= default_t).astype(int)
        )["f1"]
        row = {
            "configuration": name,
            "f1_mean": cv["f1_mean"],
            "f1_std": cv["f1_std"],
            "f1_default_threshold": f1_default,
            "precision": cv["precision_mean"],
            "recall": cv["recall_mean"],
            "accuracy": cv["accuracy_mean"],
            "seconds": time.time() - t0,
        }
        rows.append(row)
        print(
            f"{name:42s} {row['f1_mean']:.4f} +- {row['f1_std']:.4f} "
            f"{f1_default:11.4f} {row['precision']:6.3f} {row['recall']:6.3f} "
            f"{row['accuracy']:6.3f} {row['seconds']:5.0f}s",
            flush=True,
        )
        with open("results/experiments.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
