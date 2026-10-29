"""Produce the competition submission.

    python run.py          # train on all data, write submission.csv
    python run.py --cv     # also print a 5-fold cross-validated F1 estimate

Pipeline:
  1. Load data (cached as .npy in dataset/ after the first run).
  2. clean_raw: drop admin/encoded columns, BRFSS missing codes -> NaN.
  3. Preprocessor: drop >99% missing columns, one-hot categoricals,
     log1p + median-impute + standardize continuous features, add
     missingness indicators and a bias column.
  4. Class-weighted L2 logistic regression trained with Newton's method.
  5. Decision threshold chosen to maximize F1 on 5-fold out-of-fold scores.
  6. Predict test set, write submission.csv with labels in {-1, 1}.

Everything is deterministic (fixed fold seed, Newton's method has no
randomness), so re-running reproduces the same submission.
"""

import sys

import numpy as np

from helpers import create_csv_submission
from model_selection import cross_validate
from models import logistic_newton
from preprocessing import Preprocessor, clean_raw, load_data, CROSS_COLS

DATA_PATH = "dataset"
OUTPUT = "submission.csv"
SEED = 1
K_FOLD = 5

# Hyperparameters (chosen by cross-validation, see experiments.py).
PREPROCESSOR = dict(
    max_nan_frac=1.00,
    max_categories=10,
    log_skewed=True,
    missing_indicators=True,
    poly_degree=4,
    cross_cols=CROSS_COLS,
)
LAMBDA = 1e-5
POS_WEIGHT = 3.0


def main(report_cv=False):
    x_train, x_test, y_train, _, test_ids, columns = load_data(DATA_PATH)
    y = (y_train == 1).astype(int)  # {-1, 1} -> {0, 1}

    # Cleaning rules depend only on which codes appear in a column; computing
    # them on train+test together guarantees identical decisions on both.
    x_all, columns = clean_raw(np.vstack([x_train, x_test]), columns)
    x_train, x_test = x_all[: len(y)], x_all[len(y) :]
    del x_all

    make_prep = lambda: Preprocessor(**PREPROCESSOR)
    fit = logistic_newton(lambda_=LAMBDA, pos_weight=POS_WEIGHT)

    # The threshold is always chosen by CV; the metrics are printed on demand.
    print(f"{K_FOLD}-fold cross-validation...")
    cv = cross_validate(
        x_train, y, columns, make_prep, fit, K_FOLD, SEED, verbose=report_cv
    )
    threshold = cv["threshold"]
    if report_cv:
        for key in ["f1", "precision", "recall", "accuracy"]:
            print(f"  {key:9s} {cv[key + '_mean']:.4f} +- {cv[key + '_std']:.4f}")
    print(f"  threshold on score: {threshold:.4f}")

    print("Training final model on all training data...")
    prep = make_prep().fit(x_train, columns)
    score_fn = fit(prep.transform(x_train), y)
    scores = score_fn(prep.transform(x_test))
    y_pred = np.where(scores >= threshold, 1, -1)

    create_csv_submission(test_ids, y_pred, OUTPUT)
    print(f"Wrote {OUTPUT}: {np.mean(y_pred == 1):.1%} predicted positive")


if __name__ == "__main__":
    main(report_cv="--cv" in sys.argv)
