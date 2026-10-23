# CS-433 Project 1

Heart-attack prediction for the EPFL Machine Learning course (Fall 2026). The graded code lives in `implementations.py`.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

Check that the install worked:

```bash
python -c "import numpy; print(numpy.__version__)"
```

## Libraries

The six methods in `implementations.py` need NumPy and the Python standard library. `import numpy as np` is already in that file.

## Data

Keep these files in `dataset/` and do not rename them:

- `x_train.csv`
- `y_train.csv`
- `x_test.csv`

## Code

`implementations.py` must define:

- `mean_squared_error_gd`
- `mean_squared_error_sgd`
- `least_squares`
- `ridge_regression`
- `logistic_regression`
- `reg_logistic_regression`

Each function returns `(w, loss)`.

## Reproducing the submission

```bash
python run.py        # writes submission.csv
python run.py --cv   # also prints the 5-fold cross-validated F1
```

The first run parses the CSVs (~2 min) and caches them as `.npy` files in `dataset/`; later runs take ~2 min. The pipeline is deterministic.

## Files

| File | Content |
|---|---|
| `implementations.py` | The six required methods (graded). |
| `helpers.py` | Course helpers (CSV loading/submission) and loss/gradient functions. |
| `preprocessing.py` | Data loading with cache, fixed cleaning rules (`clean_raw`) and the learned `Preprocessor`. |
| `model_selection.py` | F1/precision/recall, stratified k-fold, F1-optimal threshold, `cross_validate`. |
| `models.py` | Class-weighted regularized logistic regression (Newton's method) and wrappers around the required methods. |
| `run.py` | Final pipeline producing `submission.csv`. |
| `experiments.py` | Baselines, ablation study and hyperparameter search; writes `results/experiments.csv`. |
| `exploration.ipynb` | Exploratory analysis and figures. |

## Pipeline

1. **Cleaning (`clean_raw`)**: drop 36 survey-administration columns (dates, sampling weights, phone screening) and 16 raw answers that mix units (e.g. `ALCDAY5`, `WEIGHT2`), whose cleaned computed versions (`_DRNKWEK`, `WTKG3`, ...) are kept. BRFSS codes for "don't know"/"refused" (7/9, 77/99, 777/999, ...) become missing and "none" codes (88, 888) become 0.
2. **Preprocessor** (fitted on training folds only): drop columns with more than 99% missing values (the indicators of mostly-empty optional modules carry signal); one-hot encode columns with at most 10 distinct values (most frequent level as reference) and `_STATE`; for the other columns apply `log1p` if non-negative with skewness > 2, impute the median and standardize; add one missingness indicator per distinct missingness pattern; add a bias column.
3. **Model**: L2-regularized logistic regression (lambda = 1e-4, bias not penalized) with positive samples weighted 3x, trained with Newton's method.
4. **Threshold**: chosen to maximize F1 on the out-of-fold scores of a stratified 5-fold cross-validation (seed 1).

Labels are mapped from {-1, 1} to {0, 1} for training and back for the submission.
